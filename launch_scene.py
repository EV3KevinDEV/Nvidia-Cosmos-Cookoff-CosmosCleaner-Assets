
from isaacsim import SimulationApp
import os
import sys
import numpy as np
import io
import threading
import time
import datetime
import cv2
from flask import Flask, render_template, request, jsonify, Response, send_file

# Start the simulation app
simulation_app = SimulationApp({"headless": False})

import itertools
import carb
from pxr import Usd, UsdGeom, UsdPhysics, Gf
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController

# ══════════════════════════════════════════════════════════════════════════════
# LLM NAVIGATION — Cosmos Reason2 Bridge
# ══════════════════════════════════════════════════════════════════════════════

LLM_ENABLED    = True
BRIDGE_URL     = os.environ.get("BRIDGE_URL",     "http://localhost:8080/v1/reason2/action")
BRIDGE_API_KEY = os.environ.get("BRIDGE_API_KEY", "super-secret-key")
LLM_INTERVAL   = 30      # call bridge every N sim steps (~0.5 s at 60 Hz)
LLM_TIMEOUT    = 30.0    # seconds before giving up on a response

# System prompt is configured server-side — no local prompt needed.

import base64, json as _json
try:
    import requests as _requests
    from PIL import Image as _PilImage
    from io import BytesIO as _BytesIO
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

class LLMNavigator:
    """Sends a camera frame + odometry to the Cosmos Reason2 Bridge and converts
    its JSON response into (linear m/s, angular rad/s) for DifferentialController.

    Bridge API:
      POST BRIDGE_URL
      Headers: Content-Type: application/json, X-API-Key: BRIDGE_API_KEY
      Body:    {"image_b64": <PNG base64>, "instruction": <odometry text>}
      Returns: {"turn_degrees": float, "move_meters": float, "scene_analysis": str, ...}

    Velocity mapping:
      move_meters  → linear  m/s  (used as direct speed, capped at ±1.0 m/s)
      turn_degrees → angular rad/s (capped at ±2.5 rad/s; deadband < 2° = 0)
    The robot holds this velocity until the next LLM response arrives.
    """

    _TURN_DEADBAND_DEG = 2.0   # ignore jitter below this many degrees
    _MAX_LINEAR        = 1.0   # m/s cap
    _MAX_ANGULAR       = 2.5   # rad/s cap
    _BRIDGE_IMG_W      = 640   # resize width  before sending to bridge
    _BRIDGE_IMG_H      = 400   # resize height before sending to bridge

    def __init__(self):
        self.enabled       = LLM_ENABLED
        self.last_cmd      = {"turn_degrees": 0.0, "move_meters": 0.0}
        self.last_scene    = ""
        self.last_obstacle = False
        self.last_raw      = ""
        self.error         = ""
        self._lock         = threading.Lock()
        self._pending      = False   # True while a request is in-flight
        self._cmd_history  = []      # list of (turn_deg, move_m) — last 8 commands
        self._cmd_seq      = 0       # incremented each time a fresh response arrives

    def _to_png_b64(self, jpeg_bytes: bytes) -> str:
        """Convert JPEG bytes → resize to 640×400 → PNG → base64 ASCII.
        Downsizing to half-resolution cuts the bridge payload by ~75%,
        significantly reducing inference latency."""
        with _PilImage.open(_BytesIO(jpeg_bytes)) as im:
            im = im.convert("RGB")
            im = im.resize((self._BRIDGE_IMG_W, self._BRIDGE_IMG_H),
                           _PilImage.LANCZOS)
            out = _BytesIO()
            im.save(out, format="PNG")
            return base64.b64encode(out.getvalue()).decode("ascii")

    def _build_instruction(self, odometry: dict) -> str:
        """Compact instruction string kept under the 2000-char bridge limit."""
        yaw_deg   = float(np.degrees(odometry["yaw"]))
        lin_vel   = float(odometry.get("linear_vel", 0.0))
        dist_m    = float(odometry.get("distance",   0.0))
        sweep_row = int(odometry.get("sweep_row",    0))

        with self._lock:
            hist = list(self._cmd_history)

        # Compact history: last 4 entries only, terse format
        if hist:
            hist_str = " | ".join(f"t={t:+.0f} m={m:.2f}" for t, m in hist[-4:])
        else:
            hist_str = "none"

        # Idle streak: consecutive commands with both turn~0 and move~0
        idle_streak = 0
        for t, m in reversed(hist):
            if abs(t) < 2.0 and m < 0.02:
                idle_streak += 1
            else:
                break

        alerts = ""
        if lin_vel < 0.05 and odometry.get("was_driving", False):
            alerts += "ALERT:STALL(wall_contact) "
        if idle_streak >= 3:
            alerts += f"ALERT:IDLE_x{idle_streak}(all_zero_cmds,must_output_nonzero) "

        # Row parity tells LLM which sweep direction it is on
        row_dir = "left-to-right" if sweep_row % 2 == 0 else "right-to-left"

        instr = (
            "You are the autonomous navigation brain for a robot vacuum.\n"
            "TASK: Boustrophedon (lawnmower) floor sweep — long parallel rows, U-turn at each wall.\n\n"

            "=== TELEMETRY ===\n"
            f"Position: X={odometry['x']:.2f}, Y={odometry['y']:.2f}\n"
            f"Heading: {yaw_deg:.1f} deg\n"
            f"Velocity: {lin_vel:.2f} m/s\n"
            f"Sweep State: Row={sweep_row} (Direction: {row_dir})\n"
            f"History: {hist_str}\n"
            f"Active Alerts: {alerts if alerts else 'NONE'}\n\n"

            "=== HAZARD CLASSIFICATION (identify before acting) ===\n"
            "TYPE-A (ENTANGLEMENT) — cables, cords, wires, rope, string, charger, plug, shoelaces: "
            "STOP IMMEDIATELY. Hard turn 90° away. Never approach. This is the HIGHEST hazard.\n"
            "TYPE-B (STRUCTURAL) — walls, furniture legs, boxes, table legs, baseboards: "
            "Turn 45-90° away when within 0.8m or filling >15% of image. move=0.\n"
            "TYPE-C (SOFT/SMALL) — toys, socks, paper, small debris: "
            "Gentle arc 20-45° away. move=0.\n"
            "TYPE-D (SURFACE CHANGE) — rug edges, carpet borders, steps: "
            "Turn 30-45° away. move=0.\n\n"

            "=== DECISION RULES (Strict Priority) ===\n"
            "1. ALERTS: STALL -> turn=120, move=0. IDLE -> turn=90, move=0.\n"
            "2. TRAPPED: image dark/black -> turn=90, move=0.\n"
            "3. TYPE-A visible ANYWHERE in image -> turn=90 or -90 (away), move=0.\n"
            "4. TYPE-B fills >15% OR within 0.8m -> turn=45/50/55 (away), move=0.\n"
            "5. Wall/baseboard in center 60% -> turn=90 or -90 (U-turn for row end), move=0.\n"
            "6. TYPE-C or TYPE-D visible -> turn=30/45, move=0.\n"
            "7. CLEAR FLOOR -> micro-correct heading, turn=-10 to +10, move=0.15-0.20.\n\n"

            "=== CONSTRAINTS ===\n"
            "- move: 0.0 to 0.20 only. Never negative.\n"
            "- turn: multiples of 5 only. "
            "Allowed: [-120,-90,-55,-50,-45,-20,-15,-10,-5,0,5,10,15,20,45,50,55,90,120].\n"
            "- SAFETY: Rules 1-6 always override Rule 7.\n\n"

            "Respond ONLY with raw JSON, no markdown:\n"
            "{\"scene_analysis\": \"<1 sentence max 15 words>\", "
            "\"obstacle_detected\": <bool>, "
            "\"turn_degrees\": <float mult of 5>, "
            "\"move_meters\": <float 0.0-0.20>}"
        )
        # Safety net: hard truncate to 1950 chars (bridge limit is 2000)
        return instr[:1950]

    def _parse_response(self, resp_json: dict, raw_text: str) -> dict:
        """Extract the action dict from whatever the bridge returns.
        Handles: direct fields, wrapped in 'result'/'action', or raw JSON string."""
        if "turn_degrees" in resp_json:
            return resp_json
        for key in ("result", "action", "output"):
            if key in resp_json:
                val = resp_json[key]
                return _json.loads(val) if isinstance(val, str) else val
        # Last resort: parse the raw response body as JSON
        clean = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return _json.loads(clean)

    def _call_bridge(self, jpeg_bytes: bytes, odometry: dict):
        """Blocking HTTP POST to the bridge — always run in a daemon thread."""
        if not _REQUESTS_OK:
            self.error = "'requests' or 'Pillow' package not installed"
            self._pending = False
            return
        try:
            image_b64   = self._to_png_b64(jpeg_bytes)
            instruction = self._build_instruction(odometry)
            # Hard guarantee: strip every non-ASCII character so the bridge
            # never gets a 422 from encoding issues
            instruction = instruction.encode("ascii", "ignore").decode("ascii")
            payload = {"image_b64": image_b64, "instruction": instruction}
            headers = {
                "Content-Type": "application/json",
                "X-API-Key":    BRIDGE_API_KEY,
            }
            resp = _requests.post(BRIDGE_URL, headers=headers,
                                  json=payload, timeout=LLM_TIMEOUT)
            if resp.status_code == 422:
                print(f"[LLM] 422 body: {resp.text[:400]}")
            resp.raise_for_status()
            parsed = self._parse_response(resp.json(), resp.text)
            raw_turn = float(parsed.get("turn_degrees", 0.0))
            # Round to nearest 5-degree increment for fine-grained boustrophedon control
            raw_turn = round(raw_turn / 5.0) * 5.0
            cmd = {
                "turn_degrees": float(np.clip(raw_turn,                            -180.0, 180.0)),
                # Never allow backward motion; small steps for cautious sweep
                "move_meters":  float(np.clip(parsed.get("move_meters",  0.0),    0.0,   0.26)),
            }
            scene_lower = parsed.get("scene_analysis", "").lower()

            # -- Client-side overrides ------------------------------------------
            # Minimal: trust the model. Only fix provable deadlock patterns.

            # 1) Dark/black image + no turn -> spin to escape
            if "dark" in scene_lower and "clear" not in scene_lower and abs(cmd["turn_degrees"]) < 5.0:
                cmd["turn_degrees"] = 90.0
                cmd["move_meters"]  = 0.0
                print("[LLM] override: dark image -> spin 90")

            # 2) Streak overrides
            with self._lock:
                hist_snap = list(self._cmd_history)
            # No-turn streak: only intervene after 15+ straight steps (lawnmower rows are long)
            streak = sum(1 for t, _ in reversed(hist_snap) if abs(t) < 5.0)
            if streak >= 15 and abs(cmd["turn_degrees"]) < 5.0:
                cmd["turn_degrees"] = 90.0
                print(f"[LLM] override: {streak}-step no-turn streak -> force 90 deg U-turn")
            # Full-idle streak: both turn and move ~zero 3+ times -> robot is stuck, spin
            idle_streak = sum(1 for t, m in reversed(hist_snap) if abs(t) < 2.0 and m < 0.02)
            if idle_streak >= 3 and abs(cmd["turn_degrees"]) < 2.0 and cmd["move_meters"] < 0.02:
                cmd["turn_degrees"] = 90.0
                print(f"[LLM] override: {idle_streak}-step full-idle stuck -> spin 90")
            with self._lock:
                self.last_cmd      = cmd
                self.last_scene    = parsed.get("scene_analysis", "")
                self.last_obstacle = bool(parsed.get("obstacle_detected", False))
                self.last_raw      = resp.text
                self.error         = ""
                # Keep rolling history of last 8 commands
                self._cmd_history.append((cmd["turn_degrees"], cmd["move_meters"]))
                if len(self._cmd_history) > 8:
                    self._cmd_history.pop(0)
                self._cmd_seq += 1  # signal that a fresh command is ready
            print(f"[LLM] turn={cmd['turn_degrees']:.1f}deg move={cmd['move_meters']:.2f}m  scene: {self.last_scene[:80]}")
        except Exception as exc:
            with self._lock:
                self.error = str(exc)
            print(f"[WARN] LLM bridge error: {exc}")
        finally:
            self._pending = False

    def step(self, jpeg_bytes: bytes, odometry: dict):
        """Non-blocking: fires a background thread if none is in-flight."""
        if not self.enabled or not jpeg_bytes:
            return
        if self._pending:
            return
        self._pending = True
        threading.Thread(target=self._call_bridge, args=(jpeg_bytes, odometry), daemon=True).start()

    def get_move_command(self) -> tuple[float, float]:
        """Convert the last bridge response to (linear m/s, angular rad/s).
        Values are used as continuous velocity commands held until the next response.
          move_meters  → linear  m/s  (direct, capped at ±_MAX_LINEAR)
          turn_degrees → angular rad/s (deg→rad, deadbanded, capped at ±_MAX_ANGULAR)
        """
        with self._lock:
            cmd = self.last_cmd
        turn = cmd["turn_degrees"]
        # Apply deadband to suppress sub-2° jitter that causes drift
        if abs(turn) < self._TURN_DEADBAND_DEG:
            turn = 0.0
        linear  = float(np.clip(cmd["move_meters"], -self._MAX_LINEAR,  self._MAX_LINEAR))
        angular = float(np.clip(np.deg2rad(turn),   -self._MAX_ANGULAR, self._MAX_ANGULAR))
        return linear, angular

class CosmosCleanerBotApp:
    def __init__(self):
        self.my_world = World(stage_units_in_meters=1.0)
        self.assets_root = os.getcwd()
        
        # Paths
        self.robot_usd = os.path.join(self.assets_root, "usd", "CosmosCleanerBot_Camera.usd")
        self.robot_prim_path = "/World/CosmosCleanerBot"
        
        # Initialize Scene
        self.setup_scene()
        
        # Find RGB camera prim in the USD hierarchy
        self.depth_cam_path = self._find_camera_prim("Camera_OmniVision_OV9782_Color")
        self.latest_depth_jpeg = b''
        
        # Robot Setup
        # Use the corrected DOF names and remove wheel_base from robot init (handled by controller)
        self.robot = WheeledRobot(
            prim_path=self.robot_prim_path,
            name="cosmos_cleaner_bot",
            wheel_dof_names=["Revolute_left", "Revolute_right"],
            create_robot=True
        )
        self.my_world.scene.add(self.robot)
        
        # Depth Camera Sensor
        self.depth_camera = None
        if self.depth_cam_path:
            try:
                from isaacsim.sensors.camera import Camera
                self.depth_camera = Camera(
                    prim_path=self.depth_cam_path,
                    resolution=(1280, 800),
                    frequency=20,
                    name="depth_cam",
                )
                self.my_world.scene.add(self.depth_camera)
                print(f"[INFO] RGB camera created at {self.depth_cam_path}")
            except Exception as e:
                print(f"[WARN] Could not create Camera sensor: {e}")
        
        # Controller
        self.controller = DifferentialController(name="diff_controller", wheel_radius=0.10, wheel_base=0.48)
        
        # State variables
        self.waypoints = []
        self.current_waypoint_idx = -1
        self.path_history = []  # [[x,y], ...] for web map trail
        self.current_pos = np.zeros(3)
        self.current_yaw = 0.0
        self.cmd_linear = 0.0
        self.cmd_angular = 0.0

        # Speed config (autonomous navigation)
        self.nav_speed = 0.52         # m/s for auto-nav straight drive (~30% faster)
        self.nav_angular_gain = 0.6   # gain for heading correction
        self.jpeg_quality = 80        # JPEG encode quality 20-95

        # Odometry extras
        self.distance_traveled = 0.0
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self._prev_pos2d = None
        self._prev_yaw = None
        self._vel_buf = []  # smoothing buffer: (lin, ang) tuples

        # Video recording
        self.recording = False
        self.video_writer = None
        self.recording_path = None
        self.recording_frames = 0
        self.recording_start_time = None
        self._recording_lock = threading.Lock()
        self._recordings_dir = os.path.join(self.assets_root, "recordings")
        os.makedirs(self._recordings_dir, exist_ok=True)
        self._dataset_dir = os.path.join(self.assets_root, "dataset")
        os.makedirs(self._dataset_dir, exist_ok=True)
        self._snapshot_count = 0

        # LLM navigation — Cosmos Reason2 Bridge
        self.llm_navigator = LLMNavigator()
        self.llm_mode      = False   # True = LLM drives; False = waypoint / manual
        # Phase-machine state for LLM execution
        # _llm_phase: None = idle, 'rotate' = turning in place, 'drive' = moving forward
        self._llm_phase      = None
        self._llm_target_yaw = 0.0   # heading to reach in rotate phase
        self._llm_move_m     = 0.0   # distance to cover in drive phase
        self._llm_origin     = None  # position snapshot when drive phase started
        self._llm_last_seq   = 0     # tracks which _cmd_seq we last consumed
        self._llm_stall_steps = 0    # steps where drive was commanded but robot didn't move
        self._sweep_row      = 0     # boustrophedon row counter (increments on U-turns)
        
        self.my_world.reset()
        
        # Initialize depth camera after world reset
        if self.depth_camera is not None:
            try:
                self.depth_camera.initialize()
                self.depth_camera.add_rgb_to_frame()
                self.my_world.play()  # Must be playing for camera render pipeline to produce data
                for _ in range(30):  # Warmup frames to prime the render pipeline
                    self.my_world.step(render=True)
                print("[INFO] RGB camera initialized and warmed up")
            except Exception as e:
                print(f"[WARN] RGB camera init failed: {e}")
                self.depth_camera = None

        # Auto-play so the sim runs immediately without needing to press Play in the GUI
        if not self.my_world.is_playing():
            self.my_world.play()
        
    def setup_scene(self):
        # ── Load Simple Room environment ──────────────────────────────────────
        self._isaac_assets_root = get_assets_root_path()
        if self._isaac_assets_root is None:
            carb.log_warn("[WARN] Isaac asset root not found — falling back to default ground plane.")
            self.my_world.scene.add_default_ground_plane()
        else:
            simple_room_usd = self._isaac_assets_root + "/Isaac/Environments/Simple_Room/simple_room.usd"
            print(f"[INFO] Loading Simple Room: {simple_room_usd}")
            add_reference_to_stage(usd_path=simple_room_usd, prim_path="/World/SimpleRoom")
            self._spawn_room_obstacles()

        # ── Load robot (spawn above floor so it settles without tipping) ────────
        import omni.usd
        from pxr import UsdGeom, Gf
        add_reference_to_stage(usd_path=self.robot_usd, prim_path=self.robot_prim_path)
        robot_prim = omni.usd.get_context().get_stage().GetPrimAtPath(self.robot_prim_path)
        if robot_prim.IsValid():
            xform = UsdGeom.Xformable(robot_prim)
            ops   = {op.GetOpName(): op for op in xform.GetOrderedXformOps()}
            # Place robot in a clear corner away from the centre table
            (ops.get("xformOp:translate") or xform.AddTranslateOp()).Set(Gf.Vec3d(-2.0, -2.0, 0.15))
        set_camera_view(eye=[5.0, 5.0, 4.5], target=[0.0, 0.0, 0.5])

    def _spawn_room_obstacles(self):
        """Spawn household/warehouse obstacle props into the Simple Room with
        rigid-body physics and convex-hull colliders, placed uniformly across
        the floor while avoiding the centre table zone."""
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        R   = self._isaac_assets_root
        WP  = R + "/Isaac/Environments/Simple_Warehouse/Props"
        YCB = R + "/Isaac/Props/YCB/Axis_Aligned_Physics"

        CATALOGUE = [
            # Cardboard boxes — large floor clutter
            ("BoxA_1",        WP  + "/SM_CardBoxA_01.usd",                           2.0,  None),
            ("BoxA_2",        WP  + "/SM_CardBoxA_01.usd",                           2.0,  None),
            ("BoxB_1",        WP  + "/SM_CardBoxB_01.usd",                           2.5,  None),
            ("BoxC_1",        WP  + "/SM_CardBoxC_01.usd",                           3.0,  None),
            ("BoxD_1",        WP  + "/SM_CardBoxD_01.usd",                           3.5,  None),
            # Storage bins
            ("KLT_1",         R   + "/Isaac/Props/KLT_Bin/small_KLT.usd",            0.8,  None),
            ("KLT_2",         R   + "/Isaac/Props/KLT_Bin/small_KLT.usd",            0.8,  None),
            # YCB household objects — realistic in-room debris
            ("CrackerBox",    YCB + "/003_cracker_box.usd",                           0.41, None),
            ("SoupCan",       YCB + "/005_tomato_soup_can.usd",                       0.35, None),
            ("PowerDrill",    YCB + "/035_power_drill.usd",                           0.90, None),
            ("MasterChefCan", YCB + "/002_master_chef_can.usd",                       0.41, None),
            ("SugarBox",      YCB + "/004_sugar_box.usd",                             0.51, None),
            ("MustardBottle", YCB + "/006_mustard_bottle.usd",                        0.60, None),
            ("TunaFishCan",   YCB + "/007_tuna_fish_can.usd",                         0.17, None),
            ("PuddingBox",    YCB + "/008_pudding_box.usd",                           0.19, None),
            ("Mug",           YCB + "/025_mug.usd",                                   0.12, None),
            ("WoodBlock",     YCB + "/036_wood_block.usd",                            0.73, None),
            ("TennisBall",    YCB + "/056_tennis_ball.usd",                           0.06, None),
            # Small toys
            ("RubiksCube",    R   + "/Isaac/Props/Rubiks_Cube/rubiks_cube.usd",       0.10, None),
            ("Block_1",       R   + "/Isaac/Props/Blocks/basic_block.usd",            0.30, (1,1,1)),
        ]

        # ── Stratified random layout ──────────────────────────────────────────
        # Room: 6.5 × 6.5 m → usable ±2.65 m after wall margin.
        # Robot spawns near origin, so exclude centre table zone (r < 1.5 m).
        # Also keep the robot spawn corner (-2.0, -2.0) clear with 0.8 m margin.
        ROOM_HALF  = 2.65
        TABLE_EXCL = 1.5
        ROBOT_X, ROBOT_Y = -2.0, -2.0   # robot spawns in corner
        DROP_Z     = 0.55
        OBS_MARGIN = 0.45
        ROBOT_MARGIN = 0.8

        obs_rng = np.random.default_rng(seed=42)
        TILES_X, TILES_Y = 4, 3
        tile_w = (2 * ROOM_HALF) / TILES_X
        tile_h = (2 * ROOM_HALF) / TILES_Y
        tile_order = [(c, r) for r in range(TILES_Y) for c in range(TILES_X)]
        obs_rng.shuffle(tile_order)
        tile_cycle = itertools.cycle(tile_order)

        def pick_pose(placed):
            for _ in range(len(tile_order)):
                tc, tr = next(tile_cycle)
                tx0 = -ROOM_HALF + tc * tile_w
                ty0 = -ROOM_HALF + tr * tile_h
                for _ in range(14):
                    x = obs_rng.uniform(tx0 + 0.1, tx0 + tile_w - 0.1)
                    y = obs_rng.uniform(ty0 + 0.1, ty0 + tile_h - 0.1)
                    if np.hypot(x, y) < TABLE_EXCL:
                        continue
                    if np.hypot(x - ROBOT_X, y - ROBOT_Y) < ROBOT_MARGIN:
                        continue
                    if all(np.hypot(x - ex, y - ey) >= OBS_MARGIN for ex, ey in placed):
                        return x, y, DROP_Z
            # fallback
            for _ in range(500):
                x = obs_rng.uniform(-ROOM_HALF, ROOM_HALF)
                y = obs_rng.uniform(-ROOM_HALF, ROOM_HALF)
                if np.hypot(x, y) >= TABLE_EXCL and np.hypot(x - ROBOT_X, y - ROBOT_Y) >= ROBOT_MARGIN:
                    if all(np.hypot(x - ex, y - ey) >= OBS_MARGIN for ex, ey in placed):
                        return x, y, DROP_Z
            return obs_rng.uniform(-ROOM_HALF, ROOM_HALF), obs_rng.uniform(-ROOM_HALF, ROOM_HALF), DROP_Z

        placed_xy = []
        print("[INFO] Spawning room obstacles…")
        for name, usd_path, mass_kg, scale in CATALOGUE:
            x, y, z = pick_pose(placed_xy)
            placed_xy.append((x, y))
            yaw_deg = float(obs_rng.uniform(0, 360))
            prim_path = f"/World/Obstacles/{name}"
            add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                print(f"[WARN]   ! {name} — prim not valid, skipping")
                continue
            # Position + yaw
            xform = UsdGeom.Xformable(prim)
            ops   = {op.GetOpName(): op for op in xform.GetOrderedXformOps()}
            (ops.get("xformOp:translate") or xform.AddTranslateOp()).Set(Gf.Vec3d(x, y, z))
            (ops.get("xformOp:rotateXYZ") or xform.AddRotateXYZOp()).Set(Gf.Vec3f(0.0, 0.0, yaw_deg))
            if scale is not None:
                (ops.get("xformOp:scale") or xform.AddScaleOp()).Set(Gf.Vec3f(*scale))
            # Rigid body + mass
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI.Apply(prim)
            if not prim.HasAPI(UsdPhysics.MassAPI):
                UsdPhysics.MassAPI.Apply(prim).GetMassAttr().Set(mass_kg)
            # Convex colliders on all mesh descendants
            for desc in Usd.PrimRange(prim):
                tn = desc.GetTypeName()
                if tn not in ("Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "Cone"):
                    continue
                if not desc.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(desc)
                if tn == "Mesh" and not desc.HasAPI(UsdPhysics.MeshCollisionAPI):
                    UsdPhysics.MeshCollisionAPI.Apply(desc).GetApproximationAttr().Set("convexHull")
            # Register with Isaac Sim scene
            try:
                self.my_world.scene.add(SingleRigidPrim(prim_path=prim_path, name=name))
                print(f"[INFO]   + {name} at ({x:.2f}, {y:.2f})")
            except Exception as e:
                print(f"[WARN]   ! {name} skipped ({e})")
                stage.RemovePrim(prim_path)

    def _find_camera_prim(self, name):
        """Search the USD stage for a camera prim by name."""
        import omni.usd
        from pxr import UsdGeom
        stage = omni.usd.get_context().get_stage()
        for prim in stage.Traverse():
            if prim.GetName() == name and prim.IsA(UsdGeom.Camera):
                path = str(prim.GetPath())
                print(f"[INFO] Found camera '{name}' at {path}")
                return path
        print(f"[WARN] Camera prim '{name}' not found in stage")
        return None

    def _capture_depth(self):
        """Capture RGB frame, encode as JPEG, and optionally write to video."""
        if self.depth_camera is None:
            return
        try:
            frame = self.depth_camera.get_current_frame()
            rgba = frame.get("rgb", None)
            if rgba is not None and rgba.size > 0:
                rgb = rgba[:, :, :3].astype(np.uint8)
                from PIL import Image
                img = Image.fromarray(rgb, mode='RGB')
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=self.jpeg_quality)
                self.latest_depth_jpeg = buf.getvalue()
                # Write frame to video if recording
                if self.recording:
                    with self._recording_lock:
                        if self.video_writer is not None:
                            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                            self.video_writer.write(bgr)
                            self.recording_frames += 1
        except Exception as e:
            if not getattr(self, '_depth_warned', False):
                print(f"[WARN] RGB capture error: {e}")
                self._depth_warned = True

    def start_recording(self):
        """Start video recording to a timestamped MP4 file."""
        if self.recording:
            return False
        ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"cosmos_{ts}.mp4"
        fpath = os.path.join(self._recordings_dir, fname)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # _capture_depth runs every 2 sim steps at ~60 Hz -> ~30 fps
        self.video_writer = cv2.VideoWriter(fpath, fourcc, 30, (1280, 800))
        if not self.video_writer.isOpened():
            print(f"[WARN] VideoWriter failed to open {fpath}")
            self.video_writer = None
            return False
        self.recording        = True
        self.recording_path   = fpath
        self.recording_frames = 0
        self.recording_start_time = time.time()
        print(f"[INFO] Recording started: {fname}")
        return True

    def stop_recording(self):
        """Stop recording and finalize the MP4. Returns filename or None."""
        if not self.recording:
            return None
        self.recording = False
        with self._recording_lock:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
        fname   = os.path.basename(self.recording_path)
        elapsed = time.time() - self.recording_start_time
        size_mb = os.path.getsize(self.recording_path) / 1e6
        print(f"[INFO] Recording saved: {fname}  ({self.recording_frames} frames, {elapsed:.1f}s, {size_mb:.1f} MB)")
        return fname

    def take_snapshot(self, label: str = "") -> dict:
        """Save the current camera frame to the dataset folder as a JPEG.
        Returns {filename, size, count} or raises RuntimeError."""
        jpeg = self.latest_depth_jpeg
        if not jpeg:
            raise RuntimeError("No camera frame available yet")
        self._snapshot_count += 1
        ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        n     = f"{self._snapshot_count:04d}"
        if label:
            safe  = "".join(c if c.isalnum() or c in '-_' else '_' for c in label)[:40].strip('_')
            fname = f"snap_{n}_{ts}_{safe}.jpg"
        else:
            fname = f"snap_{n}_{ts}.jpg"
        fpath = os.path.join(self._dataset_dir, fname)
        with open(fpath, "wb") as fh:
            fh.write(jpeg)
        print(f"[DATASET] saved {fname} ({len(jpeg)//1024} KB)")
        return {"filename": fname, "size": len(jpeg), "count": self._snapshot_count}

    def move_robot(self, linear, angular):
        # Negate linear only: the robot's physical forward direction is reversed.
        # Angular is correct (positive = left turn), so leave it as-is.
        wheel_actions = self.controller.forward(command=np.array([linear, angular]))
        # Revolute_right (index 1) has a flipped joint axis
        if wheel_actions.joint_velocities is not None:
            wheel_actions.joint_velocities[1] = -wheel_actions.joint_velocities[1]
        self.robot.apply_wheel_actions(wheel_actions)

    def llm_navigate_step(self):
        """Phase-machine LLM navigation.
        Phases: None (idle) -> 'rotate' (turn in place) -> 'drive' (forward only)
        A move=0 command NEVER generates forward motion.
        """
        nav = self.llm_navigator

        def _odom():
            return {
                "x":           float(self.current_pos[0]),
                "y":           float(self.current_pos[1]),
                "yaw":         float(self.current_yaw),
                "linear_vel":  float(self.linear_vel),
                "angular_vel": float(self.angular_vel),
                "distance":    float(self.distance_traveled),
                "was_driving": self._llm_phase == 'drive',
                "sweep_row":   self._sweep_row,
            }

        def _request_next():
            """Fire a new bridge request if nothing is in flight."""
            if not nav._pending:
                nav.step(self.latest_depth_jpeg, _odom())

        def _angle_diff(target, current):
            return ((target - current) + np.pi) % (2 * np.pi) - np.pi

        # ── IDLE: no active phase, consume fresh response or request one ────
        if self._llm_phase is None:
            with nav._lock:
                fresh = nav._cmd_seq > self._llm_last_seq

            if nav._pending:
                self.move_robot(0, 0)
                return

            if not fresh:
                # Nothing new yet — request a query and hold still
                self.move_robot(0, 0)
                _request_next()
                return

            # Consume the fresh command
            self._llm_last_seq = nav._cmd_seq
            with nav._lock:
                cmd = dict(nav.last_cmd)

            turn_deg = cmd["turn_degrees"]
            move_m   = cmd["move_meters"]

            if abs(turn_deg) < 2.0 and move_m < 0.02:
                # Effectively idle — request a new command immediately
                self.move_robot(0, 0)
                _request_next()
                return

            # Track U-turns to count completed rows (boustrophedon)
            if abs(turn_deg) >= 80.0:
                self._sweep_row += 1
                print(f"[NAV] sweep row {self._sweep_row} (U-turn {turn_deg:+.0f} deg)")

            # Right = clockwise = subtract from yaw
            self._llm_target_yaw = self.current_yaw - np.deg2rad(turn_deg)
            self._llm_move_m     = move_m

            if abs(turn_deg) >= 2.0:
                self._llm_phase = 'rotate'
            else:
                # No turn needed, go straight to drive
                self._llm_phase = 'drive'
                self._llm_origin = self.current_pos[:2].copy()
            return  # will execute on next step

        # ── ROTATE: spin in place until heading reached ──────────────────────
        if self._llm_phase == 'rotate':
            diff = _angle_diff(self._llm_target_yaw, self.current_yaw)
            if abs(diff) < 0.05:   # ~3 deg threshold
                # Heading reached — transition
                if self._llm_move_m >= 0.02:
                    self._llm_phase  = 'drive'
                    self._llm_origin = self.current_pos[:2].copy()
                else:
                    # Pure turn command complete — request next
                    self._llm_phase = None
                    self.move_robot(0, 0)
                    _request_next()
            else:
                # Proportional angular, no linear
                self.move_robot(0, np.clip(7.8 * diff, -4.5, 4.5))
            return

        # -- DRIVE: move forward until distance covered ----------------------
        if self._llm_phase == 'drive':
            traveled = np.linalg.norm(self.current_pos[:2] - self._llm_origin)
            if traveled >= self._llm_move_m - 0.05:
                # Distance covered -- done, request next command
                self._llm_phase = None
                self._llm_stall_steps = 0
                self.move_robot(0, 0)
                _request_next()
            else:
                # Check for stall: robot commanded to move but linear_vel is tiny
                if self.linear_vel < 0.03:
                    self._llm_stall_steps += 1
                else:
                    self._llm_stall_steps = 0

                if self._llm_stall_steps >= 12:
                    # Wall contact -- abort drive, spin 120 deg to escape
                    print(f"[NAV] stall detected after {self._llm_stall_steps} steps -> escape spin")
                    self._llm_stall_steps = 0
                    self._llm_phase = 'rotate'
                    self._llm_target_yaw = self.current_yaw - np.deg2rad(120.0)
                    self._llm_move_m = 0.0  # pure rotate, no drive after
                else:
                    # Drive forward with heading correction
                    heading_err = _angle_diff(self._llm_target_yaw, self.current_yaw)
                    self.move_robot(self.nav_speed, np.clip(3.9 * heading_err, -2.0, 2.0))
            return

    def update_state(self):
        pos, ori = self.robot.get_world_pose()
        self.current_pos = pos
        w, x, y, z = ori
        raw_yaw = np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
        # The robot's physical drive direction is +90° from the quaternion yaw.
        # Offset so navigate_step steers toward the actual physical forward.
        self.current_yaw = raw_yaw + np.pi / 2
        
        # Distance & velocity (smoothed over last 10 samples)
        pos2d = pos[:2].copy()
        if self._prev_pos2d is not None:
            d = np.linalg.norm(pos2d - self._prev_pos2d)
            self.distance_traveled += d
            yaw_d = self.current_yaw - self._prev_yaw
            yaw_d = (yaw_d + np.pi) % (2 * np.pi) - np.pi
            self._vel_buf.append((d * 60.0, yaw_d * 60.0))
            if len(self._vel_buf) > 10:
                self._vel_buf.pop(0)
            self.linear_vel  = float(np.mean([v[0] for v in self._vel_buf]))
            self.angular_vel = float(np.mean([v[1] for v in self._vel_buf]))
        self._prev_pos2d = pos2d
        self._prev_yaw   = self.current_yaw

        step_idx = int(self.my_world.current_time_step_index)
        # Record path history for the web map (every 10 steps)
        if step_idx % 10 == 0:
            self.path_history.append([float(pos[0]), float(pos[1])])
        # Capture depth camera frame (every 2 steps ~30 fps)
        if step_idx % 2 == 0:
            self._capture_depth()

    def navigate_step(self):
        # LLM mode overrides waypoint/manual when enabled:
        if getattr(self, 'llm_mode', False):
            self.llm_navigate_step()
            return

        if self.current_waypoint_idx < 0:
            self.move_robot(self.cmd_linear, self.cmd_angular)
            return
            
        target = self.waypoints[self.current_waypoint_idx]
        dist = np.linalg.norm(target[:2] - self.current_pos[:2])
        target_vector = target[:2] - self.current_pos[:2]
        target_angle = np.arctan2(target_vector[1], target_vector[0])
        # Use +π shift so a 0-degree difference gives 0, not -π
        angle_diff = ((target_angle - self.current_yaw) + np.pi) % (2 * np.pi) - np.pi
        
        if dist < 0.15:
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(self.waypoints):
                self.current_waypoint_idx = -1
            self.move_robot(0, 0)
        elif abs(angle_diff) > 0.15:
            # Phase 1: rotate in place to face the target
            self.move_robot(0, 2.0 * np.clip(angle_diff, -1.5, 1.5))
        else:
            # Phase 2: drive straight with small heading corrections
            self.move_robot(self.nav_speed, self.nav_angular_gain * angle_diff)

    def run(self):
        while simulation_app.is_running():
            self.my_world.step(render=True)
            if self.my_world.is_playing():
                self.update_state()
                self.navigate_step()
        simulation_app.close()

# Flask Web Server logic
# Instance creation must happen after SimulationApp is initialized
app_instance = CosmosCleanerBotApp()
flask_app = Flask(__name__)

@flask_app.route('/')
def index():
    return render_template('index.html')

@flask_app.route('/move', methods=['POST'])
def move():
    data = request.json
    app_instance.cmd_linear = data.get('linear', 0)
    app_instance.cmd_angular = data.get('angular', 0)
    app_instance.current_waypoint_idx = -1 # Override mission on manual move
    return jsonify(success=True)

@flask_app.route('/status')
def status():
    return jsonify({
        "x": float(app_instance.current_pos[0]),
        "y": float(app_instance.current_pos[1]),
        "yaw": float(app_instance.current_yaw),
        "waypoint_count": len(app_instance.waypoints)
    })

@flask_app.route('/add_waypoint', methods=['POST'])
def add_waypoint():
    data = request.json
    if data and 'x' in data and 'y' in data:
        # Waypoint placed from the map canvas
        pos = np.array([data['x'], data['y'], 0.0])
    else:
        # Waypoint at current robot position
        pos = app_instance.current_pos.copy()
    app_instance.waypoints.append(pos)
    return jsonify(success=True)

@flask_app.route('/clear_waypoints', methods=['POST'])
def clear_waypoints():
    app_instance.waypoints = []
    app_instance.current_waypoint_idx = -1
    app_instance.path_history = []
    return jsonify(success=True)

@flask_app.route('/start_mission', methods=['POST'])
def start_mission():
    if app_instance.waypoints:
        app_instance.current_waypoint_idx = 0
    return jsonify(success=True)

@flask_app.route('/map_data')
def map_data():
    idx   = app_instance.current_waypoint_idx
    n_wps = len(app_instance.waypoints)
    if idx >= 0 and idx < n_wps:
        wp_dist = float(np.linalg.norm(
            app_instance.waypoints[idx][:2] - app_instance.current_pos[:2]))
        mission_status = f"WP {idx+1}/{n_wps}"
        remaining = n_wps - idx
    else:
        wp_dist, remaining = 0.0, n_wps
        mission_status = "Ready" if n_wps > 0 else "Idle"
    sim_time = float(app_instance.my_world.current_time_step_index) / 60.0
    return jsonify({
        "robot":    {"x": float(app_instance.current_pos[0]),
                     "y": float(app_instance.current_pos[1]),
                     "yaw": float(app_instance.current_yaw)},
        "path":     app_instance.path_history[-500:],
        "waypoints": [[float(w[0]), float(w[1])] for w in app_instance.waypoints],
        "active_wp": idx,
        "linear_vel":  round(app_instance.linear_vel,  4),
        "angular_vel": round(app_instance.angular_vel, 4),
        "distance":    round(app_instance.distance_traveled, 4),
        "sim_time":    round(sim_time, 2),
        "mission": {"status": mission_status,
                    "wp_dist": round(wp_dist, 3),
                    "remaining": remaining},
        "speeds":  {"nav": app_instance.nav_speed,
                    "nav_angular_gain": app_instance.nav_angular_gain},
        "llm_mode":      getattr(app_instance, 'llm_mode', False),
        "llm_scene":     app_instance.llm_navigator.last_scene     if hasattr(app_instance, 'llm_navigator') else "",
        "llm_obstacle":  app_instance.llm_navigator.last_obstacle  if hasattr(app_instance, 'llm_navigator') else False,
        "llm_error":     app_instance.llm_navigator.error          if hasattr(app_instance, 'llm_navigator') else "",
        "llm_turn":      app_instance.llm_navigator.last_cmd.get("turn_degrees") if hasattr(app_instance, 'llm_navigator') else None,
        "llm_move":      app_instance.llm_navigator.last_cmd.get("move_meters")  if hasattr(app_instance, 'llm_navigator') else None,
    })

@flask_app.route('/set_speed', methods=['POST'])
def set_speed():
    data = request.json or {}
    if 'nav_speed' in data:
        app_instance.nav_speed = float(np.clip(data['nav_speed'], 0.05, 3.0))
    if 'nav_angular_gain' in data:
        app_instance.nav_angular_gain = float(np.clip(data['nav_angular_gain'], 0.1, 3.0))
    if 'jpeg_quality' in data:
        app_instance.jpeg_quality = int(np.clip(data['jpeg_quality'], 20, 95))
    return jsonify(success=True)

@flask_app.route('/stop_mission', methods=['POST'])
def stop_mission():
    app_instance.current_waypoint_idx = -1
    app_instance.cmd_linear  = 0.0
    app_instance.cmd_angular = 0.0
    return jsonify(success=True)

@flask_app.route('/delete_waypoint', methods=['POST'])
def delete_waypoint():
    data = request.json or {}
    idx = int(data.get('idx', -1))
    if 0 <= idx < len(app_instance.waypoints):
        app_instance.waypoints.pop(idx)
        if app_instance.current_waypoint_idx >= idx:
            app_instance.current_waypoint_idx = max(-1, app_instance.current_waypoint_idx - 1)
    return jsonify(success=True)

@flask_app.route('/clear_trail', methods=['POST'])
def clear_trail():
    app_instance.path_history = []
    return jsonify(success=True)

@flask_app.route('/go_home', methods=['POST'])
def go_home():
    app_instance.waypoints = [np.array([0.0, 0.0, 0.0])]
    app_instance.current_waypoint_idx = 0
    return jsonify(success=True)

@flask_app.route('/reset_odom', methods=['POST'])
def reset_odom():
    app_instance.distance_traveled = 0.0
    app_instance.linear_vel  = 0.0
    app_instance.angular_vel = 0.0
    app_instance._vel_buf    = []
    return jsonify(success=True)

@flask_app.route('/record/start', methods=['POST'])
def record_start():
    ok = app_instance.start_recording()
    return jsonify(success=ok, recording=app_instance.recording)

@flask_app.route('/record/stop', methods=['POST'])
def record_stop():
    fname = app_instance.stop_recording()
    return jsonify(success=bool(fname), filename=fname)

@flask_app.route('/record/status')
def record_status():
    elapsed = 0.0
    if app_instance.recording and app_instance.recording_start_time:
        elapsed = time.time() - app_instance.recording_start_time
    try:
        files = sorted([
            f for f in os.listdir(app_instance._recordings_dir)
            if f.endswith('.mp4')
        ])
        file_info = []
        for f in files[-10:]:
            fp = os.path.join(app_instance._recordings_dir, f)
            sz = os.path.getsize(fp)
            file_info.append({"name": f, "size": sz})
    except Exception:
        file_info = []
    return jsonify(
        recording=app_instance.recording,
        elapsed=round(elapsed, 1),
        frames=app_instance.recording_frames,
        files=file_info,
    )

@flask_app.route('/snapshot', methods=['POST'])
def snapshot():
    data  = request.json or {}
    label = str(data.get('label', ''))[:40]
    try:
        info = app_instance.take_snapshot(label=label)
        return jsonify(success=True, **info)
    except RuntimeError as e:
        return jsonify(success=False, error=str(e))

@flask_app.route('/dataset')
def dataset_list():
    try:
        files = sorted([
            f for f in os.listdir(app_instance._dataset_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ])
        file_info = [{"name": f, "size": os.path.getsize(
            os.path.join(app_instance._dataset_dir, f))} for f in files]
        return jsonify(files=file_info, total=len(files))
    except Exception as e:
        return jsonify(files=[], total=0, error=str(e))

@flask_app.route('/dataset/<path:filename>')
def serve_dataset(filename):
    fpath = os.path.join(app_instance._dataset_dir, filename)
    if not os.path.isfile(fpath):
        return Response('Not found', status=404)
    return send_file(fpath, mimetype='image/jpeg')

@flask_app.route('/dataset/<path:filename>/download')
def download_dataset(filename):
    fpath = os.path.join(app_instance._dataset_dir, filename)
    if not os.path.isfile(fpath):
        return Response('Not found', status=404)
    return send_file(fpath, as_attachment=True, download_name=filename)

@flask_app.route('/recordings/<path:filename>')
def download_recording(filename):
    fpath = os.path.join(app_instance._recordings_dir, filename)
    if not os.path.isfile(fpath):
        return Response('Not found', status=404)
    return send_file(fpath, as_attachment=True, download_name=filename)

@flask_app.route('/depth_feed')
def depth_feed():
    jpeg = app_instance.latest_depth_jpeg
    if jpeg:
        return Response(jpeg, mimetype='image/jpeg',
                       headers={'Cache-Control': 'no-cache, no-store, must-revalidate'})
    return Response(b'', status=204)

def _mjpeg_generator():
    """Yield multipart JPEG frames as fast as new ones arrive."""
    last_jpeg = None
    while True:
        jpeg = app_instance.latest_depth_jpeg
        if jpeg and jpeg is not last_jpeg:
            last_jpeg = jpeg
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
        else:
            time.sleep(0.01)  # 10 ms back-off when no new frame yet

@flask_app.route('/depth_stream')
def depth_stream():
    """MJPEG stream - single persistent connection, server pushes each new frame."""
    return Response(_mjpeg_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    headers={'Cache-Control': 'no-cache, no-store, must-revalidate',
                             'X-Accel-Buffering': 'no'})

@flask_app.route('/llm_toggle', methods=['POST'])
def llm_toggle():
    """Enable or disable LLM navigation mode at runtime."""
    app_instance.llm_mode = not app_instance.llm_mode
    if not app_instance.llm_mode:
        # Hand control back cleanly
        app_instance.cmd_linear  = 0.0
        app_instance.cmd_angular = 0.0
    status = {
        "llm_mode":      app_instance.llm_mode,
        "last_scene":    app_instance.llm_navigator.last_scene,
        "last_obstacle": app_instance.llm_navigator.last_obstacle,
        "error":         app_instance.llm_navigator.error,
    }
    return jsonify(success=True, **status)

@flask_app.route('/shutdown', methods=['POST'])
def shutdown():
    import signal
    print("[INFO] Shutdown requested via dashboard")
    # Stop recording if active
    if app_instance.recording:
        app_instance.stop_recording()
    # Schedule SIGTERM to self after response is sent
    def _kill():
        import time as _t
        _t.sleep(0.3)
        os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=_kill, daemon=True).start()
    return jsonify(success=True, message="Shutting down")

def run_flask():
    for port in [5000, 5001, 5002]:
        try:
            print(f"[INFO] Flask starting on port {port}")
            flask_app.run(host='0.0.0.0', port=port, threaded=True)
            break
        except OSError:
            print(f"[WARN] Port {port} in use, trying next…")

if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    app_instance.run()
