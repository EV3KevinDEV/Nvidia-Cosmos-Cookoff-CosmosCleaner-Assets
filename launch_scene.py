
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

import carb
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController

# ══════════════════════════════════════════════════════════════════════════════
# LLM NAVIGATION INTEGRATION  (commented out — wire up API credentials to enable)
# ══════════════════════════════════════════════════════════════════════════════
# Architecture:
#   Every N sim steps, LLMNavigator.step() is called with the latest JPEG frame
#   and current odometry.  It sends a multimodal request to the configured LLM
#   endpoint and parses the JSON response into turn + drive commands that are
#   forwarded to move_robot().
#
# To enable:
#   1. Set LLM_ENABLED = True
#   2. Fill in LLM_API_URL and LLM_API_KEY (or use env vars)
#   3. Uncomment the llm_navigator wiring in CosmosCleanerBotApp.__init__
#      and the llm_navigate_step() call in navigate_step()
#   4. Uncomment the /llm_toggle endpoint below
# ──────────────────────────────────────────────────────────────────────────────

# LLM_ENABLED  = False                # Master switch
# LLM_API_URL  = os.environ.get("LLM_API_URL",  "https://api.openai.com/v1/chat/completions")
# LLM_API_KEY  = os.environ.get("LLM_API_KEY",  "")
# LLM_MODEL    = os.environ.get("LLM_MODEL",     "gpt-4o")
# LLM_INTERVAL = 30   # call LLM every N sim steps (~0.5 s at 60 Hz)
# LLM_TIMEOUT  = 8.0  # seconds before giving up on a response

# System prompt is configured server-side — no local prompt needed.

# import base64, json as _json
# try:
#     import requests as _requests
#     _REQUESTS_OK = True
# except ImportError:
#     _REQUESTS_OK = False

# class LLMNavigator:
#     """Sends camera frame + odometry to an LLM and converts its JSON response
#     into (linear m/s, angular rad/s) commands for DifferentialController."""
#
#     # angular velocity scale: degrees → rad/s applied for one step period
#     # one step ≈ 1/60 s; we apply the command for LLM_INTERVAL steps
#     _DEG_PER_STEP  = 1.0 / LLM_INTERVAL   # fraction of requested turn per step
#     _M_PER_STEP    = 1.0 / LLM_INTERVAL   # fraction of requested distance per step
#
#     def __init__(self):
#         self.enabled      = LLM_ENABLED
#         self.last_cmd     = {"turn_degrees": 0.0, "move_meters": 0.0}
#         self.last_scene   = ""
#         self.last_obstacle= False
#         self.last_raw     = ""
#         self.error        = ""
#         self._lock        = threading.Lock()
#         self._pending     = False   # True while an async request is in-flight
#
#     def _encode_jpeg(self, jpeg_bytes: bytes) -> str:
#         """Base-64 encode a JPEG for the vision API payload."""
#         return base64.b64encode(jpeg_bytes).decode('utf-8')
#
#     def _build_payload(self, jpeg_bytes: bytes, odometry: dict) -> dict:
#         """Build the JSON body for the LLM API call."""
#         odom_text = (
#             f"Position: x={odometry['x']:.2f} m, y={odometry['y']:.2f} m  "
#             f"Yaw: {odometry['yaw']:.2f} rad  "
#             f"Linear vel: {odometry['linear_vel']:.2f} m/s  "
#             f"Angular vel: {odometry['angular_vel']:.2f} rad/s  "
#             f"Odometer: {odometry['distance']:.2f} m"
#         )
#         return {
#             "model": LLM_MODEL,
#             "max_tokens": 256,
#             "messages": [
#                 # system prompt is configured server-side
#                 {
#                     "role": "user",
#                     "content": [
#                         {
#                             "type": "image_url",
#                             "image_url": {
#                                 "url": f"data:image/jpeg;base64,{self._encode_jpeg(jpeg_bytes)}",
#                                 "detail": "low",   # low = faster + cheaper for navigation
#                             },
#                         },
#                         {"type": "text", "text": f"Current odometry:\n{odom_text}"},
#                     ],
#                 },
#             ],
#         }
#
#     def _call_llm(self, jpeg_bytes: bytes, odometry: dict):
#         """Blocking HTTP call — run inside a daemon thread so it never stalls
#         the sim loop."""
#         if not _REQUESTS_OK:
#             self.error = "'requests' package not installed"
#             self._pending = False
#             return
#         try:
#             payload  = self._build_payload(jpeg_bytes, odometry)
#             headers  = {
#                 "Authorization": f"Bearer {LLM_API_KEY}",
#                 "Content-Type":  "application/json",
#             }
#             resp = _requests.post(LLM_API_URL, headers=headers,
#                                   json=payload, timeout=LLM_TIMEOUT)
#             resp.raise_for_status()
#             raw_text = resp.json()["choices"][0]["message"]["content"].strip()
#             # Strip accidental markdown code fences
#             raw_text = raw_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
#             parsed   = _json.loads(raw_text)
#             # Validate & clamp
#             cmd = {
#                 "turn_degrees":  float(np.clip(parsed.get("turn_degrees",  0.0), -90.0, 90.0)),
#                 "move_meters":   float(np.clip(parsed.get("move_meters",   0.0), -0.5,   0.5)),
#             }
#             with self._lock:
#                 self.last_cmd      = cmd
#                 self.last_scene    = parsed.get("scene_analysis", "")
#                 self.last_obstacle = bool(parsed.get("obstacle_detected", False))
#                 self.last_raw      = raw_text
#                 self.error         = ""
#         except Exception as exc:
#             with self._lock:
#                 self.error = str(exc)
#         finally:
#             self._pending = False
#
#     def step(self, jpeg_bytes: bytes, odometry: dict):
#         """Non-blocking: fires off a background thread if none is in-flight.
#         Should be called every LLM_INTERVAL sim steps."""
#         if not self.enabled or not jpeg_bytes:
#             return
#         if self._pending:
#             return   # previous call still in-flight; reuse last command
#         self._pending = True
#         t = threading.Thread(
#             target=self._call_llm,
#             args=(jpeg_bytes, odometry),
#             daemon=True,
#         )
#         t.start()
#
#     def get_move_command(self) -> tuple[float, float]:
#         """Convert last LLM JSON response to (linear m/s, angular rad/s).
#         Called every sim step to apply a smooth fraction of the requested move."""
#         with self._lock:
#             cmd = self.last_cmd
#         # turn_degrees over LLM_INTERVAL steps → angular rad/s per step
#         angular = np.deg2rad(cmd["turn_degrees"]) * self._DEG_PER_STEP * 60.0
#         # move_meters over LLM_INTERVAL steps → linear m/s per step
#         linear  = cmd["move_meters"] * self._M_PER_STEP * 60.0
#         return float(linear), float(angular)

# ──────────────────────────────────────────────────────────────────────────────

class CosmosCleanerBotApp:
    def __init__(self):
        self.my_world = World(stage_units_in_meters=1.0)
        self.assets_root = os.getcwd()
        
        # Paths
        self.robot_usd = os.path.join(self.assets_root, "CosmosCleanerBot_Camera.usd")
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
        self.nav_speed = 0.5          # m/s for auto-nav straight drive
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

        # LLM navigation (disabled until credentials configured — see LLMNavigator above)
        # self.llm_navigator = LLMNavigator()
        # self.llm_mode      = False   # True = LLM drives; False = waypoint / manual
        
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
        self.my_world.scene.add_default_ground_plane()
        add_reference_to_stage(usd_path=self.robot_usd, prim_path=self.robot_prim_path)
        set_camera_view(eye=[2.0, 2.0, 2.0], target=[0.0, 0.0, 0.5])

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
        # _capture_depth runs every 5 sim steps at ~60 Hz → ~12 fps
        self.video_writer = cv2.VideoWriter(fpath, fourcc, 12, (1280, 800))
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

    def move_robot(self, linear, angular):
        # Negate linear only: the robot's physical forward direction is reversed.
        # Angular is correct (positive = left turn), so leave it as-is.
        wheel_actions = self.controller.forward(command=np.array([linear, angular]))
        # Revolute_right (index 1) has a flipped joint axis
        if wheel_actions.joint_velocities is not None:
            wheel_actions.joint_velocities[1] = -wheel_actions.joint_velocities[1]
        self.robot.apply_wheel_actions(wheel_actions)

    # def llm_navigate_step(self):
    #     """Replace navigate_step() when self.llm_mode is True.
    #     Fires the LLM every LLM_INTERVAL steps and applies its command each step."""
    #     step_idx = int(self.my_world.current_time_step_index)
    #     if step_idx % LLM_INTERVAL == 0:
    #         odometry = {
    #             "x":           float(self.current_pos[0]),
    #             "y":           float(self.current_pos[1]),
    #             "yaw":         float(self.current_yaw),
    #             "linear_vel":  float(self.linear_vel),
    #             "angular_vel": float(self.angular_vel),
    #             "distance":    float(self.distance_traveled),
    #         }
    #         self.llm_navigator.step(self.latest_depth_jpeg, odometry)
    #     linear, angular = self.llm_navigator.get_move_command()
    #     self.move_robot(linear, angular)

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
        # Capture depth camera frame (every 5 steps)
        if step_idx % 5 == 0:
            self._capture_depth()

    def navigate_step(self):
        # LLM mode overrides waypoint/manual when enabled:
        # if getattr(self, 'llm_mode', False):
        #     self.llm_navigate_step()
        #     return

        if self.current_waypoint_idx < 0:
            self.move_robot(self.cmd_linear, self.cmd_angular)
            return
            
        target = self.waypoints[self.current_waypoint_idx]
        dist = np.linalg.norm(target[:2] - self.current_pos[:2])
        target_vector = target[:2] - self.current_pos[:2]
        target_angle = np.arctan2(target_vector[1], target_vector[0])
        # Physical front = current_yaw + π (canvas uses ctx.rotate(-yaw+π)).
        # Correct diff = target - (current_yaw + π), which simplifies the formula to:
        angle_diff = (target_angle - self.current_yaw) % (2 * np.pi) - np.pi
        
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
        # LLM fields (active when llm_mode is enabled):
        # "llm_mode":      getattr(app_instance, 'llm_mode', False),
        # "llm_scene":     app_instance.llm_navigator.last_scene      if hasattr(app_instance, 'llm_navigator') else "",
        # "llm_obstacle":  app_instance.llm_navigator.last_obstacle   if hasattr(app_instance, 'llm_navigator') else False,
        # "llm_error":     app_instance.llm_navigator.error           if hasattr(app_instance, 'llm_navigator') else "",
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

# @flask_app.route('/llm_toggle', methods=['POST'])
# def llm_toggle():
#     """Enable or disable LLM navigation mode at runtime."""
#     app_instance.llm_mode = not app_instance.llm_mode
#     if not app_instance.llm_mode:
#         # Hand control back cleanly
#         app_instance.cmd_linear  = 0.0
#         app_instance.cmd_angular = 0.0
#     status = {
#         "llm_mode":      app_instance.llm_mode,
#         "last_scene":    app_instance.llm_navigator.last_scene,
#         "last_obstacle": app_instance.llm_navigator.last_obstacle,
#         "error":         app_instance.llm_navigator.error,
#     }
#     return jsonify(success=True, **status)

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
