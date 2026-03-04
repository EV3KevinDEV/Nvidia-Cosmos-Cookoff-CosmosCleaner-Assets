"""
test_room_scene.py
──────────────────
Isaac Sim 5.0 — Simple Room with robot-vacuum-style obstacles.

Spawns 9 props (cardboard boxes, KLT bins, blocks, a Rubik's cube) with
rigid-body physics and convex-hull colliders so the vacuum can't pass through.

Run:
    /home/kevin/anaconda3/envs/isaaclab/bin/python test_room_scene.py
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.prims import SingleRigidPrim

# ── Locate built-in assets ────────────────────────────────────────────────────
assets_root = get_assets_root_path()
if assets_root is None:
    carb.log_error("Could not find Isaac Sim asset root.")
    simulation_app.close()
    raise RuntimeError("Asset root not found.")

SIMPLE_ROOM_USD = assets_root + "/Isaac/Environments/Simple_Room/simple_room.usd"
print(f"[test_room] Loading room: {SIMPLE_ROOM_USD}")

# ── World ─────────────────────────────────────────────────────────────────────
world = World(stage_units_in_meters=1.0)
stage = world.stage

# ── Simple Room ───────────────────────────────────────────────────────────────
add_reference_to_stage(usd_path=SIMPLE_ROOM_USD, prim_path="/World/SimpleRoom")

# ── Obstacle helper ───────────────────────────────────────────────────────────
def spawn_obstacle(name: str, usd_path: str, position, scale=None, mass_kg=2.0,
                   yaw_deg: float = 0.0):
    """Load a USD prop, place it, and bake in rigid-body + convex colliders."""
    prim_path = f"/World/Obstacles/{name}"
    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
    prim = stage.GetPrimAtPath(prim_path)

    # ── Position / yaw / scale via xform ops ────────────────────────────────
    xform    = UsdGeom.Xformable(prim)
    existing = {op.GetOpName(): op for op in xform.GetOrderedXformOps()}

    t_op = existing.get("xformOp:translate") or xform.AddTranslateOp()
    t_op.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))

    r_op = existing.get("xformOp:rotateXYZ") or xform.AddRotateXYZOp()
    r_op.Set(Gf.Vec3f(0.0, 0.0, float(yaw_deg)))

    if scale is not None:
        s_op = existing.get("xformOp:scale") or xform.AddScaleOp()
        s_op.Set(Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2])))

    # ── Rigid body on root ───────────────────────────────────────────────────
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(prim)
    if not prim.HasAPI(UsdPhysics.MassAPI):
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.GetMassAttr().Set(mass_kg)

    # ── Convex collider on every geometry descendant ──────────────────────────
    for desc in Usd.PrimRange(prim):
        type_name = desc.GetTypeName()
        if type_name not in ("Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "Cone"):
            continue
        if not desc.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(desc)
        if type_name == "Mesh" and not desc.HasAPI(UsdPhysics.MeshCollisionAPI):
            mesh_col = UsdPhysics.MeshCollisionAPI.Apply(desc)
            mesh_col.GetApproximationAttr().Set("convexHull")

    # ── Wrap in SingleRigidPrim so Isaac Sim tracks it ────────────────────────
    try:
        rigid = world.scene.add(SingleRigidPrim(prim_path=prim_path, name=name))
        print(f"[test_room]   + {name} at ({position[0]:.2f}, {position[1]:.2f})")
        return rigid
    except Exception as e:
        print(f"[test_room]   ! {name} skipped ({e})")
        # Remove the failed prim so it doesn't leave a broken reference
        stage.RemovePrim(prim_path)
        return None

# ── Obstacle catalogue (name, usd_path, mass_kg, optional scale) ──────────────
R   = assets_root
WP  = R + "/Isaac/Environments/Simple_Warehouse/Props"
YCB = R + "/Isaac/Props/YCB/Axis_Aligned_Physics"   # pre-baked physics variants

CATALOGUE = [
    # ── Cardboard boxes (warehouse props) — big floor clutter ────────────────
    ("BoxA_1",        WP  + "/SM_CardBoxA_01.usd",            2.0,  None),
    ("BoxA_2",        WP  + "/SM_CardBoxA_01.usd",            2.0,  None),
    ("BoxB_1",        WP  + "/SM_CardBoxB_01.usd",            2.5,  None),
    ("BoxC_1",        WP  + "/SM_CardBoxC_01.usd",            3.0,  None),
    ("BoxD_1",        WP  + "/SM_CardBoxD_01.usd",            3.5,  None),

    # ── Storage bins ─────────────────────────────────────────────────────────
    ("KLT_1",         R   + "/Isaac/Props/KLT_Bin/small_KLT.usd",   0.8, None),
    ("KLT_2",         R   + "/Isaac/Props/KLT_Bin/small_KLT.usd",   0.8, None),

    # ── YCB household objects — realistic in-room debris ─────────────────────
    # Confirmed present in Isaac Sim asset catalog:
    ("CrackerBox",    YCB + "/003_cracker_box.usd",           0.41, None),
    ("SoupCan",       YCB + "/005_tomato_soup_can.usd",       0.35, None),
    ("PowerDrill",    YCB + "/035_power_drill.usd",           0.90, None),
    # Common YCB items — available on Nucleus with full asset pack:
    ("MasterChefCan", YCB + "/002_master_chef_can.usd",       0.41, None),
    ("SugarBox",      YCB + "/004_sugar_box.usd",             0.51, None),
    ("MustardBottle", YCB + "/006_mustard_bottle.usd",        0.60, None),
    ("TunaFishCan",   YCB + "/007_tuna_fish_can.usd",         0.17, None),
    ("PuddingBox",    YCB + "/008_pudding_box.usd",           0.19, None),
    ("GelatinBox",    YCB + "/009_gelatin_box.usd",           0.10, None),
    ("PottedMeat",    YCB + "/010_potted_meat_can.usd",       0.37, None),
    ("Mug",           YCB + "/025_mug.usd",                   0.12, None),
    ("Bowl",          YCB + "/024_bowl.usd",                   0.15, None),
    ("WoodBlock",     YCB + "/036_wood_block.usd",            0.73, None),
    ("TennisBall",    YCB + "/056_tennis_ball.usd",           0.06, None),
    ("Baseball",      YCB + "/055_baseball.usd",              0.14, None),

    # ── Small toys / misc floor items ────────────────────────────────────────
    ("RubiksCube",    R   + "/Isaac/Props/Rubiks_Cube/rubiks_cube.usd", 0.10, None),
]

# ── Random layout ─────────────────────────────────────────────────────────────
# Room is 6.5 × 6.5 m.  Keep 0.6 m away from walls → usable range ≈ ±2.65 m.
# Table occupies roughly the centre 1.5 m radius — exclude that zone.
# Props are distributed uniformly across the four quadrants so nothing clusters.
import itertools

ROOM_HALF   = 2.65    # usable half-extent after wall margin
TABLE_EXCL  = 1.5     # radius around origin to avoid (table zone)
DROP_Z      = 0.55    # drop height so items tumble and settle

rng = np.random.default_rng(seed=42)   # fixed seed → reproducible; remove for true random

# Pre-build a stratified grid: split room into a 4×3 tile grid so props are
# spread across the whole floor instead of piling in the middle.
_TILES_X, _TILES_Y = 4, 3
_tile_w = (2 * ROOM_HALF) / _TILES_X
_tile_h = (2 * ROOM_HALF) / _TILES_Y
_tile_order = [(c, r) for r in range(_TILES_Y) for c in range(_TILES_X)]
rng.shuffle(_tile_order)
_tile_cycle = itertools.cycle(_tile_order)   # infinite — never exhausts

def _random_pose(existing, margin=0.45, tiles_to_try=None, samples_per_tile=12):
    """Pick (x, y) by cycling through tiles, skipping the centre table zone."""
    if tiles_to_try is None:
        tiles_to_try = len(_tile_order)
    for _ in range(tiles_to_try):
        tile_col, tile_row = next(_tile_cycle)
        tx0 = -ROOM_HALF + tile_col * _tile_w
        ty0 = -ROOM_HALF + tile_row * _tile_h
        for _ in range(samples_per_tile):
            x = rng.uniform(tx0 + 0.1, tx0 + _tile_w - 0.1)
            y = rng.uniform(ty0 + 0.1, ty0 + _tile_h - 0.1)
            if np.hypot(x, y) < TABLE_EXCL:
                continue
            if all(np.hypot(x - ex, y - ey) >= margin for ex, ey in existing):
                return x, y, DROP_Z
    # last-resort: scan anywhere outside table zone
    for _ in range(500):
        x = rng.uniform(-ROOM_HALF, ROOM_HALF)
        y = rng.uniform(-ROOM_HALF, ROOM_HALF)
        if np.hypot(x, y) >= TABLE_EXCL and all(np.hypot(x-ex, y-ey) >= margin for ex, ey in existing):
            return x, y, DROP_Z
    return float(rng.uniform(-ROOM_HALF, ROOM_HALF)), float(rng.uniform(-ROOM_HALF, ROOM_HALF)), DROP_Z

placed_xy = []

print("[test_room] Spawning obstacles (random drop)…")
obstacles = []
for name, usd_path, mass, scale in CATALOGUE:
    x, y, z = _random_pose(placed_xy)
    placed_xy.append((x, y))
    yaw = float(rng.uniform(0, 360))
    obs = spawn_obstacle(name, usd_path, (x, y, z), scale=scale, mass_kg=mass, yaw_deg=yaw)
    if obs is not None:
        obstacles.append(obs)

# ── Camera ────────────────────────────────────────────────────────────────────
set_camera_view(
    eye    = np.array([4.0, 4.0, 3.5]),
    target = np.array([0.0, 0.0, 0.3]),
)

# ── Reset and run ─────────────────────────────────────────────────────────────
world.reset()
print(f"[test_room] Scene ready ({len(obstacles)} obstacles) — close the viewport to exit.")

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
