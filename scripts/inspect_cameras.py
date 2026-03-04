from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom
import os

usd_path = "/home/kevin/Nvidia-Cosmos-Cookoff-CosmosCleaner-Assets/CosmosCleanerBot_Camera.usd"
stage = Usd.Stage.Open(usd_path)

print(f"Inspecting cameras in: {usd_path}")
print("=" * 80)
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Camera):
        print(f"\nCamera Prim: {prim.GetPath()}")
        print(f"  Type: {prim.GetTypeName()}")
        # Print all attributes
        for attr in prim.GetAttributes():
            val = attr.Get()
            if val is not None:
                print(f"  {attr.GetName()} = {val}")
    # Also check for any prim with "camera" or "depth" in the name (case-insensitive)
    name_lower = prim.GetName().lower()
    if ("camera" in name_lower or "depth" in name_lower) and not prim.IsA(UsdGeom.Camera):
        print(f"\nRelated prim: {prim.GetPath()} (type: {prim.GetTypeName()})")

simulation_app.close()
