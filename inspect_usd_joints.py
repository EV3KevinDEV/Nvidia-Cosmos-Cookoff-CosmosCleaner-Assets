from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdPhysics
import os

usd_path = "/home/kevin/Nvidia-Cosmos-Cookoff-CosmosCleaner-Assets/CosmosCleanerBot_Camera.usd"
stage = Usd.Stage.Open(usd_path)

print(f"Inspecting: {usd_path}")
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.RevoluteJoint):
        print(f"Found Revolute Joint: {prim.GetPath()}")
    elif "Joint" in prim.GetTypeName():
        print(f"Found Joint-like Prim ({prim.GetTypeName()}): {prim.GetPath()}")

simulation_app.close()
