
from isaacsim import SimulationApp
import os

simulation_app = SimulationApp({"headless": True}) # Headless for just checking stage

import sys
import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Usd, UsdGeom

usd_path = "/home/kevin/Nvidia-Cosmos-Cookoff-CosmosCleaner-Assets/CosmosCleanerBot_Camera.usd"

if not os.path.exists(usd_path):
    print(f"USD file not found: {usd_path}")
    simulation_app.close()
    sys.exit()

stage = Usd.Stage.Open(usd_path)
print("--- USD Structure ---")
for prim in stage.Traverse():
    print(prim.GetPath())

simulation_app.close()
