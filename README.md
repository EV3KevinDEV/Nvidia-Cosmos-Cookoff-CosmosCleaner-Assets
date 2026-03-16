# CosmosCleaner — Nvidia Cosmos Cookoff

An autonomous robot-vacuum simulation built on **Isaac Sim 5.0** with LLM-driven navigation via the **Cosmos Reason2 Bridge**.  
The robot performs boustrophedon (lawnmower) path planning, streaming its camera feed to a live Flask dashboard.

---
## Demo Video 🎥
[https://streamable.com/zf4r1t
](url) 

## Project Structure

```
.
├── launch_scene.py          # Main entry point — sim loop + Flask server
├── usd/                     # USD scene & robot assets
│   ├── CosmosCleanerBot_Camera.usd   # Robot with onboard RGB camera
│   ├── CosmosCleanerBot.usd          # Robot (no camera variant)
│   └── Cosmos.usd                    # Top-level scene reference
├── templates/
│   └── index.html           # Dashboard web UI
├── scripts/                 # Utility / debug scripts
│   ├── dashboard.py         # Standalone dashboard prototype
│   ├── test_room_scene.py   # Standalone room scene test
│   ├── check_usd.py         # USD validation helper
│   ├── inspect_cameras.py   # Camera prim inspector
│   └── inspect_usd_joints.py # Joint DOF inspector
├── recordings/              # MP4 video recordings (auto-created)
└── obstacles/               # Extra obstacle USD assets
```

---

## Requirements

| Dependency | Version |
|---|---|
| Isaac Sim | 5.0 |
| Python | 3.11 (isaaclab conda env) |
| Flask | any |
| requests | any |
| Pillow | any |
| OpenCV (`cv2`) | any |
| numpy | any |

Activate the environment before running:
```bash
conda activate isaaclab
```

## Quick Start

### 1. Start the Cosmos Reason2 Bridge (on a remote GPU)
```bash
brev port-forward cosmos-reason -p 8080:8080
```

## LLM Navigation Logic

The robot's navigation is driven by the **Cosmos Reason2 Bridge**, which acts as the "brain." Instead of traditional hard-coded pathfinding, it uses a Vision-Language Model (VLM) to interpret the scene and make movement decisions.

### How it Works:
1.  **Visual Perception**: Every 30 simulation steps (~0.5 seconds), the robot captures its onboard RGB camera feed.
2.  **Telemetry Data**: The current odometry (X/Y position, heading, velocity, and sweep state) is bundled with the image.
3.  **VLM Processing**: The image and telemetry are sent to the remote LLM bridge via a `POST` request.
4.  **Action Output**: The LLM analyzes the scene (e.g., "identified cables," "approaching wall") and responds with a standardized JSON command:
    ```json
    {
      "scene_analysis": "Clear path ahead; continuing sweep.",
      "obstacle_detected": false,
      "turn_degrees": 0.0,
      "move_meters": 0.20
    }
    ```
5.  **Velocity Mapping**:
    *   `move_meters` is mapped to **Linear Velocity** (capped at 1.0 m/s).
    *   `turn_degrees` is mapped to **Angular Velocity** (capped at 2.5 rad/s).
    *   The robot maintains this velocity until the next LLM command arrives, ensuring smooth, reactive movement.

### Behavioral Rules (Strict Priority):
The LLM is prompted to follow a specific hierarchy of rules:
1.  **Stall Recovery**: If telemetry indicates a `STALL`, the robot must turn 90-120° away.
2.  **Hazard Mitigation**:
    *   **Type-A (Entanglement)**: Cables, cords, or wires trigger an immediate STOP and a 90° turn away.
    *   **Type-B (Structural)**: Walls and furniture trigger a 45-90° turn (or a U-turn for row ends).
3.  **Boustrophedon Sweep**: If the path is clear, the robot performs long parallel rows, executing a 180° turn (two 90° turns) at every wall.

---

### 2. Launch the simulation
```bash
cd /home/kevin/Nvidia-Cosmos-Cookoff-CosmosCleaner-Assets
/home/kevin/anaconda3/envs/isaaclab/bin/python launch_scene.py
```

### 3. Open the dashboard
Navigate to **http://localhost:5000** in your browser.

---

## Dashboard Controls

| Control | Description |
|---|---|
| **LLM Toggle** | Enable / disable autonomous LLM navigation |
| **D-pad / WASD** | Manual robot control |
| **Speed Sliders** | Adjust `nav_speed` and `angular_gain` |
| **Waypoints** | Click map to set waypoints, then Start Mission |
| **Record** | Start / stop MP4 video recording |
| **Shutdown** | Gracefully terminate the simulation |

-

## Environment

- **Room**: `simple_room.usd` (6.5 × 6.5 m)
- **Robot spawn**: `(-2.0, -2.0, 0.15)` — clear corner
- **19 physics obstacles** spawned in a stratified 4×3 grid (seed=42 for reproducibility)
- Obstacle types: cardboard boxes, KLT bins, YCB household objects, toy blocks

---

## Video Recording

Recordings are saved to `recordings/` as timestamped MP4 files (`cosmos_YYYYMMDD_HHMMSS.mp4`) at 30 fps.  
Files can be downloaded via the dashboard or directly from `http://localhost:5000/recordings/<filename>`.
