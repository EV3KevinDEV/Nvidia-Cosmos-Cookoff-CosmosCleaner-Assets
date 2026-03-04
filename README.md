# CosmosCleaner — Nvidia Cosmos Cookoff

An autonomous robot-vacuum simulation built on **Isaac Sim 5.0** with LLM-driven navigation via the **Cosmos Reason2 Bridge**.  
The robot performs boustrophedon (lawnmower) path planning, streaming its camera feed to a live Flask dashboard.

---

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

---

## Quick Start

### 1. Start the Cosmos Reason2 Bridge (on a remote GPU)
```bash
brev port-forward cosmos-reason -p 8080:8080
```

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

---

## LLM Navigation

The robot uses a **phase machine** driven by the Cosmos Reason2 Bridge:

```
IDLE → ROTATE (turn to heading) → DRIVE (forward N meters) → IDLE → ...
```

### Boustrophedon Path Planning
The LLM is prompted to perform a lawnmower sweep:
- Drive in long parallel rows
- At each wall, execute a 90° + 90° U-turn to start the next row
- `_sweep_row` tracks completed rows; direction (`left-to-right` / `right-to-left`) is passed in every prompt

### Image Inference
Camera frames are captured at **1280×800** (30 fps) for the dashboard stream.  
Before sending to the bridge, frames are **resized to 640×400** (~75% fewer pixels), reducing bridge payload and inference latency.

### Obstacle Avoidance Rules (in prompt priority order)
| Rule | Trigger | Response |
|---|---|---|
| A | Dark/black image | Spin 90°, stop |
| B | Wall fills center of image | Begin U-turn (90°) |
| C | Object >15% of frame / <0.8 m | Dodge 45–55°, stop |
| D | STALL alert (wall contact) | Escape turn 120° |
| E | IDLE alert (zero cmds ×3) | Spin 90° |
| F | Distant object | Gentle arc 10–20°, 0.10 m |
| G | Clear floor | Straight + micro-correction ±5–10°, 0.15–0.20 m |

### Key Tunable Constants (`launch_scene.py`)
| Constant | Value | Description |
|---|---|---|
| `nav_speed` | 0.40 m/s | Forward drive speed |
| `_BRIDGE_IMG_W/H` | 640 × 400 | Bridge inference resolution |
| `move_meters` clip | 0 – 0.20 m | Max step size per command |
| Stall threshold | 12 steps | Steps at low velocity before escape spin |
| No-turn override | 15 steps | Force U-turn after N straight steps |
| Turn granularity | 5° increments | All LLM turns rounded to nearest 5° |

---

## Environment

- **Room**: `simple_room.usd` (6.5 × 6.5 m)
- **Robot spawn**: `(-2.0, -2.0, 0.15)` — clear corner
- **19 physics obstacles** spawned in a stratified 4×3 grid (seed=42 for reproducibility)
- Obstacle types: cardboard boxes, KLT bins, YCB household objects, toy blocks

---

## Video Recording

Recordings are saved to `recordings/` as timestamped MP4 files (`cosmos_YYYYMMDD_HHMMSS.mp4`) at 30 fps.  
Files can be downloaded via the dashboard or directly from `http://localhost:5000/recordings/<filename>`.
