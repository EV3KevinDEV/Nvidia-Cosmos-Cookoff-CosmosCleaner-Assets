#!/usr/bin/env python3
"""
test_prompts.py — Multi-scenario prompt ablation tester for the Cosmos Reason2 Bridge.

Usage:
    python test_prompts.py [--scenario cables|wall|all] [--bridge URL] [--key KEY]

Scenarios:
  cables  — tangled cords on hardwood floor (TYPE-A entanglement hazard)
  wall    — robot facing wall/baseboard at row end (TYPE-B U-turn trigger)
  all     — run both (default)

Grading is scenario-specific (see SCENARIOS dict below).
"""

import argparse
import base64
import json
import os
import sys
import time
from io import BytesIO

try:
    import requests
    from PIL import Image
except ImportError:
    sys.exit("Install dependencies: pip install requests Pillow")

# ── Config ────────────────────────────────────────────────────────────────────
BRIDGE_URL = os.environ.get("BRIDGE_URL", "http://localhost:8080/v1/reason2/action")
BRIDGE_KEY = os.environ.get("BRIDGE_API_KEY", "super-secret-key")
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..")
BRIDGE_W, BRIDGE_H = 640, 400

# ── Shared telemetry stubs ────────────────────────────────────────────────────
_TELEM_SWEEP = (
    "Position: X=0.80, Y=1.20\n"
    "Heading: 0.0 deg\n"
    "Velocity: 0.35 m/s\n"
    "Sweep State: Row=2 (Direction: right-to-left)\n"
    "History: t=+0 m=0.20 | t=+0 m=0.20 | t=+0 m=0.18 | t=+0 m=0.19\n"
    "Active Alerts: NONE\n"
)
_TELEM_WALL = (
    "Position: X=2.58, Y=1.20\n"
    "Heading: 0.0 deg\n"
    "Velocity: 0.10 m/s\n"
    "Sweep State: Row=2 (Direction: right-to-left)\n"
    "History: t=+0 m=0.20 | t=+0 m=0.20 | t=+0 m=0.20 | t=+0 m=0.20\n"
    "Active Alerts: ALERT:STALL(wall_contact)\n"
)

# ── Prompt Variants (applied to both scenarios) ───────────────────────────────
def make_prompts(telem: str) -> dict:
    return {
        "1_production": (
            "You are the autonomous navigation brain for a robot vacuum.\n"
            "TASK: Execute a Boustrophedon (lawnmower) sweep.\n\n"
            "=== TELEMETRY ===\n" + telem +
            "\n=== DECISION RULES (Strict Priority) ===\n"
            "1. ALERTS (STALL/IDLE): STALL -> turn=120, move=0. IDLE -> turn=90, move=0.\n"
            "2. TRAPPED: dark/black image -> turn=90, move=0.\n"
            "3. ROW END (U-TURN): wall/baseboard in center 60% -> turn=90 or -90, move=0.\n"
            "4. MAJOR OBSTACLE: object fills >15% OR closer than 0.8m -> turn=45/50/55, move=0.\n"
            "5. MINOR OBSTACLE: small/far/off-center -> turn=10/15/20, move=0.10.\n"
            "6. CLEAR PATH: micro-correct heading, move=0.15-0.20.\n\n"
            "=== CONSTRAINTS ===\n"
            "- MOVEMENT: 0.0 to 0.20. NO REVERSING.\n"
            "- TURN GRAIN: multiples of 5. Allowed: [-120,-90,-55,-50,-45,-20,-15,-10,-5,0,5,10,15,20,45,50,55,90,120].\n"
            "Respond ONLY with raw JSON:\n"
            '{"scene_analysis": "<1 sentence>", "obstacle_detected": <bool>, '
            '"turn_degrees": <float mult of 5>, "move_meters": <0.0-0.20>}'
        ),

        "2_hazard_typed": (
            "You are the autonomous navigation brain for a robot vacuum.\n"
            "TASK: Boustrophedon (lawnmower) floor sweep — long parallel rows, U-turn at each wall.\n\n"
            "=== TELEMETRY ===\n" + telem +
            "\n=== HAZARD CLASSIFICATION (identify before acting) ===\n"
            "TYPE-A (ENTANGLEMENT) — cables, cords, wires, rope, string, charger, plug, shoelaces: "
            "STOP IMMEDIATELY. Hard turn 90° away. Never approach. This is the HIGHEST hazard.\n"
            "TYPE-B (STRUCTURAL) — walls, furniture legs, boxes, table legs, baseboards: "
            "Turn 45-90° away when within 0.8m or filling >15% of image. move=0.\n"
            "TYPE-C (SOFT/SMALL) — toys, socks, paper, small debris: Gentle arc 20-45° away. move=0.\n"
            "TYPE-D (SURFACE CHANGE) — rug edges, carpet borders, steps: Turn 30-45° away. move=0.\n\n"
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
            "- turn: multiples of 5 only. Allowed: [-120,-90,-55,-50,-45,-20,-15,-10,-5,0,5,10,15,20,45,50,55,90,120].\n"
            "Respond ONLY with raw JSON, no markdown:\n"
            '{"scene_analysis": "<1 sentence max 15 words>", "obstacle_detected": <bool>, '
            '"turn_degrees": <float mult of 5>, "move_meters": <float 0.0-0.20>}'
        ),

        "3_wall_explicit": (
            "You are the autonomous navigation brain for a robot vacuum.\n"
            "TASK: Boustrophedon floor sweep. U-turn at every wall.\n\n"
            "=== TELEMETRY ===\n" + telem +
            "\n=== WALL/BASEBOARD DETECTION (HIGHEST PRIORITY after alerts) ===\n"
            "If you can see a WALL, BASEBOARD, or SKIRTING BOARD anywhere in the image:\n"
            "  - The row is OVER. Do NOT drive forward.\n"
            "  - Output turn=90 (or -90), move=0 to begin U-turn.\n"
            "  - The next command will complete the U-turn with another 90° turn.\n"
            "  - A CARPET EDGE at the floor-wall junction is also a wall indicator.\n"
            "  - Even a partial wall visible at image edges = treat as wall ahead.\n\n"
            "=== HAZARD CLASSIFICATION ===\n"
            "TYPE-A (ENTANGLEMENT) — cables/cords/wires: turn=90, move=0. Highest hazard.\n"
            "TYPE-B (STRUCTURAL) — walls/furniture/baseboards: turn=45-90, move=0.\n"
            "TYPE-C (SOFT) — toys/debris: turn=30-45, move=0.\n"
            "TYPE-D (SURFACE) — rug edges/carpet: turn=30-45, move=0.\n\n"
            "=== DECISION RULES ===\n"
            "1. STALL alert -> turn=120, move=0.\n"
            "2. IDLE alert -> turn=90, move=0.\n"
            "3. TYPE-A anywhere -> turn=90, move=0.\n"
            "4. Wall/baseboard/skirting visible -> turn=90 or -90, move=0.\n"
            "5. TYPE-B fills >15% -> turn=55, move=0.\n"
            "6. TYPE-C or TYPE-D -> turn=45, move=0.\n"
            "7. Clear floor only -> turn=-10 to +10, move=0.15-0.20.\n\n"
            "=== CONSTRAINTS ===\n"
            "move: 0.0-0.20 only. Never negative. turn: multiples of 5.\n\n"
            "Respond ONLY with raw JSON:\n"
            '{"scene_analysis": "<1 sentence max 15 words>", "obstacle_detected": <bool>, '
            '"turn_degrees": <float mult of 5>, "move_meters": <float 0.0-0.20>}'
        ),

        "4_stall_aware": (
            "You are the autonomous navigation brain for a robot vacuum.\n"
            "TASK: Boustrophedon floor sweep.\n\n"
            "=== TELEMETRY ===\n" + telem +
            "\n=== CRITICAL: READ ALERTS FIRST ===\n"
            "If Active Alerts contains STALL: the robot is physically touching a wall.\n"
            "  -> You MUST turn away. turn=90 or 120, move=0. No exceptions.\n"
            "If Active Alerts contains IDLE: robot is stuck with zero output.\n"
            "  -> turn=90, move=0.\n\n"
            "=== HAZARD CLASSIFICATION ===\n"
            "TYPE-A (ENTANGLEMENT) — cables, cords, wires, charger: turn=90, move=0.\n"
            "TYPE-B (STRUCTURAL) — wall, baseboard, furniture, skirting: "
            "If filling >20% of image OR STALL alert present -> turn=90, move=0.\n"
            "TYPE-C (SOFT) — toys, debris: turn=45, move=0.\n"
            "TYPE-D (SURFACE) — carpet edge, rug: turn=30, move=0.\n\n"
            "=== RULES ===\n"
            "1. STALL alert -> turn=90 or 120, move=0.\n"
            "2. IDLE alert -> turn=90, move=0.\n"
            "3. TYPE-A anywhere -> turn=90, move=0.\n"
            "4. TYPE-B visible center of image -> turn=90, move=0.\n"
            "5. TYPE-B small/edge -> turn=55, move=0.\n"
            "6. TYPE-C or D -> turn=30/45, move=0.\n"
            "7. Clear -> turn=-10 to +10, move=0.15-0.20.\n\n"
            "CONSTRAINTS: move 0-0.20. Never negative. turn multiples of 5.\n\n"
            "Respond ONLY with raw JSON:\n"
            '{"scene_analysis": "<1 sentence max 15 words>", "obstacle_detected": <bool>, '
            '"turn_degrees": <float mult of 5>, "move_meters": <float 0.0-0.20>}'
        ),
    }


# ── Grading rubrics ───────────────────────────────────────────────────────────
def grade_cables(resp: dict) -> tuple[int, list[str]]:
    """Grade response for the cable/cord hazard scenario."""
    notes, score = [], 0
    obs   = resp.get("obstacle_detected", False)
    turn  = float(resp.get("turn_degrees", 0.0))
    move  = float(resp.get("move_meters", 999.0))
    scene = resp.get("scene_analysis", "").lower()

    if obs:
        score += 2; notes.append("✅ obstacle_detected=true")
    else:
        notes.append("❌ obstacle_detected=false (missed cables)")

    if move == 0.0:
        score += 2; notes.append("✅ move=0 (stopped)")
    elif move <= 0.05:
        score += 1; notes.append(f"⚠️  move={move:.2f} (near-stop)")
    else:
        notes.append(f"❌ move={move:.2f} (driving toward cables!)")

    if abs(turn) in {45, 50, 55, 90, 120}:
        score += 2; notes.append(f"✅ turn={turn:+.0f}° (strong escape)")
    elif abs(turn) >= 20:
        score += 1; notes.append(f"⚠️  turn={turn:+.0f}° (moderate escape)")
    else:
        notes.append(f"❌ turn={turn:+.0f}° (insufficient escape)")

    cable_words = {"cable", "cord", "wire", "tangle", "wires", "cords", "cables", "charger", "plug"}
    if any(w in scene for w in cable_words):
        score += 2; notes.append("✅ scene mentions cables/cords")
    else:
        notes.append(f"❌ no cable mention: \"{resp.get('scene_analysis','')[:60]}\"")

    if abs(turn) % 5 == 0:
        score += 2; notes.append("✅ turn mult of 5")
    else:
        notes.append(f"❌ turn={turn} not mult of 5")

    return score, notes


def grade_wall(resp: dict) -> tuple[int, list[str]]:
    """Grade response for the wall/baseboard row-end U-turn scenario."""
    notes, score = [], 0
    obs   = resp.get("obstacle_detected", False)
    turn  = float(resp.get("turn_degrees", 0.0))
    move  = float(resp.get("move_meters", 999.0))
    scene = resp.get("scene_analysis", "").lower()

    # 1. Must detect obstacle (wall = obstacle)
    if obs:
        score += 2; notes.append("✅ obstacle_detected=true")
    else:
        notes.append("❌ obstacle_detected=false (missed wall)")

    # 2. Must stop — do NOT drive into wall
    if move == 0.0:
        score += 2; notes.append("✅ move=0 (stopped at wall)")
    elif move <= 0.05:
        score += 1; notes.append(f"⚠️  move={move:.2f} (near-stop)")
    else:
        notes.append(f"❌ move={move:.2f} (driving into wall!)")

    # 3. Must be a strong 90° U-turn (not a small drift correction)
    if abs(turn) in {90, 120}:
        score += 2; notes.append(f"✅ turn={turn:+.0f}° (proper U-turn)")
    elif abs(turn) in {45, 50, 55}:
        score += 1; notes.append(f"⚠️  turn={turn:+.0f}° (partial turn, needs 90°)")
    else:
        notes.append(f"❌ turn={turn:+.0f}° (not a U-turn!)")

    # 4. scene mentions wall/baseboard
    wall_words = {"wall", "baseboard", "skirting", "board", "carpet", "floor end", "row end", "boundary"}
    if any(w in scene for w in wall_words):
        score += 2; notes.append("✅ scene mentions wall/baseboard")
    else:
        notes.append(f"❌ no wall mention: \"{resp.get('scene_analysis','')[:60]}\"")

    # 5. turn is multiple of 5
    if abs(turn) % 5 == 0:
        score += 2; notes.append("✅ turn mult of 5")
    else:
        notes.append(f"❌ turn={turn} not mult of 5")

    return score, notes


# ── Scenario definitions ──────────────────────────────────────────────────────
SCENARIOS = {
    "cables": {
        "image":   os.path.join(ASSETS_DIR, "test_scene.webp"),
        "telem":   _TELEM_SWEEP,
        "grade":   grade_cables,
        "label":   "CABLES on hardwood floor (TYPE-A entanglement hazard)",
        "expect":  "stop + 90° turn away from cables",
    },
    "wall": {
        "image":   os.path.join(ASSETS_DIR, "test_wall.webp"),
        "telem":   _TELEM_WALL,
        "grade":   grade_wall,
        "label":   "WALL/BASEBOARD at row end (U-turn trigger)",
        "expect":  "stop + 90° U-turn",
    },
}


# ── Bridge helpers ────────────────────────────────────────────────────────────
def load_image_b64(path: str) -> str:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = im.resize((BRIDGE_W, BRIDGE_H), Image.LANCZOS)
        buf = BytesIO()
        im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")


def call_bridge(image_b64: str, instruction: str, bridge_url: str, api_key: str):
    payload = {"image_b64": image_b64, "instruction": instruction}
    headers = {"Content-Type": "application/json", "X-API-Key": api_key}
    t0 = time.time()
    resp = requests.post(bridge_url, headers=headers, json=payload, timeout=60)
    elapsed = time.time() - t0
    resp.raise_for_status()
    raw = resp.text
    try:
        data = resp.json()
        if "turn_degrees" in data:
            return data, raw, elapsed
        for key in ("result", "action", "output"):
            if key in data:
                val = data[key]
                return (json.loads(val) if isinstance(val, str) else val), raw, elapsed
    except Exception:
        pass
    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(clean), raw, elapsed


# ── Runner ────────────────────────────────────────────────────────────────────
def run_scenario(name: str, scenario: dict, args):
    print(f"\n{'═'*70}")
    print(f"  SCENARIO: {name.upper()} — {scenario['label']}")
    print(f"  Expected: {scenario['expect']}")
    print(f"{'═'*70}")

    print(f"  Loading image: {scenario['image']}")
    try:
        image_b64 = load_image_b64(scenario["image"])
        print(f"  ✓ {BRIDGE_W}×{BRIDGE_H}, {len(image_b64)//1024} KB base64\n")
    except Exception as e:
        print(f"  ❌ Cannot load image: {e}\n")
        return []

    prompts = make_prompts(scenario["telem"])
    grader  = scenario["grade"]
    results = []

    for pname, prompt in prompts.items():
        print(f"  {'─'*65}")
        print(f"  PROMPT: {pname}  ({len(prompt)} chars)")
        try:
            parsed, raw, elapsed = call_bridge(image_b64, prompt, args.bridge, args.key)
            score, notes = grader(parsed)

            print(f"  Response ({elapsed:.1f}s):")
            print(f"    scene_analysis   : {parsed.get('scene_analysis', 'N/A')}")
            print(f"    obstacle_detected: {parsed.get('obstacle_detected', 'N/A')}")
            print(f"    turn_degrees     : {parsed.get('turn_degrees', 'N/A')}")
            print(f"    move_meters      : {parsed.get('move_meters', 'N/A')}")
            print(f"\n  Score: {score}/10")
            for n in notes:
                print(f"    {n}")
            results.append((pname, score, elapsed, parsed))
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results.append((pname, -1, 0, {}))

        print()
        time.sleep(1)

    # Summary for this scenario
    print(f"\n  {'─'*65}")
    print(f"  SCENARIO SUMMARY: {name.upper()}")
    print(f"  {'Prompt':<25} {'Score':>6}  {'Time':>6}  {'Turn':>7}  {'Move':>6}")
    print(f"  {'-'*55}")
    sorted_r = sorted(results, key=lambda r: r[1], reverse=True)
    for pn, sc, el, pr in sorted_r:
        sc_str = f"{sc}/10" if sc >= 0 else "ERROR"
        turn = pr.get("turn_degrees", "?")
        move = pr.get("move_meters",  "?")
        print(f"  {pn:<25} {sc_str:>6}  {el:>5.1f}s  {str(turn):>7}  {str(move):>6}")
    if sorted_r and sorted_r[0][1] >= 0:
        best = sorted_r[0]
        print(f"\n  🏆 Best: {best[0]} ({best[1]}/10)")
    return sorted_r


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="all",
                        choices=list(SCENARIOS.keys()) + ["all"])
    parser.add_argument("--bridge", default=BRIDGE_URL)
    parser.add_argument("--key",    default=BRIDGE_KEY)
    args = parser.parse_args()

    to_run = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]
    all_results = {}

    for name in to_run:
        all_results[name] = run_scenario(name, SCENARIOS[name], args)

    # Cross-scenario leaderboard
    if len(to_run) > 1:
        print(f"\n{'═'*70}")
        print("  CROSS-SCENARIO LEADERBOARD (avg score across both scenarios)")
        print(f"{'═'*70}")
        prompt_totals = {}
        for sname, results in all_results.items():
            for pname, score, _, _ in results:
                if score >= 0:
                    prompt_totals.setdefault(pname, []).append(score)
        ranked = sorted(prompt_totals.items(),
                        key=lambda kv: sum(kv[1])/len(kv[1]), reverse=True)
        print(f"  {'Prompt':<25} {'Avg':>6}  {'Cables':>8}  {'Wall':>6}")
        print(f"  {'-'*50}")
        for pname, scores in ranked:
            avg = sum(scores) / len(scores)
            per_scene = {sname: next((sc for pn, sc, _, _ in res if pn == pname), "?")
                         for sname, res in all_results.items()}
            cables_sc = per_scene.get("cables", "?")
            wall_sc   = per_scene.get("wall", "?")
            print(f"  {pname:<25} {avg:>5.1f}  {str(cables_sc)+'/10':>8}  {str(wall_sc)+'/10':>6}")
        print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()

