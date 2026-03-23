import sys
import os
import time

# Ensure our backend src is in path
sys.path.append(os.path.abspath('backend/src'))

from logic.scoring_engine import PadelScoringEngine
from cv.calibration import CameraCalibration

def run_simulation():
    print("🎾 Starting Padel Analyzer Simulation...")
    
    # 1. Setup Logic Engine
    engine = PadelScoringEngine()
    
    # 2. Setup Calibration (Digital Twin)
    # Let's say we placed a camera behind the baseline at (5m, 22m), 5m high.
    calib = CameraCalibration()
    P = calib.calculate_projection_matrix(pos_x=5, pos_y=22, height=5, tilt_deg=-30, pan_deg=0)
    
    # 3. Simulate a Point:
    # Ball starts at (5, 10, 0) - middle of net
    # Ball bounces at (3, 5, 0) - opponent side
    # Ball is hit by opponent at (2, 2, 1) - backcourt
    # Ball hits the net at (5, 10, 0.5) - FAULT
    
    simulation_steps = [
        {"desc": "Ball is served...", "pos": (5, 18, 1)},
        {"desc": "Ball bounces in opponent court!", "pos": (3, 5, 0), "event": "BOUNCE"},
        {"desc": "Opponent hits back!", "pos": (2, 2, 1), "event": "HIT"},
        {"desc": "Ball hits the net!", "pos": (5, 10, 0.4), "event": "NET_HIT"},
        {"desc": "Team 1 wins the point!", "team_won": 1}
    ]

    for step in simulation_steps:
        print(f"\n[ACTION]: {step['desc']}")
        
        if "pos" in step:
            x, y, z = step["pos"]
            px, py = calib.world_to_pixel(x, y, z, P)
            print(f"   -> 3D Position: ({x}m, {y}m, {z}m)")
            print(f"   -> Screen Pixel: (x={px}, y={py})")
        
        if "team_won" in step:
            engine.add_point(step["team_won"])
            score = engine.get_score_display()
            print(f"🔥 SCORE UPDATED: {score['score']}")
            print(f"📊 SETS: {score['sets']} | GAMES: {score['games']}")
        
        time.sleep(1) # Slow it down for the demo

    print("\n✅ Simulation Complete. The Brain is ready.")

if __name__ == "__main__":
    run_simulation()
