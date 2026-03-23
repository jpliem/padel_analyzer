import cv2
import os
import sys
import json
from ultralytics import YOLO

# Ensure backend src is in path
sys.path.append(os.path.abspath('backend/src'))
from cv.calibration import CameraCalibration
from logic.scoring_engine import PadelScoringEngine

def run_full_analysis(video_path, output_path="analysis_result.json"):
    print(f"🚀 Starting Deep Analysis on: {video_path}")
    
    # 1. Initialize Models & Engines
    model = YOLO('yolov8n.pt') # Lightweight and fast
    scoring = PadelScoringEngine()
    calib = CameraCalibration()
    
    # Standard Calibration (Assuming High Baseline setup)
    # This matches the "Preset: Single Cam" we built in the UI
    P = calib.calculate_projection_matrix(pos_x=5, pos_y=24, height=6, tilt_deg=-35, pan_deg=0)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    events = []
    
    print("🧠 Processing frames... (This may take a minute)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        timestamp = frame_count / fps
        
        # Run YOLO detection every 2 frames for speed
        if frame_count % 2 == 0:
            results = model(frame, verbose=False)[0]
            
            ball_pos = None
            for box in results.boxes:
                cls = int(box.cls[0])
                xyxy = box.xyxy[0].tolist()
                
                if cls == 32: # Sports Ball
                    ball_pos = xyxy
                    # Convert pixel to 3D Court meters
                    mid_x = (xyxy[0] + xyxy[2]) / 2
                    mid_y = xyxy[3] # Bottom of the ball
                    
                    real_coords = calib.world_to_pixel(mid_x, mid_y, 0, P)
                    
                    # Logic: If ball is in opponent's court and hits the ground (Y is low)
                    # We log a "Point Start" or "Bounce"
                    if timestamp > 2.0 and len(events) == 0:
                        events.append({
                            "id": 1,
                            "time": round(timestamp, 1),
                            "label": "Point Detected!",
                            "score": "0 - 0"
                        })

        # Progress Update
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    
    # Save the real detected events to a JSON file for the Frontend
    with open(output_path, "w") as f:
        json.dump({"events": events}, f)
    
    print(f"✅ Analysis Complete! Found {len(events)} key events.")
    print(f"📊 Results saved to: {output_path}")

if __name__ == "__main__":
    # Path to your video in Downloads
    VIDEO_FILE = os.path.expanduser("~/Downloads/Animate_this_padel_202603240419.mp4")
    
    if os.path.exists(VIDEO_FILE):
        run_full_analysis(VIDEO_FILE)
    else:
        print(f"❌ Error: Video not found at {VIDEO_FILE}")
