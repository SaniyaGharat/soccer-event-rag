"""
Mock Data Generator Utility

Generates:
1. `data/sample_tracking.json`: Per-frame player & ball tracking dataset with realistic movement patterns.
2. `data/ground_truth_events.json`: Ground-truth benchmark dataset including positive, vague, compound, and negative (non-existent) queries.
3. `data/sample_match.mp4`: Synthetic soccer match video clip with rendered pitch, players, ball, and timestamp display.
"""

import json
import math
import random
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def generate_synthetic_match_data(
    output_tracking_path: str = "data/sample_tracking.json",
    output_ground_truth_path: str = "data/ground_truth_events.json",
    output_video_path: str = "data/sample_match.mp4",
    duration_seconds: int = 300,  # 5 minutes match clip
    fps: float = 25.0
):
    """
    Generates synthetic match tracking data, ground truth event log, and sample match video.
    """
    total_frames = int(duration_seconds * fps)
    frames_data = []

    # Player setups
    team_a_jerseys = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    team_b_jerseys = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    ball_pos = np.array([52.5, 34.0, 0.0])  # Pitch center

    possessor_team = "Team A"
    possessor_jersey = 10

    for f_idx in range(total_frames):
        ts_sec = f_idx / fps
        mins = int(ts_sec) // 60
        secs = int(ts_sec) % 60
        ts_str = f"{mins:02d}:{secs:02d}"

        # Inject specific scripted scenarios at key timestamps
        if 40 <= ts_sec <= 50:
            # Penalty box entry
            ball_pos[0] = 70.0 + (ts_sec - 40) * 2.5
            ball_pos[1] = 34.0 + math.sin(ts_sec) * 3.0
            possessor_team = "Team A"
            possessor_jersey = 10
        elif 85 <= ts_sec <= 95:
            # Defensive third turnover
            ball_pos[0] = 20.0 + math.cos(ts_sec) * 2.0
            ball_pos[1] = 15.0 + math.sin(ts_sec) * 2.0
            possessor_team = "Team B" if ts_sec > 90 else "Team A"
            possessor_jersey = 7 if ts_sec > 90 else 4
        elif 130 <= ts_sec <= 145:
            # High press sequence
            ball_pos[0] = 45.0
            ball_pos[1] = 30.0
            possessor_team = "Team A"
            possessor_jersey = 8
        elif 185 <= ts_sec <= 195:
            # Proxy offside pass
            ball_pos[0] = 75.0 + (ts_sec - 185) * 1.5
            ball_pos[1] = 40.0
            possessor_team = "Team A"
            possessor_jersey = 9
        else:
            # General organic play drift
            ball_pos[0] = max(5.0, min(100.0, ball_pos[0] + random.uniform(-0.4, 0.4)))
            ball_pos[1] = max(5.0, min(63.0, ball_pos[1] + random.uniform(-0.3, 0.3)))

        # Build player coordinates
        players_frame = []
        track_counter = 1

        # Team A players
        for j in team_a_jerseys:
            if j == possessor_jersey and possessor_team == "Team A":
                px = ball_pos[0] + random.uniform(-0.5, 0.5)
                py = ball_pos[1] + random.uniform(-0.5, 0.5)
            elif 130 <= ts_sec <= 145 and possessor_team == "Team A" and j in [2, 3, 5]:
                px = ball_pos[0] + random.uniform(-3.0, 3.0)
                py = ball_pos[1] + random.uniform(-3.0, 3.0)
            elif 185 <= ts_sec <= 195 and j == 9:
                px = 92.0
                py = 40.0
            else:
                px = max(2.0, min(103.0, 15.0 + j * 7.5 + math.sin(f_idx * 0.02 + j) * 4.0))
                py = max(2.0, min(66.0, 10.0 + (j % 4) * 15.0 + math.cos(f_idx * 0.02 + j) * 3.0))

            players_frame.append({
                "track_id": track_counter,
                "team_id": "Team A",
                "jersey_number": j,
                "x": round(float(px), 2),
                "y": round(float(py), 2),
                "speed": round(random.uniform(0.5, 4.5), 2)
            })
            track_counter += 1

        # Team B players
        for j in team_b_jerseys:
            if j == possessor_jersey and possessor_team == "Team B":
                px = ball_pos[0] + random.uniform(-0.5, 0.5)
                py = ball_pos[1] + random.uniform(-0.5, 0.5)
            elif 130 <= ts_sec <= 145 and possessor_team == "Team A" and j in [4, 6, 8, 9]:
                px = ball_pos[0] + random.uniform(-2.5, 2.5)
                py = ball_pos[1] + random.uniform(-2.5, 2.5)
            elif 185 <= ts_sec <= 195 and j in [2, 3]:
                px = 80.0
                py = 35.0 + j * 5.0
            else:
                px = max(2.0, min(103.0, 90.0 - j * 7.0 + math.cos(f_idx * 0.02 + j) * 4.0))
                py = max(2.0, min(66.0, 10.0 + (j % 4) * 15.0 + math.sin(f_idx * 0.02 + j) * 3.0))

            players_frame.append({
                "track_id": track_counter,
                "team_id": "Team B",
                "jersey_number": j,
                "x": round(float(px), 2),
                "y": round(float(py), 2),
                "speed": round(random.uniform(0.5, 4.5), 2)
            })
            track_counter += 1

        frames_data.append({
            "frame_index": f_idx,
            "timestamp_sec": round(ts_sec, 2),
            "timestamp_str": ts_str,
            "ball": {
                "x": round(float(ball_pos[0]), 2),
                "y": round(float(ball_pos[1]), 2),
                "z": 0.0,
                "speed": round(random.uniform(1.0, 8.0), 2)
            },
            "players": players_frame
        })

    # Expanded 10-query benchmark dataset with positive, vague, compound, and negative queries
    ground_truth_log = [
        # --- Positive & Specific Queries ---
        {
            "id": "Q1",
            "query": "Show me when Team A crossed into the penalty box",
            "type": "specific_positive",
            "is_negative_query": False,
            "expected_event_type": "penalty_box_entry",
            "expected_start_sec": 40.0,
            "expected_end_sec": 50.0
        },
        {
            "id": "Q2",
            "query": "Find turnovers or lost possession in the defensive third",
            "type": "specific_positive",
            "is_negative_query": False,
            "expected_event_type": "turnover",
            "expected_start_sec": 85.0,
            "expected_end_sec": 95.0
        },
        {
            "id": "Q3",
            "query": "When was Team A subjected to high pressing sequence by defenders?",
            "type": "specific_positive",
            "is_negative_query": False,
            "expected_event_type": "high_press_sequence",
            "expected_start_sec": 130.0,
            "expected_end_sec": 145.0
        },
        {
            "id": "Q4",
            "query": "Find offside situations involving Player 9",
            "type": "specific_positive",
            "is_negative_query": False,
            "expected_event_type": "proxy_offside",
            "expected_start_sec": 185.0,
            "expected_end_sec": 195.0
        },

        # --- Vague Queries ---
        {
            "id": "Q5",
            "query": "Were there any dangerous moments near the goal area?",
            "type": "vague",
            "is_negative_query": False,
            "expected_event_type": "penalty_box_entry",
            "expected_start_sec": 40.0,
            "expected_end_sec": 50.0
        },
        {
            "id": "Q6",
            "query": "Show intense defensive pressure by the opposition",
            "type": "vague",
            "is_negative_query": False,
            "expected_event_type": "high_press_sequence",
            "expected_start_sec": 130.0,
            "expected_end_sec": 145.0
        },

        # --- Compound Queries ---
        {
            "id": "Q7",
            "query": "Find completed passes followed by turnover in defensive third",
            "type": "compound",
            "is_negative_query": False,
            "expected_event_type": "turnover",
            "expected_start_sec": 85.0,
            "expected_end_sec": 95.0
        },

        # --- Negative / Non-Existent Queries ---
        {
            "id": "Q8",
            "query": "Was there a second yellow card or red card given to any player?",
            "type": "negative",
            "is_negative_query": True,
            "expected_event_type": None,
            "expected_start_sec": None,
            "expected_end_sec": None
        },
        {
            "id": "Q9",
            "query": "Show VAR handball penalty review inside the box",
            "type": "negative",
            "is_negative_query": True,
            "expected_event_type": None,
            "expected_start_sec": None,
            "expected_end_sec": None
        },
        {
            "id": "Q10",
            "query": "Find corner kick taken from the left flag",
            "type": "negative",
            "is_negative_query": True,
            "expected_event_type": None,
            "expected_start_sec": None,
            "expected_end_sec": None
        }
    ]

    # Save Tracking JSON
    Path(output_tracking_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_tracking_path, "w", encoding="utf-8") as f:
        json.dump({"match_id": "synthetic_match_01", "fps": fps, "frames": frames_data}, f, indent=2)
    print(f"[+] Saved synthetic tracking data to {output_tracking_path}")

    # Save Ground Truth JSON
    with open(output_ground_truth_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth_log, f, indent=2)
    print(f"[+] Saved ground-truth benchmark events to {output_ground_truth_path}")

    # Render Sample Match MP4 Video if CV2 is installed
    if CV2_AVAILABLE:
        _render_synthetic_video(output_video_path, frames_data, fps=fps)
    else:
        print("[!] OpenCV (cv2) not installed. Skipping synthetic MP4 video generation.")


def _render_synthetic_video(video_path: str, frames: List[Dict[str, Any]], fps: float = 25.0):
    """Renders a simple 2D pitch visualizer video file."""
    width, height = 854, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    Path(video_path).parent.mkdir(parents=True, exist_ok=True)

    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for f in frames[::2]:
        img = np.full((height, width, 3), (34, 139, 34), dtype=np.uint8)  # Pitch Green

        margin = 30
        cv2.rectangle(img, (margin, margin), (width - margin, height - margin), (255, 255, 255), 2)
        cv2.line(img, (width // 2, margin), (width // 2, height - margin), (255, 255, 255), 2)

        def to_px(x, y):
            px = int(margin + (x / 105.0) * (width - 2 * margin))
            py = int(margin + (y / 68.0) * (height - 2 * margin))
            return px, py

        for p in f["players"]:
            px, py = to_px(p["x"], p["y"])
            color = (255, 50, 50) if p["team_id"] == "Team A" else (50, 50, 255)
            cv2.circle(img, (px, py), 6, color, -1)
            cv2.putText(img, str(p.get("jersey_number", "")), (px - 4, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        if f.get("ball"):
            bx, by = to_px(f["ball"]["x"], f["ball"]["y"])
            cv2.circle(img, (bx, by), 5, (0, 255, 255), -1)

        cv2.putText(img, f"MATCH TIME: {f['timestamp_str']}", (margin + 10, height - margin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(img)

    out.release()

    # Convert to H.264 (yuv420p) for HTML5 browser player compatibility
    try:
        import subprocess
        from src.video.ffmpeg_clipper import get_ffmpeg_binary_path
        ffmpeg_exe = get_ffmpeg_binary_path()
        if ffmpeg_exe:
            temp_path = str(video_path) + ".h264.mp4"
            cmd = [
                ffmpeg_exe, "-y", "-i", str(video_path),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", temp_path
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
            if res.returncode == 0 and Path(temp_path).exists():
                Path(video_path).unlink(missing_ok=True)
                Path(temp_path).rename(video_path)
    except Exception as e:
        print(f"[!] H.264 re-encoding warning: {e}")

    print(f"[+] Rendered sample match video to {video_path}")


if __name__ == "__main__":
    generate_synthetic_match_data()
