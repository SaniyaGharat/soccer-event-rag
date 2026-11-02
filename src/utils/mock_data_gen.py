"""
Mock Data Generator Utility

Generates:
1. `data/sample_tracking.json`: Per-frame player & ball tracking dataset with realistic, fluid soccer movement patterns.
2. `data/ground_truth_events.json`: Ground-truth benchmark dataset including positive, vague, compound, and negative (non-existent) queries.
3. `data/sample_match.mp4`: Synthetic soccer match video clip with full pitch markings, fluid player movements, and timestamp display.
"""

import os
import json
import math
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
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
    Generates synthetic match tracking data, ground truth event log, and sample match video
    with fluid, non-overlapping soccer dynamics and realistic tactical formations.
    """
    total_frames = int(duration_seconds * fps)
    frames_data = []

    # Realistic 4-3-3 tactical anchor positions (Pitch: 105m x 68m)
    # Team A attacks from left to right (towards x=105)
    team_a_base = {
        1: (6.0, 34.0),    # GK
        2: (26.0, 10.0),   # RB
        3: (20.0, 24.0),   # CB
        4: (20.0, 44.0),   # CB
        5: (26.0, 58.0),   # LB
        6: (38.0, 34.0),   # DM
        7: (48.0, 18.0),   # RCM
        8: (48.0, 50.0),   # LCM
        9: (68.0, 12.0),   # RW
        10: (70.0, 34.0),  # CF
        11: (68.0, 56.0)   # LW
    }

    # Team B attacks from right to left (towards x=0)
    team_b_base = {
        1: (99.0, 34.0),   # GK
        2: (79.0, 10.0),   # LB
        3: (85.0, 24.0),   # CB
        4: (85.0, 44.0),   # CB
        5: (79.0, 58.0),   # RB
        6: (67.0, 34.0),   # DM
        7: (57.0, 18.0),   # LCM
        8: (57.0, 50.0),   # RCM
        9: (37.0, 12.0),   # LW
        10: (35.0, 34.0),  # CF
        11: (37.0, 56.0)   # RW
    }

    team_a_jerseys = list(range(1, 12))
    team_b_jerseys = list(range(1, 12))

    # Passing sequence cycle for organic play
    team_a_pass_cycle = [(3, 6), (6, 7), (7, 10), (10, 8), (8, 11), (11, 5), (5, 4), (4, 2), (2, 9), (9, 6)]
    team_b_pass_cycle = [(3, 6), (6, 8), (8, 10), (10, 7), (7, 9), (9, 5), (5, 4), (4, 2), (2, 11), (11, 6)]

    for f_idx in range(total_frames):
        ts_sec = f_idx / fps
        mins = int(ts_sec) // 60
        secs = int(ts_sec) % 60
        ts_str = f"{mins:02d}:{secs:02d}"

        # ---------------------------------------------------------
        # 1. SCENARIO COMPUTATION (Ball Position & Possessor)
        # ---------------------------------------------------------
        if 40.0 <= ts_sec <= 50.0:
            # Scripted Event 1: Penalty Box Entry by Team A
            prog = (ts_sec - 40.0) / 10.0
            # Player 10 carries ball forward diagonally into penalty box (crosses x=88.5 around ts=47)
            ball_x = 68.0 + prog * 27.0   # 68.0 -> 95.0
            ball_y = 22.0 + prog * 11.0   # 22.0 -> 33.0
            possessor_team = "Team A"
            possessor_jersey = 10

        elif 85.0 <= ts_sec <= 95.0:
            # Scripted Event 2: Defensive Third Turnover
            if ts_sec < 90.0:
                # Team A defender 4 under pressure inside defensive third
                prog = (ts_sec - 85.0) / 5.0
                ball_x = 16.0 + prog * 8.0   # 16.0 -> 24.0
                ball_y = 16.0 + prog * 6.0   # 16.0 -> 22.0
                possessor_team = "Team A"
                possessor_jersey = 4
            else:
                # Team B attacker 7 intercepts at ts=90 and drives towards goal
                prog = (ts_sec - 90.0) / 5.0
                ball_x = 24.0 - prog * 10.0  # 24.0 -> 14.0
                ball_y = 22.0 + prog * 8.0   # 22.0 -> 30.0
                possessor_team = "Team B"
                possessor_jersey = 7

        elif 130.0 <= ts_sec <= 145.0:
            # Scripted Event 3: High Pressing Sequence on Team A Player 8
            # Player 8 maneuvers across 10m of pitch trying to evade a 4-man press
            prog = (ts_sec - 130.0) / 15.0
            ball_x = 52.0 - prog * 10.0 + math.sin(prog * 2 * math.pi) * 1.5
            ball_y = 34.0 - prog * 6.0 + math.cos(prog * 2 * math.pi) * 1.5
            possessor_team = "Team A"
            possessor_jersey = 8

        elif 185.0 <= ts_sec <= 195.0:
            # Scripted Event 4: Proxy Offside Pass
            if ts_sec < 189.0:
                prog = (ts_sec - 185.0) / 4.0
                ball_x = 62.0 + prog * 10.0  # 62.0 -> 72.0
                ball_y = 30.0 + prog * 2.0
                possessor_team = "Team A"
                possessor_jersey = 7
            else:
                # Forward through-pass to striker 9 who broke behind defensive line
                prog = (ts_sec - 189.0) / 6.0
                ball_x = 72.0 + prog * 14.0  # 72.0 -> 86.0
                ball_y = 32.0 + prog * 4.0
                possessor_team = "Team A"
                possessor_jersey = 9

        else:
            # General Organic Match Play: Dynamic Passing Sequences
            spell_cycle = int(ts_sec // 35)
            active_team = "Team A" if spell_cycle % 2 == 0 else "Team B"
            poss_cycle = team_a_pass_cycle if active_team == "Team A" else team_b_pass_cycle

            # Each pass cycle lasts 4.0 seconds (2.5s dribble, 1.5s pass)
            cycle_idx = int((ts_sec % 35) // 4.0) % len(poss_cycle)
            passer_j, receiver_j = poss_cycle[cycle_idx]

            sub_t = (ts_sec % 35) % 4.0
            base_map = team_a_base if active_team == "Team A" else team_b_base
            p_src = base_map[passer_j]
            p_dst = base_map[receiver_j]

            if sub_t < 2.5:
                # Player carrying/dribbling ball
                prog = sub_t / 2.5
                ball_x = p_src[0] + (p_dst[0] - p_src[0]) * 0.25 * prog + math.sin(ts_sec * 0.5) * 1.5
                ball_y = p_src[1] + (p_dst[1] - p_src[1]) * 0.25 * prog + math.cos(ts_sec * 0.5) * 1.0
                possessor_team = active_team
                possessor_jersey = passer_j
            else:
                # Ball in flight towards receiver
                prog = (sub_t - 2.5) / 1.5
                ball_x = (p_src[0] + (p_dst[0] - p_src[0]) * 0.25) + (p_dst[0] - (p_src[0] + (p_dst[0] - p_src[0]) * 0.25)) * prog
                ball_y = (p_src[1] + (p_dst[1] - p_src[1]) * 0.25) + (p_dst[1] - (p_src[1] + (p_dst[1] - p_src[1]) * 0.25)) * prog
                possessor_team = active_team
                possessor_jersey = receiver_j

        # Keep ball strictly inside pitch boundaries
        ball_x = max(2.0, min(103.0, ball_x))
        ball_y = max(2.0, min(66.0, ball_y))

        # ---------------------------------------------------------
        # 2. COLLECTIVE TEAM SHIFTING & PLAYER COORDINATES
        # ---------------------------------------------------------
        shift_a_x = (ball_x - 52.5) * 0.30
        shift_a_y = (ball_y - 34.0) * 0.20
        shift_b_x = (ball_x - 52.5) * 0.26
        shift_b_y = (ball_y - 34.0) * 0.20

        players_frame = []
        track_counter = 1

        # Build Team A players
        for j in team_a_jerseys:
            if j == possessor_jersey and possessor_team == "Team A":
                # Ball carrier running alongside the ball
                px = ball_x - 0.35
                py = ball_y - 0.25
            elif 130.0 <= ts_sec <= 145.0 and j in [2, 3, 5]:
                # Teammates supporting high-press escape in open passing channels
                offsets = {2: (-11.0, -9.0), 3: (-13.0, 7.0), 5: (9.0, -13.0)}
                ox, oy = offsets[j]
                px = ball_x + ox
                py = ball_y + oy
            elif 185.0 <= ts_sec <= 195.0 and j == 9:
                # Offside striker run
                px = 86.0 + (ts_sec - 185.0) * 0.4
                py = 36.0
            elif 40.0 <= ts_sec <= 50.0 and j == 9:
                # Striker attacking the near post
                px = 94.0 + (ts_sec - 40.0) * 0.3
                py = 27.0
            else:
                base_x, base_y = team_a_base[j]
                if j == 1:
                    px = base_x + (ball_x - 52.5) * 0.05
                    py = base_y + (ball_y - 34.0) * 0.15
                else:
                    drift_x = math.sin(ts_sec * 0.35 + j) * 1.2
                    drift_y = math.cos(ts_sec * 0.35 + j) * 0.9
                    px = base_x + shift_a_x + drift_x
                    py = base_y + shift_a_y + drift_y

            px = max(1.5, min(103.5, px))
            py = max(1.5, min(66.5, py))

            players_frame.append({
                "track_id": track_counter,
                "team_id": "Team A",
                "jersey_number": j,
                "x": round(float(px), 2),
                "y": round(float(py), 2),
                "speed": round(random.uniform(1.2, 4.0), 2)
            })
            track_counter += 1

        # Build Team B players
        for j in team_b_jerseys:
            if j == possessor_jersey and possessor_team == "Team B":
                px = ball_x + 0.35
                py = ball_y + 0.25
            elif 130.0 <= ts_sec <= 145.0 and possessor_team == "Team A" and j in [4, 6, 8, 9]:
                # Non-overlapping surrounding angles for the 4 pressing defenders
                # Distances strictly between 3.3m and 3.8m from ball (<= 5.0m threshold)
                press_positions = {
                    4: (ball_x + 3.3, ball_y + 1.2),   # Front press (~3.5m)
                    6: (ball_x + 0.4, ball_y + 3.5),   # Top-flank press (~3.5m)
                    8: (ball_x - 1.2, ball_y - 3.4),   # Rear-flank press (~3.6m)
                    9: (ball_x - 3.5, ball_y + 1.2),   # Behind-cutoff (~3.7m)
                }
                px, py = press_positions[j]
            elif 185.0 <= ts_sec <= 195.0 and j in [2, 3, 4, 5]:
                # Team B defensive line holding the offside trap at x=78.0
                y_slots = {2: 14.0, 3: 28.0, 4: 42.0, 5: 56.0}
                px = 78.0 + math.sin(ts_sec * 0.2) * 0.5
                py = y_slots[j]
            elif 85.0 <= ts_sec <= 90.0 and j == 7:
                # Forward pressing on defender 4 before turnover
                px = ball_x + 3.0 + (90.0 - ts_sec) * 0.8
                py = ball_y + 2.0
            elif 40.0 <= ts_sec <= 50.0 and j in [3, 4]:
                # Center-backs defending inside penalty box
                y_defs = {3: 28.0, 4: 38.0}
                px = 92.0 + math.sin(ts_sec * 0.3) * 1.0
                py = y_defs[j]
            else:
                base_x, base_y = team_b_base[j]
                if j == 1:
                    px = base_x + (ball_x - 52.5) * 0.05
                    py = base_y + (ball_y - 34.0) * 0.15
                else:
                    drift_x = math.sin(ts_sec * 0.35 + j) * 1.2
                    drift_y = math.cos(ts_sec * 0.35 + j) * 0.9
                    px = base_x + shift_b_x + drift_x
                    py = base_y + shift_b_y + drift_y

            px = max(1.5, min(103.5, px))
            py = max(1.5, min(66.5, py))

            players_frame.append({
                "track_id": track_counter,
                "team_id": "Team B",
                "jersey_number": j,
                "x": round(float(px), 2),
                "y": round(float(py), 2),
                "speed": round(random.uniform(1.2, 4.0), 2)
            })
            track_counter += 1

        frames_data.append({
            "frame_index": f_idx,
            "timestamp_sec": round(ts_sec, 2),
            "timestamp_str": ts_str,
            "ball": {
                "x": round(float(ball_x), 2),
                "y": round(float(ball_y), 2),
                "z": 0.0,
                "speed": round(random.uniform(2.0, 7.5), 2)
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

    # Save Tracking JSON (Atomically to prevent read race conditions)
    Path(output_tracking_path).parent.mkdir(parents=True, exist_ok=True)
    temp_tracking = Path(output_tracking_path).with_suffix(".tmp")
    with open(temp_tracking, "w", encoding="utf-8") as f:
        json.dump({"match_id": "synthetic_match_01", "fps": fps, "frames": frames_data}, f, indent=2)
    if os.path.exists(output_tracking_path):
        os.remove(output_tracking_path)
    os.rename(str(temp_tracking), str(output_tracking_path))
    print(f"[+] Saved synthetic tracking data to {output_tracking_path}")

    # Save Ground Truth JSON (Atomically)
    temp_gt = Path(output_ground_truth_path).with_suffix(".tmp")
    with open(temp_gt, "w", encoding="utf-8") as f:
        json.dump(ground_truth_log, f, indent=2)
    if os.path.exists(output_ground_truth_path):
        os.remove(output_ground_truth_path)
    os.rename(str(temp_gt), str(output_ground_truth_path))
    print(f"[+] Saved ground-truth benchmark events to {output_ground_truth_path}")

    # Render Sample Match MP4 Video if CV2 is installed
    if CV2_AVAILABLE:
        _render_synthetic_video(output_video_path, frames_data, fps=fps)
    else:
        print("[!] OpenCV (cv2) not installed. Skipping synthetic MP4 video generation.")


def _render_synthetic_video(video_path: str, frames: List[Dict[str, Any]], fps: float = 25.0):
    """Renders a complete 2D tactical pitch visualizer with anti-aliasing and markings."""
    width, height = 854, 480
    margin = 35
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    Path(video_path).parent.mkdir(parents=True, exist_ok=True)

    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    def to_px(x, y):
        px = int(margin + (x / 105.0) * (width - 2 * margin))
        py = int(margin + (y / 68.0) * (height - 2 * margin))
        return px, py

    pitch_w = width - 2 * margin
    r_center_circle = int((9.15 / 105.0) * pitch_w)

    for f in frames:
        # 1. Pitch Turf (Smooth deep grass green)
        img = np.full((height, width, 3), (38, 115, 45), dtype=np.uint8)

        # 2. Pitch Markings (Anti-aliased crisp white)
        line_color = (245, 245, 245)
        # Outer pitch boundary
        cv2.rectangle(img, to_px(0, 0), to_px(105, 68), line_color, 2, cv2.LINE_AA)
        # Halfway line
        cv2.line(img, to_px(52.5, 0), to_px(52.5, 68), line_color, 2, cv2.LINE_AA)
        # Center circle & spot
        cv2.circle(img, to_px(52.5, 34.0), r_center_circle, line_color, 2, cv2.LINE_AA)
        cv2.circle(img, to_px(52.5, 34.0), 3, line_color, -1, cv2.LINE_AA)
        # Penalty Boxes
        cv2.rectangle(img, to_px(0, 13.84), to_px(16.5, 54.16), line_color, 2, cv2.LINE_AA)
        cv2.rectangle(img, to_px(88.5, 13.84), to_px(105, 54.16), line_color, 2, cv2.LINE_AA)
        # Goal Boxes
        cv2.rectangle(img, to_px(0, 24.84), to_px(5.5, 43.16), line_color, 1, cv2.LINE_AA)
        cv2.rectangle(img, to_px(99.5, 24.84), to_px(105, 43.16), line_color, 1, cv2.LINE_AA)
        # Penalty spots
        cv2.circle(img, to_px(11.0, 34.0), 3, line_color, -1, cv2.LINE_AA)
        cv2.circle(img, to_px(94.0, 34.0), 3, line_color, -1, cv2.LINE_AA)

        # 3. Players
        for p in f["players"]:
            px, py = to_px(p["x"], p["y"])
            if p["team_id"] == "Team A":
                # Crimson Red with white border
                cv2.circle(img, (px, py), 8, (50, 50, 230), -1, cv2.LINE_AA)
                cv2.circle(img, (px, py), 8, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                # Royal Blue with white border
                cv2.circle(img, (px, py), 8, (230, 120, 50), -1, cv2.LINE_AA)
                cv2.circle(img, (px, py), 8, (255, 255, 255), 1, cv2.LINE_AA)

            # Jersey Number centered
            num_str = str(p.get("jersey_number", ""))
            (tw, th), _ = cv2.getTextSize(num_str, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.putText(img, num_str, (px - tw // 2, py + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. Ball (Bright gold yellow with dark border)
        if f.get("ball"):
            bx, by = to_px(f["ball"]["x"], f["ball"]["y"])
            cv2.circle(img, (bx, by), 6, (0, 230, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (bx, by), 6, (30, 30, 30), 1, cv2.LINE_AA)

        # 5. Match Time Display (clean dark pill at bottom left)
        time_label = f"MATCH TIME: {f['timestamp_str']}"
        cv2.rectangle(img, (margin, height - margin - 26), (margin + 175, height - margin), (20, 35, 20), -1)
        cv2.rectangle(img, (margin, height - margin - 26), (margin + 175, height - margin), (120, 160, 120), 1, cv2.LINE_AA)
        cv2.putText(img, time_label, (margin + 10, height - margin - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        out.write(img)

    out.release()

    # Convert to H.264 (yuv420p) with faststart for instant HTML5 web playback
    try:
        import subprocess
        from src.video.ffmpeg_clipper import get_ffmpeg_binary_path
        ffmpeg_exe = get_ffmpeg_binary_path()
        if ffmpeg_exe:
            temp_path = str(video_path) + ".h264.mp4"
            cmd = [
                ffmpeg_exe, "-y", "-i", str(video_path),
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                temp_path
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
            if res.returncode == 0 and Path(temp_path).exists():
                Path(video_path).unlink(missing_ok=True)
                Path(temp_path).rename(video_path)
    except Exception as e:
        print(f"[!] H.264 re-encoding warning: {e}")

    print(f"[+] Rendered sample match video to {video_path}")


if __name__ == "__main__":
    generate_synthetic_match_data()
