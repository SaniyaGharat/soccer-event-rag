import json
from typing import List, Dict, Any, Union
from pathlib import Path
from src.event_extraction.models import FrameData, TrackedPlayer, TrackedBall


def format_seconds_to_timestamp(seconds: float) -> str:
    """Converts seconds float into MM:SS format."""
    total_sec = int(seconds)
    mins = total_sec // 60
    secs = total_sec % 60
    return f"{mins:02d}:{secs:02d}"


def parse_tracking_json(json_path: Union[str, Path]) -> List[FrameData]:
    """
    Parses Game State Reconstruction (GSR) or synthetic per-frame tracking JSON data.
    
    Expected JSON structure:
    {
      "match_id": "match_01",
      "fps": 25.0,
      "frames": [
        {
          "frame_index": 0,
          "timestamp_sec": 0.0,
          "ball": {"x": 52.5, "y": 34.0, "speed": 0.0},
          "players": [
            {"track_id": 1, "team_id": "Team A", "jersey_number": 10, "x": 50.0, "y": 34.0, "speed": 1.2}
          ]
        }
      ]
    }
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Tracking file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_frames = data.get("frames", data) if isinstance(data, dict) else data

    parsed_frames: List[FrameData] = []
    fps = data.get("fps", 25.0) if isinstance(data, dict) else 25.0

    for item in raw_frames:
        frame_idx = item.get("frame_index", 0)
        ts_sec = item.get("timestamp_sec", frame_idx / fps)
        ts_str = item.get("timestamp_str", format_seconds_to_timestamp(ts_sec))

        ball_data = item.get("ball")
        ball_obj = TrackedBall(**ball_data) if ball_data else None

        players_obj = []
        for p in item.get("players", []):
            players_obj.append(TrackedPlayer(
                track_id=p["track_id"],
                team_id=p.get("team_id", "Unknown"),
                jersey_number=p.get("jersey_number"),
                x=float(p.get("x", 0.0)),
                y=float(p.get("y", 0.0)),
                speed=float(p.get("speed", 0.0))
            ))

        parsed_frames.append(FrameData(
            frame_index=frame_idx,
            timestamp_sec=ts_sec,
            timestamp_str=ts_str,
            players=players_obj,
            ball=ball_obj
        ))

    # Ensure frames are sorted chronologically
    parsed_frames.sort(key=lambda f: f.frame_index)
    return parsed_frames
