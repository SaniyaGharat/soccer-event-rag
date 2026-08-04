"""
Unit Tests for Preprocessing and Event Extraction Heuristics
"""

import pytest
from src.event_extraction.models import FrameData, TrackedPlayer, TrackedBall, EventType
from src.event_extraction.preprocessor import TrackingPreprocessor
from src.event_extraction.heuristics import EventExtractor


def test_track_gap_interpolation():
    preprocessor = TrackingPreprocessor(max_interpolation_gap_frames=5)

    # Frame 0: Player 1 at (10, 10)
    # Frame 1-3: Player 1 missing (gap of 3)
    # Frame 4: Player 1 at (20, 20)
    frames = [
        FrameData(frame_index=0, timestamp_sec=0.0, timestamp_str="00:00",
                  players=[TrackedPlayer(track_id=1, team_id="Team A", x=10.0, y=10.0)]),
        FrameData(frame_index=1, timestamp_sec=0.04, timestamp_str="00:00", players=[]),
        FrameData(frame_index=2, timestamp_sec=0.08, timestamp_str="00:00", players=[]),
        FrameData(frame_index=3, timestamp_sec=0.12, timestamp_str="00:00", players=[]),
        FrameData(frame_index=4, timestamp_sec=0.16, timestamp_str="00:00",
                  players=[TrackedPlayer(track_id=1, team_id="Team A", x=20.0, y=20.0)])
    ]

    cleaned_frames = preprocessor.interpolate_short_gaps(frames)

    # Verify Player 1 was interpolated into Frame 2
    f2_p1 = next((p for p in cleaned_frames[2].players if p.track_id == 1), None)
    assert f2_p1 is not None
    assert pytest.approx(f2_p1.x, 0.1) == 15.0
    assert pytest.approx(f2_p1.y, 0.1) == 15.0


def test_possession_dwell_time_smoothing():
    preprocessor = TrackingPreprocessor(possession_dwell_time_frames=4, possession_max_distance_meters=1.5)

    # 3 frames of Player 1 candidate possession, then 4 frames of Player 2 candidate possession
    frames = []
    for i in range(10):
        p_id = 1 if i < 3 else 2
        ball_x = 10.0 if p_id == 1 else 30.0
        frames.append(FrameData(
            frame_index=i, timestamp_sec=i * 0.04, timestamp_str="00:00",
            ball=TrackedBall(x=ball_x, y=10.0),
            players=[
                TrackedPlayer(track_id=1, team_id="Team A", x=10.0, y=10.0),
                TrackedPlayer(track_id=2, team_id="Team B", x=30.0, y=10.0)
            ]
        ))

    smoothed = preprocessor.compute_smoothed_possessions(frames)

    # Candidate 1 held for only 3 frames (< 4 threshold), so not confirmed
    # Candidate 2 held for 4+ frames, so confirmed starting around frame 6
    assert smoothed[2] is None
    assert smoothed[7] == ("Team B", 2)


def test_event_extraction_passes_and_turnovers():
    extractor = EventExtractor()

    frames = [
        FrameData(frame_index=0, timestamp_sec=0.0, timestamp_str="00:00",
                  ball=TrackedBall(x=10.0, y=10.0),
                  players=[TrackedPlayer(track_id=1, team_id="Team A", jersey_number=10, x=10.0, y=10.0)]),
        FrameData(frame_index=20, timestamp_sec=0.8, timestamp_str="00:00",
                  ball=TrackedBall(x=25.0, y=10.0),
                  players=[TrackedPlayer(track_id=2, team_id="Team A", jersey_number=7, x=25.0, y=10.0)])
    ]

    possessions = [("Team A", 1), ("Team A", 2)]
    smoothed_press = [False, False]

    events = extractor.extract_events(frames, possessions, smoothed_press)
    pass_evts = [e for e in events if e.event_type == EventType.COMPLETED_PASS]

    assert len(pass_evts) == 1
    assert pass_evts[0].team_id == "Team A"
    assert pass_evts[0].jersey_numbers == [10, 7]
