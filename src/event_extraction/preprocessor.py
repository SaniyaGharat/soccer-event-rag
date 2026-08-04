"""
Preprocessing & Noise Handling Pipeline Stage (Stage 1)

Raw Game State Reconstruction (GSR) tracking data often suffers from:
1. Occlusion gaps (players disappearing for 2-8 frames).
2. High-frequency coordinate jitter.
3. Rapid possession/pressing flicker due to noisy bounding boxes or distance boundaries.

This module provides deterministic noise suppression before rule-based heuristics run.
"""

import math
from typing import List, Dict, Optional, Tuple
from copy import deepcopy
from src.event_extraction.models import FrameData, TrackedPlayer, TrackedBall


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two 2D points."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class TrackingPreprocessor:
    """
    Discrete preprocessing stage to clean raw tracking data prior to heuristic event extraction.
    """

    def __init__(
        self,
        max_interpolation_gap_frames: int = 8,
        possession_dwell_time_frames: int = 5,
        pressing_dwell_time_frames: int = 10,
        possession_max_distance_meters: float = 1.5,
        pressing_max_distance_meters: float = 5.0,
        pressing_min_defenders: int = 3
    ):
        self.max_gap = max_interpolation_gap_frames
        self.possession_dwell = possession_dwell_time_frames
        self.pressing_dwell = pressing_dwell_time_frames
        self.possession_max_dist = possession_max_distance_meters
        self.pressing_max_dist = pressing_max_distance_meters
        self.pressing_min_defenders = pressing_min_defenders

    def preprocess(self, frames: List[FrameData]) -> List[FrameData]:
        """
        Runs complete preprocessing pipeline:
        1. Interpolates missing tracks over short occlusion gaps.
        2. Returns smoothed frames.
        """
        frames_copy = deepcopy(frames)
        interpolated_frames = self.interpolate_short_gaps(frames_copy)
        return interpolated_frames

    def interpolate_short_gaps(self, frames: List[FrameData]) -> List[FrameData]:
        """
        Linearly interpolates player coordinates for missing frames up to max_gap.
        """
        if not frames:
            return frames

        # Collect all unique track_ids
        all_tracks: Dict[int, List[Tuple[int, float, float, str, Optional[int]]]] = {}
        for f_idx, frame in enumerate(frames):
            for player in frame.players:
                if player.track_id not in all_tracks:
                    all_tracks[player.track_id] = []
                all_tracks[player.track_id].append((
                    f_idx, player.x, player.y, player.team_id, player.jersey_number
                ))

        # Find gaps and build interpolated entries
        interpolations: Dict[int, Dict[int, TrackedPlayer]] = {i: {} for i in range(len(frames))}

        for track_id, appearances in all_tracks.items():
            for k in range(len(appearances) - 1):
                f1, x1, y1, team, jersey = appearances[k]
                f2, x2, y2, _, _ = appearances[k + 1]
                gap = f2 - f1 - 1

                if 0 < gap <= self.max_gap:
                    for step in range(1, gap + 1):
                        target_f = f1 + step
                        alpha = step / (gap + 1)
                        interp_x = x1 + alpha * (x2 - x1)
                        interp_y = y1 + alpha * (y2 - y1)

                        interpolations[target_f][track_id] = TrackedPlayer(
                            track_id=track_id,
                            team_id=team,
                            jersey_number=jersey,
                            x=interp_x,
                            y=interp_y,
                            speed=distance(x1, y1, x2, y2) / ((gap + 1) * 0.04)  # approx m/s
                        )

        # Inject interpolated players back into frames
        for f_idx, frame in enumerate(frames):
            existing_ids = {p.track_id for p in frame.players}
            for t_id, player_obj in interpolations[f_idx].items():
                if t_id not in existing_ids:
                    frame.players.append(player_obj)

        return frames

    def compute_smoothed_possessions(
        self, frames: List[FrameData]
    ) -> List[Optional[Tuple[str, int]]]:
        """
        Computes frame-by-frame confirmed possession with dwell-time filter.
        Returns a list of (team_id, player_track_id) or None for each frame.
        """
        raw_candidates: List[Optional[Tuple[str, int]]] = []

        # Step A: Identify candidate closest player per frame
        for frame in frames:
            if not frame.ball:
                raw_candidates.append(None)
                continue

            closest_p: Optional[TrackedPlayer] = None
            min_d = float('inf')

            for p in frame.players:
                d = distance(p.x, p.y, frame.ball.x, frame.ball.y)
                if d < min_d:
                    min_d = d
                    closest_p = p

            if closest_p and min_d <= self.possession_max_dist:
                raw_candidates.append((closest_p.team_id, closest_p.track_id))
            else:
                raw_candidates.append(None)

        # Step B: Apply dwell-time smoothing
        smoothed: List[Optional[Tuple[str, int]]] = [None] * len(frames)
        current_confirmed: Optional[Tuple[str, int]] = None
        candidate: Optional[Tuple[str, int]] = None
        candidate_count = 0

        for i, raw_cand in enumerate(raw_candidates):
            if raw_cand == candidate:
                candidate_count += 1
            else:
                candidate = raw_cand
                candidate_count = 1

            if candidate_count >= self.possession_dwell:
                current_confirmed = candidate

            smoothed[i] = current_confirmed

        return smoothed

    def compute_smoothed_pressing_states(
        self, frames: List[FrameData], confirmed_possessions: List[Optional[Tuple[str, int]]]
    ) -> List[bool]:
        """
        Computes frame-by-frame high pressing indicator with dwell-time smoothing.
        Avoids fragmented press events caused by proximity flicker.
        """
        raw_pressing: List[bool] = []

        for frame, poss in zip(frames, confirmed_possessions):
            if not poss or not frame.ball:
                raw_pressing.append(False)
                continue

            ball_team, ball_track_id = poss

            # Count opposing defenders near ball carrier
            defenders_near = 0
            for p in frame.players:
                if p.team_id != ball_team:
                    d = distance(p.x, p.y, frame.ball.x, frame.ball.y)
                    if d <= self.pressing_max_dist:
                        defenders_near += 1

            raw_pressing.append(defenders_near >= self.pressing_min_defenders)

        # Apply dwell-time hysteresis smoothing
        smoothed_press: List[bool] = [False] * len(frames)
        count = 0

        for i, is_press in enumerate(raw_pressing):
            if is_press:
                count += 1
            else:
                count = 0

            if count >= self.pressing_dwell:
                # Backfill the dwell window so the entire sequence is marked
                for back in range(max(0, i - self.pressing_dwell + 1), i + 1):
                    smoothed_press[back] = True
            elif count > 0 and smoothed_press[i - 1]:
                # Keep active if it was already active
                smoothed_press[i] = True

        return smoothed_press
