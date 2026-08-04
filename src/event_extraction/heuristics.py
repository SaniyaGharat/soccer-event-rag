"""
Rule-Based Event Extraction Heuristics (Stage 2)

Extracts discrete human-readable events from preprocessed tracking frames:
- Possession Spells
- Completed Passes
- Turnovers / Interceptions
- Penalty Box Entries
- Shots on Goal
- High Pressing Sequences
- Simplified Offside Proxy (Frame-window approximation)
"""

import json
import uuid
import math
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from src.event_extraction.models import FrameData, Event, EventType, TrackedPlayer, TrackedBall


def get_pitch_zone(x: float, y: float, pitch_config: Dict[str, Any]) -> List[str]:
    """Determines pitch zones matching given (x, y) coordinates."""
    zones = pitch_config.get("zones", {})
    matched_zones = []

    # Defensive/Midfield/Attacking Third check
    if x <= 35.0:
        matched_zones.append("Defensive Third")
    elif x <= 70.0:
        matched_zones.append("Midfield")
    else:
        matched_zones.append("Attacking Third")

    # Specific Zone Overlays
    for zone_name, bounds in zones.items():
        if zone_name in ["Defensive Third", "Midfield", "Attacking Third"]:
            continue
        if bounds["x_min"] <= x <= bounds["x_max"] and bounds["y_min"] <= y <= bounds["y_max"]:
            matched_zones.append(zone_name)

    return matched_zones if matched_zones else ["Midfield"]


class EventExtractor:
    """
    Decoupled rule-based event extractor layer.
    """

    def __init__(
        self,
        pitch_config_path: Optional[str] = None,
        pass_min_velocity: float = 3.5
    ):
        self.pass_min_velocity = pass_min_velocity
        self.pitch_config = self._load_pitch_config(pitch_config_path)

    def _load_pitch_config(self, path: Optional[str]) -> Dict[str, Any]:
        if path and Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "pitch_length_meters": 105.0,
            "pitch_width_meters": 68.0,
            "zones": {
                "Defensive Third": {"x_min": 0.0, "x_max": 35.0, "y_min": 0.0, "y_max": 68.0},
                "Midfield": {"x_min": 35.0, "x_max": 70.0, "y_min": 0.0, "y_max": 68.0},
                "Attacking Third": {"x_min": 70.0, "x_max": 105.0, "y_min": 0.0, "y_max": 68.0},
                "Defensive Penalty Box": {"x_min": 0.0, "x_max": 16.5, "y_min": 13.84, "y_max": 54.16},
                "Attacking Penalty Box": {"x_min": 88.5, "x_max": 105.0, "y_min": 13.84, "y_max": 54.16},
                "Left Flank": {"x_min": 0.0, "x_max": 105.0, "y_min": 0.0, "y_max": 20.0},
                "Right Flank": {"x_min": 0.0, "x_max": 105.0, "y_min": 48.0, "y_max": 68.0}
            }
        }

    def extract_events(
        self,
        frames: List[FrameData],
        confirmed_possessions: List[Optional[Tuple[str, int]]],
        smoothed_pressing: List[bool]
    ) -> List[Event]:
        """
        Extracts all discrete events across the preprocessed frames.
        """
        events: List[Event] = []

        # 1. Passes & Turnovers
        pass_events, turnover_events = self._extract_passes_and_turnovers(
            frames, confirmed_possessions
        )
        events.extend(pass_events)
        events.extend(turnover_events)

        # 2. Penalty Box Entries
        box_events = self._extract_penalty_box_entries(frames)
        events.extend(box_events)

        # 3. Shots on Goal
        shot_events = self._extract_shots_on_goal(frames)
        events.extend(shot_events)

        # 4. Simplified Offside Proxy
        offside_events = self._extract_proxy_offsides(frames, pass_events)
        events.extend(offside_events)

        # 5. Continuous Pressing Sequences
        pressing_events = self._extract_pressing_sequences(frames, smoothed_pressing, confirmed_possessions)
        events.extend(pressing_events)

        # 6. Continuous Possession Spells
        spell_events = self._extract_possession_spells(frames, confirmed_possessions)
        events.extend(spell_events)

        # Sort all events chronologically by start_timestamp
        events.sort(key=lambda e: e.start_timestamp)
        return events

    def _extract_passes_and_turnovers(
        self,
        frames: List[FrameData],
        possessions: List[Optional[Tuple[str, int]]]
    ) -> Tuple[List[Event], List[Event]]:
        passes: List[Event] = []
        turnovers: List[Event] = []

        last_poss: Optional[Tuple[str, int, int]] = None  # (team_id, track_id, frame_idx)

        for idx, (frame, poss) in enumerate(zip(frames, possessions)):
            if poss is None:
                continue

            team, track_id = poss

            if last_poss is not None:
                prev_team, prev_track_id, prev_frame_idx, prev_frame_obj = last_poss
                frame_diff = frame.frame_index - prev_frame_obj.frame_index

                # Check if possession transferred within a reasonable window (e.g. 5-75 frames / 0.2s - 3s)
                if 5 <= frame_diff <= 75 and (prev_team != team or prev_track_id != track_id):
                    start_f = prev_frame_obj
                    end_f = frame

                    # Ball positions
                    ball_start = start_f.ball
                    ball_end = end_f.ball

                    zones = []
                    if ball_start:
                        zones.extend(get_pitch_zone(ball_start.x, ball_start.y, self.pitch_config))
                    if ball_end:
                        zones.extend(get_pitch_zone(ball_end.x, ball_end.y, self.pitch_config))
                    unique_zones = list(dict.fromkeys(zones))

                    # Get jersey numbers
                    prev_jersey = next((p.jersey_number for p in start_f.players if p.track_id == prev_track_id), None)
                    curr_jersey = next((p.jersey_number for p in end_f.players if p.track_id == track_id), None)

                    jerseys = [j for j in [prev_jersey, curr_jersey] if j is not None]

                    if prev_team == team:
                        # Completed Pass
                        passes.append(Event(
                            event_id=f"pass_{uuid.uuid4().hex[:8]}",
                            event_type=EventType.COMPLETED_PASS,
                            start_frame=start_f.frame_index,
                            end_frame=end_f.frame_index,
                            start_timestamp=start_f.timestamp_sec,
                            end_timestamp=end_f.timestamp_sec,
                            start_timestamp_str=start_f.timestamp_str,
                            end_timestamp_str=end_f.timestamp_str,
                            team_id=team,
                            player_track_ids=[prev_track_id, track_id],
                            jersey_numbers=jerseys,
                            pitch_zones=unique_zones,
                            text_description=f"[{start_f.timestamp_str} - {end_f.timestamp_str}] Completed pass by {team} "
                                             f"from Player #{prev_jersey or prev_track_id} to Player #{curr_jersey or track_id} "
                                             f"in {', '.join(unique_zones)}.",
                            metadata={"sender_id": prev_track_id, "receiver_id": track_id}
                        ))
                    else:
                        # Turnover / Interception
                        turnovers.append(Event(
                            event_id=f"turnover_{uuid.uuid4().hex[:8]}",
                            event_type=EventType.TURNOVER,
                            start_frame=start_f.frame_index,
                            end_frame=end_f.frame_index,
                            start_timestamp=start_f.timestamp_sec,
                            end_timestamp=end_f.timestamp_sec,
                            start_timestamp_str=start_f.timestamp_str,
                            end_timestamp_str=end_f.timestamp_str,
                            team_id=team,
                            player_track_ids=[prev_track_id, track_id],
                            jersey_numbers=jerseys,
                            pitch_zones=unique_zones,
                            text_description=f"[{start_f.timestamp_str} - {end_f.timestamp_str}] Turnover in {', '.join(unique_zones)}! "
                                             f"Possession lost by {prev_team} (Player #{prev_jersey or prev_track_id}) to "
                                             f"{team} (Player #{curr_jersey or track_id}).",
                            metadata={"lost_by_team": prev_team, "won_by_team": team}
                        ))

            last_poss = (team, track_id, idx, frame)

        return passes, turnovers

    def _extract_penalty_box_entries(self, frames: List[FrameData]) -> List[Event]:
        box_events: List[Event] = []
        in_box = False
        start_frame: Optional[FrameData] = None

        for frame in frames:
            if not frame.ball:
                continue

            zones = get_pitch_zone(frame.ball.x, frame.ball.y, self.pitch_config)
            is_in_box = any("Penalty Box" in z for z in zones)

            if is_in_box and not in_box:
                in_box = True
                start_frame = frame
            elif not is_in_box and in_box and start_frame:
                end_frame = frame
                dur = end_frame.timestamp_sec - start_frame.timestamp_sec

                if dur >= 1.0:  # Box entry duration
                    box_events.append(Event(
                        event_id=f"box_entry_{uuid.uuid4().hex[:8]}",
                        event_type=EventType.PENALTY_BOX_ENTRY,
                        start_frame=start_frame.frame_index,
                        end_frame=end_frame.frame_index,
                        start_timestamp=start_frame.timestamp_sec,
                        end_timestamp=end_frame.timestamp_sec,
                        start_timestamp_str=start_frame.timestamp_str,
                        end_timestamp_str=end_frame.timestamp_str,
                        pitch_zones=zones,
                        text_description=f"[{start_frame.timestamp_str} - {end_frame.timestamp_str}] Ball crossed into the penalty box area."
                    ))
                in_box = False
                start_frame = None

        return box_events

    def _extract_shots_on_goal(self, frames: List[FrameData]) -> List[Event]:
        shots: List[Event] = []
        for i in range(len(frames) - 5):
            f1 = frames[i]
            f5 = frames[i + 5]

            if not f1.ball or not f5.ball:
                continue

            # Shot velocity check
            dist = math.sqrt((f5.ball.x - f1.ball.x)**2 + (f5.ball.y - f1.ball.y)**2)
            speed = dist / 0.2  # 5 frames @ 25fps = 0.2s

            if speed >= 12.0:  # High velocity ball towards goal
                # Check if moving towards goal (x > 95 or x < 10)
                moving_to_attacking_goal = (f5.ball.x > 90.0 and f5.ball.x > f1.ball.x)
                moving_to_defensive_goal = (f5.ball.x < 15.0 and f5.ball.x < f1.ball.x)

                if moving_to_attacking_goal or moving_to_defensive_goal:
                    zones = get_pitch_zone(f1.ball.x, f1.ball.y, self.pitch_config)
                    shots.append(Event(
                        event_id=f"shot_{uuid.uuid4().hex[:8]}",
                        event_type=EventType.SHOT_ON_GOAL,
                        start_frame=f1.frame_index,
                        end_frame=f5.frame_index,
                        start_timestamp=f1.timestamp_sec,
                        end_timestamp=f5.timestamp_sec,
                        start_timestamp_str=f1.timestamp_str,
                        end_timestamp_str=f5.timestamp_str,
                        pitch_zones=zones,
                        text_description=f"[{f1.timestamp_str} - {f5.timestamp_str}] Shot on goal attempt! High-velocity strike ({speed:.1f} m/s) towards goal."
                    ))
                    i += 15  # Skip ahead to prevent duplicate shot triggers
        return shots

    def _extract_proxy_offsides(self, frames: List[FrameData], passes: List[Event]) -> List[Event]:
        """
        SIMPLIFIED OFFSIDE PROXY (v1 Heuristic):
        Checks if an attacking player is positioned ahead of the 2nd-to-last defender
        during a forward pass frame-window.
        Note: This is a loose proxy heuristic, not a rules-accurate VAR offside engine.
        """
        offside_events: List[Event] = []
        frame_dict = {f.frame_index: f for f in frames}

        for pass_evt in passes:
            start_f = frame_dict.get(pass_evt.start_frame)
            if not start_f or not pass_evt.team_id:
                continue

            attacking_team = pass_evt.team_id
            defending_team = "Team B" if attacking_team == "Team A" else "Team A"

            # Sort defenders by X coordinate to find second-to-last defender
            defenders = [p for p in start_f.players if p.team_id == defending_team]
            attackers = [p for p in start_f.players if p.team_id == attacking_team]

            if len(defenders) < 2 or not attackers:
                continue

            # Assuming Team A attacks towards X=105, Team B attacks towards X=0
            if attacking_team == "Team A":
                defenders_sorted = sorted(defenders, key=lambda p: p.x, reverse=True)
                second_last_def_x = defenders_sorted[1].x
                # Find attackers beyond second-last defender
                offside_attackers = [a for a in attackers if a.x > second_last_def_x + 0.5]
            else:
                defenders_sorted = sorted(defenders, key=lambda p: p.x)
                second_last_def_x = defenders_sorted[1].x
                offside_attackers = [a for a in attackers if a.x < second_last_def_x - 0.5]

            if offside_attackers:
                off_jersey = offside_attackers[0].jersey_number or offside_attackers[0].track_id
                offside_events.append(Event(
                    event_id=f"offside_{uuid.uuid4().hex[:8]}",
                    event_type=EventType.PROXY_OFFSIDE,
                    start_frame=start_f.frame_index,
                    end_frame=pass_evt.end_frame,
                    start_timestamp=start_f.timestamp_sec,
                    end_timestamp=pass_evt.end_timestamp,
                    start_timestamp_str=start_f.timestamp_str,
                    end_timestamp_str=pass_evt.end_timestamp_str,
                    team_id=attacking_team,
                    jersey_numbers=[off_jersey],
                    pitch_zones=pass_evt.pitch_zones,
                    text_description=f"[{start_f.timestamp_str} - {pass_evt.end_timestamp_str}] Potential offside-adjacent situation! "
                                     f"{attacking_team} Player #{off_jersey} positioned ahead of the defensive line during forward pass.",
                    metadata={"proxy_heuristic": True, "offside_player_id": offside_attackers[0].track_id}
                ))

        return offside_events

    def _extract_pressing_sequences(
        self,
        frames: List[FrameData],
        smoothed_pressing: List[bool],
        possessions: List[Optional[Tuple[str, int]]]
    ) -> List[Event]:
        events: List[Event] = []
        in_press = False
        start_idx = 0

        for i, is_press in enumerate(smoothed_pressing):
            if is_press and not in_press:
                in_press = True
                start_idx = i
            elif not is_press and in_press:
                end_idx = i - 1
                start_f = frames[start_idx]
                end_f = frames[end_idx]
                duration = end_f.timestamp_sec - start_f.timestamp_sec

                if duration >= 3.0:
                    poss = possessions[start_idx]
                    pressed_team = poss[0] if poss else "attraction team"
                    zones = get_pitch_zone(start_f.ball.x, start_f.ball.y, self.pitch_config) if start_f.ball else ["Midfield"]

                    events.append(Event(
                        event_id=f"press_{uuid.uuid4().hex[:8]}",
                        event_type=EventType.HIGH_PRESS_SEQUENCE,
                        start_frame=start_f.frame_index,
                        end_frame=end_f.frame_index,
                        start_timestamp=start_f.timestamp_sec,
                        end_timestamp=end_f.timestamp_sec,
                        start_timestamp_str=start_f.timestamp_str,
                        end_timestamp_str=end_f.timestamp_str,
                        pitch_zones=zones,
                        text_description=f"[{start_f.timestamp_str} - {end_f.timestamp_str}] High pressing sequence! "
                                         f"Multiple defenders closing down ball carrier ({pressed_team}) in {', '.join(zones)}.",
                        metadata={"duration_sec": duration}
                    ))
                in_press = False

        return events

    def _extract_possession_spells(
        self,
        frames: List[FrameData],
        possessions: List[Optional[Tuple[str, int]]]
    ) -> List[Event]:
        events: List[Event] = []
        curr_team: Optional[str] = None
        start_idx: int = 0

        for i, poss in enumerate(possessions):
            team = poss[0] if poss else None
            if team != curr_team:
                if curr_team is not None:
                    end_idx = i - 1
                    start_f = frames[start_idx]
                    end_f = frames[end_idx]
                    dur = end_f.timestamp_sec - start_f.timestamp_sec

                    if dur >= 10.0:  # Sustained possession spell >= 10s
                        zones = get_pitch_zone(start_f.ball.x, start_f.ball.y, self.pitch_config) if start_f.ball else ["Midfield"]
                        events.append(Event(
                            event_id=f"spell_{uuid.uuid4().hex[:8]}",
                            event_type=EventType.POSSESSION_SPELL,
                            start_frame=start_f.frame_index,
                            end_frame=end_f.frame_index,
                            start_timestamp=start_f.timestamp_sec,
                            end_timestamp=end_f.timestamp_sec,
                            start_timestamp_str=start_f.timestamp_str,
                            end_timestamp_str=end_f.timestamp_str,
                            team_id=curr_team,
                            pitch_zones=zones,
                            text_description=f"[{start_f.timestamp_str} - {end_f.timestamp_str}] Sustained possession spell by {curr_team} "
                                             f"lasting {dur:.1f} seconds across {', '.join(zones)}.",
                            metadata={"duration_sec": dur}
                        ))
                curr_team = team
                start_idx = i

        return events
