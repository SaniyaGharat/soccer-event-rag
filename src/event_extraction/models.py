from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class TeamId(str, Enum):
    TEAM_A = "Team A"
    TEAM_B = "Team B"
    UNKNOWN = "Unknown"


class EventType(str, Enum):
    POSSESSION_CHANGE = "possession_change"
    COMPLETED_PASS = "completed_pass"
    TURNOVER = "turnover"
    PENALTY_BOX_ENTRY = "penalty_box_entry"
    SHOT_ON_GOAL = "shot_on_goal"
    PROXY_OFFSIDE = "proxy_offside"
    HIGH_PRESS_SEQUENCE = "high_press_sequence"
    POSSESSION_SPELL = "possession_spell"


class TrackedPlayer(BaseModel):
    track_id: int
    team_id: str
    jersey_number: Optional[int] = None
    x: float  # Pitch X in meters [0, 105]
    y: float  # Pitch Y in meters [0, 68]
    speed: float = 0.0  # Speed in m/s


class TrackedBall(BaseModel):
    x: float
    y: float
    z: float = 0.0
    speed: float = 0.0


class FrameData(BaseModel):
    frame_index: int
    timestamp_sec: float
    timestamp_str: str  # MM:SS or HH:MM:SS
    players: List[TrackedPlayer] = Field(default_factory=list)
    ball: Optional[TrackedBall] = None


class Event(BaseModel):
    event_id: str
    event_type: EventType
    start_frame: int
    end_frame: int
    start_timestamp: float  # in seconds
    end_timestamp: float    # in seconds
    start_timestamp_str: str  # e.g. "01:15"
    end_timestamp_str: str    # e.g. "01:25"
    team_id: Optional[str] = None
    player_track_ids: List[int] = Field(default_factory=list)
    jersey_numbers: List[int] = Field(default_factory=list)
    pitch_zones: List[str] = Field(default_factory=list)
    text_description: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
