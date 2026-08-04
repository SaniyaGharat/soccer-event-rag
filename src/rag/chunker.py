"""
Event-Adaptive Semantic Chunker

Rationale:
Uniform fixed 10-15 second chunking dilutes instantaneous events (e.g. 2-second passes or turnovers)
with irrelevant background frames, while truncating long continuous states (e.g. 30-second possession spells).
Instead, chunking is tailored adaptively to event nature:
- Short discrete events (pass, turnover, box entry, shot, offside): Encapsulated in tight 3-6s windows.
- Sustained continuous states (possession spell, pressing sequence): Grouped over their natural duration.
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from src.event_extraction.models import Event, EventType


def format_seconds_to_timestamp(seconds: float) -> str:
    """Formats seconds to MM:SS string."""
    tot = int(max(0, seconds))
    m, s = divmod(tot, 60)
    return f"{m:02d}:{s:02d}"


class EventAdaptiveChunker:
    """
    Transforms extracted Event data models into structured LangChain Documents with event-adaptive windows.
    """

    def __init__(
        self,
        instantaneous_window_before_sec: float = 2.0,
        instantaneous_window_after_sec: float = 4.0
    ):
        self.win_before = instantaneous_window_before_sec
        self.win_after = instantaneous_window_after_sec

    def chunk_events(self, events: List[Event]) -> List[Document]:
        """
        Converts list of Event objects into adaptive LangChain Documents with metadata.
        """
        documents: List[Document] = []

        # Instantaneous event types requiring tight windowing
        short_event_types = {
            EventType.COMPLETED_PASS,
            EventType.TURNOVER,
            EventType.PENALTY_BOX_ENTRY,
            EventType.SHOT_ON_GOAL,
            EventType.PROXY_OFFSIDE
        }

        for evt in events:
            if evt.event_type in short_event_types:
                # Tight window around event time
                chunk_start_sec = max(0.0, evt.start_timestamp - self.win_before)
                chunk_end_sec = evt.end_timestamp + self.win_after
            else:
                # Continuous sustained spell duration
                chunk_start_sec = evt.start_timestamp
                chunk_end_sec = evt.end_timestamp

            chunk_start_str = format_seconds_to_timestamp(chunk_start_sec)
            chunk_end_str = format_seconds_to_timestamp(chunk_end_sec)

            # Enrich page content for vector embedding
            page_content = (
                f"Time Window [{chunk_start_str} - {chunk_end_str}] | "
                f"Event Type: {evt.event_type.value} | "
                f"Team: {evt.team_id or 'N/A'} | "
                f"Pitch Zones: {', '.join(evt.pitch_zones)} | "
                f"Details: {evt.text_description}"
            )

            metadata = {
                "event_id": evt.event_id,
                "event_type": evt.event_type.value,
                "start_timestamp": chunk_start_sec,
                "end_timestamp": chunk_end_sec,
                "start_timestamp_str": chunk_start_str,
                "end_timestamp_str": chunk_end_str,
                "team_id": evt.team_id or "N/A",
                "pitch_zones": ", ".join(evt.pitch_zones),
                "jersey_numbers": ", ".join(map(str, evt.jersey_numbers)),
                "raw_description": evt.text_description
            }

            documents.append(Document(page_content=page_content, metadata=metadata))

        return documents
