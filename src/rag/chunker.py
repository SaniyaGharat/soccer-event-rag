"""
Event-Adaptive Semantic Chunker with Multi-Event Narrative Chains

Rationale:
1. Short discrete events (pass, turnover, box entry, shot, offside) are chunked into tight 3-6s windows.
2. Sustained states (possession spells, pressing sequences) are chunked over their natural duration.
3. Multi-Event Narrative Chains group temporally adjacent events (occurring within 12 seconds of each other)
   into a single compound document, allowing the RAG pipeline to answer complex sequential queries.
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
    Also builds compound Multi-Event Narrative Chains.
    """

    def __init__(
        self,
        instantaneous_window_before_sec: float = 2.0,
        instantaneous_window_after_sec: float = 4.0,
        narrative_chain_gap_sec: float = 12.0
    ):
        self.win_before = instantaneous_window_before_sec
        self.win_after = instantaneous_window_after_sec
        self.narrative_gap = narrative_chain_gap_sec

    def chunk_events(self, events: List[Event]) -> List[Document]:
        """
        Converts list of Event objects into adaptive LangChain Documents,
        including individual events and compound multi-event narrative chains.
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

        # 1. Generate Individual Event Documents
        for evt in events:
            if evt.event_type in short_event_types:
                chunk_start_sec = max(0.0, evt.start_timestamp - self.win_before)
                chunk_end_sec = evt.end_timestamp + self.win_after
            else:
                chunk_start_sec = evt.start_timestamp
                chunk_end_sec = evt.end_timestamp

            chunk_start_str = format_seconds_to_timestamp(chunk_start_sec)
            chunk_end_str = format_seconds_to_timestamp(chunk_end_sec)

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

        # 2. Generate Multi-Event Narrative Chain Documents
        # Sort events to find sequential chains
        sorted_events = sorted(events, key=lambda e: e.start_timestamp)
        current_chain: List[Event] = []

        for evt in sorted_events:
            if not current_chain:
                current_chain.append(evt)
                continue

            last_evt = current_chain[-1]
            time_gap = evt.start_timestamp - last_evt.end_timestamp

            # Chain events if they occur close to each other
            if 0 <= time_gap <= self.narrative_gap:
                current_chain.append(evt)
            else:
                if len(current_chain) >= 2:
                    documents.append(self._create_narrative_document(current_chain))
                current_chain = [evt]

        # Handle remaining chain
        if len(current_chain) >= 2:
            documents.append(self._create_narrative_document(current_chain))

        return documents

    def _create_narrative_document(self, chain_events: List[Event]) -> Document:
        """Helper to create a single compound Document from a list of sequential events."""
        start_sec = min(e.start_timestamp for e in chain_events)
        end_sec = max(e.end_timestamp for e in chain_events)
        start_str = format_seconds_to_timestamp(start_sec)
        end_str = format_seconds_to_timestamp(end_sec)

        event_sequence = " -> ".join(e.event_type.value.replace("_", " ").title() for e in chain_events)
        descriptions = " Then ".join(e.text_description.split("] ")[-1] for e in chain_events)

        all_zones = set()
        all_teams = set()
        all_jerseys = set()
        for e in chain_events:
            all_zones.update(e.pitch_zones)
            if e.team_id:
                all_teams.add(e.team_id)
            all_jerseys.update(e.jersey_numbers)

        page_content = (
            f"Time Window [{start_str} - {end_str}] | "
            f"Multi-Event Narrative Chain | "
            f"Sequence: {event_sequence} | "
            f"Teams: {', '.join(all_teams) if all_teams else 'N/A'} | "
            f"Pitch Zones: {', '.join(all_zones)} | "
            f"Details: {descriptions}"
        )

        metadata = {
            "event_id": f"chain_{int(start_sec)}_{int(end_sec)}",
            "event_type": "multi_event_narrative",
            "start_timestamp": start_sec,
            "end_timestamp": end_sec,
            "start_timestamp_str": start_str,
            "end_timestamp_str": end_str,
            "team_id": ", ".join(all_teams) if all_teams else "N/A",
            "pitch_zones": ", ".join(all_zones),
            "jersey_numbers": ", ".join(map(str, all_jerseys)),
            "raw_description": f"Narrative sequence: {descriptions}"
        }

        return Document(page_content=page_content, metadata=metadata)
