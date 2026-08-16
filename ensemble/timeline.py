"""Symbolic (beat-based, not wall-clock) note output from an ensemble session."""

from dataclasses import dataclass, field
from typing import List

BEATS_PER_BAR = 4.0  # matches song/chart.py's assumption of 4/4 throughout


@dataclass(frozen=True)
class NoteEvent:
    voice_id: str
    pitch: int  # MIDI note number
    velocity: int  # 0-127
    start_beat: float
    duration_beats: float


@dataclass
class Timeline:
    events: List[NoteEvent] = field(default_factory=list)

    def add(self, event: NoteEvent) -> None:
        self.events.append(event)

    def merge(self, other: "Timeline") -> "Timeline":
        return Timeline(sorted(self.events + other.events, key=lambda e: e.start_beat))

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)
