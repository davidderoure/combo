"""A harmonic progression: one chorus' worth of chord changes, cycled
repeatedly by a Song's form (see song.py)."""

from dataclasses import dataclass
from typing import List

from .chord import Chord


@dataclass(frozen=True)
class ChangesEvent:
    chord: Chord
    duration_beats: float


@dataclass(frozen=True)
class Changes:
    events: List[ChangesEvent]

    @property
    def total_beats(self) -> float:
        return sum(e.duration_beats for e in self.events)

    def chord_at(self, beat: float) -> Chord:
        """`beat` is taken modulo total_beats — Changes is one cycle."""
        if not self.events:
            raise ValueError("Changes has no events")
        position = beat % self.total_beats
        elapsed = 0.0
        for event in self.events:
            elapsed += event.duration_beats
            if position < elapsed:
                return event.chord
        return self.events[-1].chord

    def transpose(self, semitones: int) -> "Changes":
        return Changes([
            ChangesEvent(e.chord.transpose(semitones), e.duration_beats)
            for e in self.events
        ])
