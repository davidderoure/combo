"""Feature extraction from a Timeline — the listening side of DESIGN.md §5.

Each function below is a small pure function over a Timeline, a voice_id, and a beat
window — reusable by any future accompanist, not tied to ensemble/comping.py
specifically. Only `density` is actually consumed by this build's accompanist
(ensemble/comping.py); `pitch_range`, `average_velocity`, and `beats_of_silence` are
extracted and tested here because DESIGN.md §5 names all four ("density, register,
dynamics, space/rests") as the features an accompanist should have available, but
nothing yet consumes the other three. Said plainly here rather than left implicit.
"""

import random
from typing import List, Optional, Tuple

from .timeline import BEATS_PER_BAR, NoteEvent, Timeline
from .voice import Generator


def _events_for(timeline: Timeline, voice_id: str, since_beat: float, until_beat: float) -> List[NoteEvent]:
    return [e for e in timeline if e.voice_id == voice_id and since_beat <= e.start_beat < until_beat]


def density(timeline: Timeline, voice_id: str, since_beat: float, until_beat: float) -> float:
    """Notes per beat for voice_id in [since_beat, until_beat). 0.0 for an empty or
    zero-width window — silence, not an error."""
    span = until_beat - since_beat
    if span <= 0:
        return 0.0
    return len(_events_for(timeline, voice_id, since_beat, until_beat)) / span


def pitch_range(
    timeline: Timeline, voice_id: str, since_beat: float, until_beat: float
) -> Optional[Tuple[int, int]]:
    """(min_pitch, max_pitch) for voice_id's notes in the window, or None if it
    played nothing there."""
    pitches = [e.pitch for e in _events_for(timeline, voice_id, since_beat, until_beat)]
    return (min(pitches), max(pitches)) if pitches else None


def average_velocity(
    timeline: Timeline, voice_id: str, since_beat: float, until_beat: float
) -> Optional[float]:
    velocities = [e.velocity for e in _events_for(timeline, voice_id, since_beat, until_beat)]
    return sum(velocities) / len(velocities) if velocities else None


def beats_of_silence(timeline: Timeline, voice_id: str, until_beat: float) -> float:
    """Beats since voice_id's last note before until_beat. inf if it hasn't played
    anything at all before until_beat — genuinely unbounded silence, not a guess."""
    starts = [e.start_beat for e in timeline if e.voice_id == voice_id and e.start_beat < until_beat]
    return until_beat - max(starts) if starts else float("inf")


# Test/demo fixture, NOT a real instrument voice. chord_tone_generator plays exactly
# 4 notes every single bar with zero density variation ever, so there's nothing for
# an accompanist to react to when demonstrating duck/fill — this fixture alternates
# busy/sparse/moderate bars on a 3-bar cycle so that behaviour is actually visible,
# in both tests and ensemble/demo.py.
def synthetic_varying_density_generator(seed: Optional[int] = None) -> Generator:
    rng = random.Random(seed)

    def generate(song, bar_index: int, timeline: Timeline) -> List[NoteEvent]:
        cycle = bar_index % 3
        if cycle == 0:
            offsets = [i * 0.5 for i in range(8)]  # busy: 8 hits/bar
        elif cycle == 1:
            offsets = []  # sparse: silence
        else:
            offsets = [0.0, 2.0]  # moderate: 2 hits/bar

        events = []
        for offset in offsets:
            start = bar_index * BEATS_PER_BAR + offset + rng.uniform(-0.01, 0.01)
            events.append(
                NoteEvent(voice_id="", pitch=60, velocity=80, start_beat=start, duration_beats=0.2)
            )
        return events

    return generate
