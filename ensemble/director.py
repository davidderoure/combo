"""The musical director's dial channel, plus the data model for its gesture channel
— see DESIGN.md §11.

Scope, stated up front: §11 as designed is large (a continuously-running critic, two
channels, N human-or-AI instances, real-time nudge *and* batch scoring). This module
builds the dial channel end-to-end, with a real consumer (ensemble/comping.py), and
the gesture channel's *data model and aggregation only* — DirectorSignal can carry a
Gesture and aggregate_director_signals() handles it sensibly, but nothing consumes a
director-emitted gesture yet, because it has nowhere to act: §4.1's runtime-tempo and
§8's handover-triggered transitions are both still design-only. Batch-mode scoring and
live human/MIDI director input aren't attempted at all here.

This is the first module where `ensemble` imports from `gesture` — a real integration
point between two previously-independent packages, not a smell.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional

from gesture.vocabulary import Gesture

from .listening import density
from .timeline import BEATS_PER_BAR, Timeline

if TYPE_CHECKING:
    from song import Song

NEUTRAL_INTENSITY = 0.5  # "no director present" — every generator's default behaviour


@dataclass(frozen=True)
class DirectorSignal:
    intensity: float = NEUTRAL_INTENSITY
    gesture: Optional[Gesture] = None


# (song, bar_index, timeline-of-prior-bars) -> a signal for this bar. Exactly
# Generator's shape (ensemble/voice.py), returning a DirectorSignal instead of notes.
DirectorSource = Callable[["Song", int, Timeline], DirectorSignal]


@dataclass
class Director:
    id: str
    source: str  # "human" or "ai"
    signal_source: Optional[DirectorSource] = None

    def __post_init__(self) -> None:
        if self.source not in ("human", "ai"):
            raise ValueError(f"source must be 'human' or 'ai', got {self.source!r}")
        if self.source == "ai" and self.signal_source is None:
            raise ValueError(f"AI-sourced director {self.id!r} needs a signal_source")


def aggregate_director_signals(signals: List[DirectorSignal]) -> DirectorSignal:
    """Empty -> the neutral default (no director present). Otherwise the mean
    intensity (DESIGN.md §11: "start simple, e.g. a weighted average" — unweighted
    here; weighting is future work) and the first non-None gesture, if any (a simple
    placeholder tie-break for the rare case of two directors gesturing in the same
    bar, not a considered conflict-resolution design)."""
    if not signals:
        return DirectorSignal()
    mean_intensity = sum(s.intensity for s in signals) / len(signals)
    gesture = next((s.gesture for s in signals if s.gesture is not None), None)
    return DirectorSignal(intensity=mean_intensity, gesture=gesture)


def constant_director_source(intensity: float) -> DirectorSource:
    """Stands in for a human holding a dial at a fixed position — the simplest
    possible source, for tests and the demo."""

    def source(song, bar_index: int, timeline: Timeline) -> DirectorSignal:
        return DirectorSignal(intensity=intensity)

    return source


# Deliberately simple normalisation reference, same status as comping.py's
# thresholds or drums.py's density tiers: a placeholder to prove the mechanism,
# needing real tuning once there's real (non-stub) generation to tune it against.
REFERENCE_MAX_DENSITY = 3.0  # notes/beat, summed across voice_ids, treated as "1.0 intensity"


def ensemble_intensity_critic(voice_ids: List[str], lookback_bars: int = 2) -> DirectorSource:
    """A genuine, if deliberately simple, AI critic (DESIGN.md §11's own phrase):
    intensity tracks the ensemble's own combined density across voice_ids over the
    recent bars, normalised against REFERENCE_MAX_DENSITY."""

    def source(song, bar_index: int, timeline: Timeline) -> DirectorSignal:
        since_beat = max(0, bar_index - lookback_bars) * BEATS_PER_BAR
        until_beat = bar_index * BEATS_PER_BAR
        combined_density = sum(density(timeline, voice_id, since_beat, until_beat) for voice_id in voice_ids)
        intensity = max(0.0, min(1.0, combined_density / REFERENCE_MAX_DENSITY))
        return DirectorSignal(intensity=intensity)

    return source
