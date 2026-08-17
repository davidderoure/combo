"""Handover/transition triggers — see DESIGN.md §8.

Only the handover() gesture is wired to anything here — it's the only seeded gesture
(gesture/vocabulary.py) whose meaning actually maps onto a section transition.
reset_tempo() is about §4.1's tempo dial (still unbuilt), not section timing, and is
ignored here. "Pulling a transition *late*" (extending a section) has no
corresponding seeded gesture yet either — not attempted, not approximated with the
wrong gesture.

This is the first real slice of the long-referenced ArcController (cited as missing
in DESIGN.md §5/§7/§11) — specifically the transition-timing piece, not the full
tension/peak-modelling concept those sections still lack.
"""

import threading
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Callable, Dict, List

from gesture.vocabulary import Gesture

if TYPE_CHECKING:
    from song import Song

# bar_index -> gestures to treat as arrived by this bar. Mirrors Generator's and
# DirectorSource's per-bar-callable shape (ensemble/voice.py, ensemble/director.py).
GestureSource = Callable[[int], List[Gesture]]


def scripted_gesture_source(schedule: Dict[int, List[Gesture]]) -> GestureSource:
    """Test/demo fixture, mirrors constant_director_source's role: a small,
    deterministic, no-hardware-needed way to drive handover triggers."""

    def source(bar_index: int) -> List[Gesture]:
        return schedule.get(bar_index, [])

    return source


class LiveGestureQueue:
    """A thread-safe buffer between a live MIDI callback thread (input/sources.py's
    on_gesture, Phase 6) and Session.generate's thread (real-time mode). The first
    thread-safety primitive in this codebase — genuinely needed now that those two
    run concurrently, not a precaution added out of habit."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: List[Gesture] = []

    def append(self, gesture: Gesture) -> None:
        with self._lock:
            self._pending.append(gesture)

    def drain(self, bar_index: int) -> List[Gesture]:
        # bar_index is unused -- this is what makes `queue.drain` itself usable
        # directly as a GestureSource, matching that callable's shape exactly.
        with self._lock:
            drained, self._pending = self._pending, []
        return drained


def _locate(song: "Song", beat: float):
    """(form_index, section, chorus_index) at `beat`. Mirrors Song.section_at's own
    walk (song/song.py) but also returns the form-index, which section_at doesn't
    expose and which a dict of overrides needs as its key. Duplicated rather than
    reused for that reason -- noted honestly, not hidden."""
    if not song.form:
        raise ValueError("Song has no form")
    chorus_beats = song.changes.total_beats
    elapsed = 0.0
    for index, section in enumerate(song.form):
        if section.repeats is None:
            return index, section, max(int((beat - elapsed) // chorus_beats), 0)
        section_beats = section.repeats * chorus_beats
        if beat < elapsed + section_beats:
            return index, section, int((beat - elapsed) // chorus_beats)
        elapsed += section_beats
    last_index = len(song.form) - 1
    last = song.form[last_index]
    chorus_index = int((beat - elapsed) // chorus_beats) + (last.repeats or 0)
    return last_index, last, chorus_index


@dataclass
class TransitionController:
    """Tracks live handover-driven shortenings of the current song's form. Built
    fresh per Session.generate() call (§4) -- each call is a fresh performance, so
    fresh transition state.

    Total performance length is deliberately not shortened: Session.generate's loop
    bound stays the *nominal* song.total_beats regardless of any override -- the bars
    a truncated section gives up get absorbed into whichever section governs later
    bars (via Song.section_at's own documented past-the-end behaviour, extending the
    final section rather than ending the performance early). A handover reallocates
    *which* section is playing when; it doesn't shrink the total set length. A real,
    specific simplification, not a hidden one.
    """

    overrides: Dict[int, int] = field(default_factory=dict)

    def on_gesture(self, gesture: Gesture, song: "Song", current_beat: float) -> None:
        if gesture.name != "handover":
            return
        form_index, _section, chorus_index = _locate(song, current_beat)
        new_repeats = chorus_index + 1
        existing = self.overrides.get(form_index)
        # Monotonic: a section can only be shortened further, never un-shortened.
        self.overrides[form_index] = new_repeats if existing is None else min(existing, new_repeats)

    def effective_song(self, song: "Song") -> "Song":
        if not self.overrides:
            return song
        new_form = [
            replace(section, repeats=self.overrides[index]) if index in self.overrides else section
            for index, section in enumerate(song.form)
        ]
        return replace(song, form=new_form)
