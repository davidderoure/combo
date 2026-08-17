"""Starts MIDI input for every role-tagged source in config.MIDI_SOURCES — the
integration point between raw MIDI (rtmidi) and what the rest of the system
consumes: named gestures (gesture/vocabulary.py) for performers, a live
DirectorSignal (ensemble/director.py) for directors. See DESIGN.md §6/§11.

Since Phase 13: role determines DESTINATION, not CAPABILITY. Every source gets
the same MidiListener + GestureRecognizer (input/midi_listener.py) — a director
sitting at a keyboard uses the same recognition machinery a performer does ("dual
control car"). Only where the recognised output is routed differs by role:
performer gestures reach the caller's on_gesture callback (bound for
Session.gesture_source / TransitionController); director gestures are latched
and consumed once per DirectorSignal poll, alongside the same live-CC intensity
DirectorMidiListener used to provide alone. DirectorMidiListener (CC-only) is
retired — its logic is now just MidiListener's cc_number path, available to
every source, not a role-specific class.

Honest limitation: there's no MIDI hardware attached in this development
environment. cc_value_to_intensity, MidiSourceConfig's validation, and
MidiListener's callback logic are all tested directly (synthetic MIDI byte
tuples, no real port needed — the same technique gesture/recognizer.py's own tests
use). The actual port-opening path (start_midi_sources with a non-empty, real-port
source list) is not covered by automated tests this phase, and hasn't been verified
against real hardware.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ensemble.director import DirectorSignal, DirectorSource
from gesture.recognizer import SubGestureRecognizer
from gesture.vocabulary import Gesture, GestureRecognizer

from .midi_listener import MidiListener, cc_value_to_intensity

PERFORMER = "performer"
DIRECTOR = "director"
DEFAULT_CC_NUMBER = 1  # mod wheel — the most common continuous controller on a MIDI keyboard

__all__ = [
    "PERFORMER", "DIRECTOR", "DEFAULT_CC_NUMBER",
    "MidiSourceConfig", "MidiSources", "start_midi_sources", "cc_value_to_intensity",
]


@dataclass
class MidiSourceConfig:
    id: str
    role: str  # "performer" or "director" — DESTINATION, not capability (see module docstring)
    port: int
    cc_number: int = DEFAULT_CC_NUMBER  # tracked for every source; only director sources have a consumer for it today

    def __post_init__(self) -> None:
        if self.role not in (PERFORMER, DIRECTOR):
            raise ValueError(f"role must be {PERFORMER!r} or {DIRECTOR!r}, got {self.role!r}")


@dataclass
class MidiSources:
    performers: Dict[str, MidiListener] = field(default_factory=dict)
    directors: Dict[str, MidiListener] = field(default_factory=dict)
    director_sources: Dict[str, DirectorSource] = field(default_factory=dict)

    def stop_all(self) -> None:
        for listener in list(self.performers.values()) + list(self.directors.values()):
            listener.stop()


def _director_source(listener: MidiListener, pending: dict) -> DirectorSource:
    """Reads the *current* intensity each time it's called, not a value frozen
    at this call — a live fader genuinely drives Session.generate's per-bar
    aggregation as it moves (§11) — and pops the most recently latched gesture,
    if any, so it's delivered exactly once, not re-delivered every subsequent
    bar it happens to still be the last one seen. listener.intensity is never
    None here: start_midi_sources always constructs director listeners with a
    cc_number, so MidiListener.__init__ has already set a starting value."""

    def source(song, bar_index, timeline) -> DirectorSignal:
        gesture = pending.pop("gesture", None)
        return DirectorSignal(intensity=listener.intensity, gesture=gesture)

    return source


def start_midi_sources(
    sources: List[MidiSourceConfig],
    on_gesture: Optional[Callable[[str, Gesture], None]] = None,
    pitch_bend_range: float = 400,
) -> MidiSources:
    """One listener per source — the SAME kind for every role (recognition is
    uniform; see module docstring). on_gesture(source_id, gesture) is called for
    every recognised gesture from EVERY source now, performer or director (a
    real, deliberate change from before Phase 13, when only performers had a
    recognizer at all) — mirrors the AGRP concert setup's own modularity (one
    instrument, one Sonuus, one recogniser), just N of them in one process
    instead of N browser tabs.

    pitch_bend_range applies to every source alike — GestureRecognizer doesn't
    expose it directly, only via a pre-built SubGestureRecognizer passed as
    sub_recognizer, so that's constructed explicitly here rather than silently
    falling back to SubGestureRecognizer's own default."""
    result = MidiSources()
    for source in sources:
        pending: dict = {}

        def _latch_and_forward(gesture: Gesture, source_id=source.id, pending=pending) -> None:
            pending["gesture"] = gesture
            if on_gesture:
                on_gesture(source_id, gesture)

        recognizer = GestureRecognizer(
            on_gesture=_latch_and_forward,
            sub_recognizer=SubGestureRecognizer(pitch_bend_range=pitch_bend_range),
        )
        listener = MidiListener(recognizer, cc_number=source.cc_number)
        listener.start(source.port)

        if source.role == PERFORMER:
            result.performers[source.id] = listener
        else:  # DIRECTOR, validated by MidiSourceConfig.__post_init__
            result.directors[source.id] = listener
            result.director_sources[source.id] = _director_source(listener, pending)
    return result
