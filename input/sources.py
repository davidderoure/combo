"""Starts MIDI input for every role-tagged source in config.MIDI_SOURCES — the
integration point between raw MIDI (rtmidi) and what the rest of the system
consumes: named gestures (gesture/vocabulary.py) for performers, a live
DirectorSignal (ensemble/director.py) for directors. See DESIGN.md §6.

Honest limitation: there's no MIDI hardware attached in this development
environment. cc_value_to_intensity, MidiSourceConfig's validation, and
DirectorMidiListener's callback logic are all tested directly (synthetic MIDI byte
tuples, no real port needed — the same technique gesture/recognizer.py's own tests
use). The actual port-opening path (start_midi_sources with a non-empty, real-port
source list) is not covered by automated tests this phase, and hasn't been verified
against real hardware.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import rtmidi

from ensemble.director import DirectorSignal, DirectorSource, NEUTRAL_INTENSITY
from gesture.recognizer import SubGestureRecognizer
from gesture.vocabulary import Gesture, GestureRecognizer

from .midi_listener import MidiListener

PERFORMER = "performer"
DIRECTOR = "director"
DEFAULT_CC_NUMBER = 1  # mod wheel — the most common continuous controller on a MIDI keyboard


@dataclass
class MidiSourceConfig:
    id: str
    role: str  # "performer" or "director"
    port: int
    cc_number: int = DEFAULT_CC_NUMBER  # only meaningful for role="director"

    def __post_init__(self) -> None:
        if self.role not in (PERFORMER, DIRECTOR):
            raise ValueError(f"role must be {PERFORMER!r} or {DIRECTOR!r}, got {self.role!r}")


def cc_value_to_intensity(value: int) -> float:
    """0-127 MIDI CC value -> 0.0-1.0 intensity, linear."""
    return max(0.0, min(1.0, value / 127.0))


class DirectorMidiListener:
    """A MIDI keyboard/controller's fader or mod wheel, read as a live intensity
    value — DESIGN.md §6's "director: read as discrete note/CC values mapped to the
    shared dial parameters," §11's dial channel. A fundamentally simpler use of MIDI
    than MidiListener's note-level pitch tracking: reading one control value, not
    extracting a melodic line — the same distinction §6 draws between the director's
    and the performer's input problems.
    """

    def __init__(self, cc_number: int = DEFAULT_CC_NUMBER):
        self.cc_number = cc_number
        self.intensity = NEUTRAL_INTENSITY
        self._midi_in = rtmidi.MidiIn()

    @staticmethod
    def list_ports() -> List[str]:
        return rtmidi.MidiIn().get_ports()

    def start(self, port_index: int) -> None:
        ports = self._midi_in.get_ports()
        if not ports:
            raise RuntimeError("No MIDI input ports found.")
        if port_index >= len(ports):
            raise RuntimeError(
                f"MIDI port {port_index} out of range ({len(ports)} available): {ports}"
            )
        print(f"MIDI director input: {ports[port_index]}")
        self._midi_in.open_port(port_index)
        self._midi_in.set_callback(self._callback)

    def stop(self) -> None:
        self._midi_in.close_port()

    def _callback(self, event, _data) -> None:
        message, _delta = event
        status = message[0] & 0xF0
        if status == 0xB0 and message[1] == self.cc_number:  # Control Change
            self.intensity = cc_value_to_intensity(message[2])

    def as_source(self) -> DirectorSource:
        """Returns a DirectorSource reading the *current* intensity each time it's
        called, not a value frozen at this call — so a live fader genuinely drives
        Session.generate's per-bar aggregation (§11) as it moves."""

        def source(song, bar_index, timeline) -> DirectorSignal:
            return DirectorSignal(intensity=self.intensity)

        return source


@dataclass
class MidiSources:
    performers: Dict[str, MidiListener] = field(default_factory=dict)
    directors: Dict[str, DirectorMidiListener] = field(default_factory=dict)

    def stop_all(self) -> None:
        for listener in list(self.performers.values()) + list(self.directors.values()):
            listener.stop()


def start_midi_sources(
    sources: List[MidiSourceConfig],
    on_gesture: Optional[Callable[[str, Gesture], None]] = None,
    pitch_bend_range: float = 400,
) -> MidiSources:
    """One listener per source, dispatched by role. on_gesture(source_id, gesture)
    is called for every recognised gesture, tagged with which source produced it —
    mirrors the AGRP concert setup's own modularity (one instrument, one Sonuus, one
    recogniser), just N of them in one process instead of N browser tabs.

    pitch_bend_range applies to every performer source alike — GestureRecognizer
    doesn't expose it directly, only via a pre-built SubGestureRecognizer passed as
    sub_recognizer, so that's constructed explicitly here rather than silently
    falling back to SubGestureRecognizer's own default."""
    result = MidiSources()
    for source in sources:
        if source.role == PERFORMER:
            recognizer = GestureRecognizer(
                on_gesture=(lambda g, sid=source.id: on_gesture(sid, g)) if on_gesture else None,
                sub_recognizer=SubGestureRecognizer(pitch_bend_range=pitch_bend_range),
            )
            listener = MidiListener(recognizer)
            listener.start(source.port)
            result.performers[source.id] = listener
        else:  # DIRECTOR, validated by MidiSourceConfig.__post_init__
            director_listener = DirectorMidiListener(cc_number=source.cc_number)
            director_listener.start(source.port)
            result.directors[source.id] = director_listener
    return result
