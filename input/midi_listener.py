"""Real-time MIDI input feeding a SubGestureRecognizer, and (optionally) a live
Control Change value.

One MidiListener wraps one MIDI input port/channel and owns one
SubGestureRecognizer, mirroring how the AGRP concert setup worked (one
instrument, one Sonuus, one recogniser) — just as a class instead of a browser
tab. Style follows wolfson's input/midi_listener.py.

Since Phase 13 (DESIGN.md §6/§11), this is the ONE listener type for every MIDI
source regardless of role — a director sitting at a keyboard uses the same
recognition machinery as a performer ("dual control car": same controls, a
second seat that can also act). What differs by role is where the recognised
output is ROUTED (input/sources.py's concern), not what gets recognised here.
cc_number is optional and orthogonal to gesture recognition — a single physical
controller can send Control Change and note messages at the same time, so this
class tracks both on one open port/connection rather than needing two.

Deliberately ensemble-agnostic: this module knows nothing about DirectorSignal
or NEUTRAL_INTENSITY (ensemble/director.py) — intensity starts at the literal
0.5 (the same value, written locally rather than imported, so this module has
no dependency on the ensemble/ package). input/sources.py, which already
imports ensemble.director, is where that value's *meaning* ("no director
present") belongs.
"""

import time
from typing import Optional

import rtmidi

from gesture.recognizer import SubGestureRecognizer

DEFAULT_INTENSITY = 0.5  # same value as ensemble.director.NEUTRAL_INTENSITY -- see module docstring


def cc_value_to_intensity(value: int) -> float:
    """0-127 MIDI CC value -> 0.0-1.0 intensity, linear."""
    return max(0.0, min(1.0, value / 127.0))


class MidiListener:
    """Listens on a MIDI input port and drives a SubGestureRecognizer, and
    optionally tracks a live Control Change value if cc_number is given."""

    def __init__(self, recognizer: SubGestureRecognizer, cc_number: Optional[int] = None):
        self.recognizer = recognizer
        self.cc_number = cc_number
        self.intensity: Optional[float] = DEFAULT_INTENSITY if cc_number is not None else None
        self._midi_in = rtmidi.MidiIn()

    @staticmethod
    def list_ports() -> list[str]:
        return rtmidi.MidiIn().get_ports()

    def start(self, port_index: int) -> None:
        ports = self._midi_in.get_ports()
        if not ports:
            raise RuntimeError("No MIDI input ports found.")
        if port_index >= len(ports):
            raise RuntimeError(
                f"MIDI port {port_index} out of range ({len(ports)} available): {ports}"
            )
        print(f"MIDI input: {ports[port_index]}")
        self._midi_in.open_port(port_index)
        self._midi_in.set_callback(self._callback)

    def stop(self) -> None:
        self._midi_in.close_port()

    def _callback(self, event, _data) -> None:
        message, _delta = event
        status = message[0] & 0xF0
        now = time.time()

        if status == 0x90 and message[2] != 0:  # note on
            self.recognizer.midi_note_on(message[1], message[2], now)
        elif status == 0x80 or (status == 0x90 and message[2] == 0):  # note off
            self.recognizer.midi_note_off(message[1], now)
        elif status == 0xE0:  # pitch bend, lsb, msb
            self.recognizer.midi_pitch_bend(message[1], message[2], now)
        elif status == 0xB0 and self.cc_number is not None and message[1] == self.cc_number:  # Control Change
            self.intensity = cc_value_to_intensity(message[2])
