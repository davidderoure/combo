"""Stub generators — placeholder voice output to prove the ensemble pipeline works.

These are explicitly not meant to sound good. Real generation (an adapted Wolfson
model, DYCI2/Dicy2-python, or something new — DESIGN.md §12) replaces these once the
ensemble skeleton (voices, timeline, generation modes, accompaniment-listening,
director) is proven against them.
"""

from typing import List, Tuple

from .timeline import BEATS_PER_BAR, NoteEvent
from .voice import Generator

DEFAULT_VELOCITY = 80
DEFAULT_DURATION_BEATS = 1.0


def _place_in_register(pitch_class: int, register: Tuple[int, int]) -> int:
    """Lowest MIDI note >= register[0] with this pitch class."""
    low, high = register
    note = low + ((pitch_class - low) % 12)
    if note > high:
        note -= 12
    return note


def chord_tone_generator(register: Tuple[int, int]) -> Generator:
    """Build a generator that plays root + fifth on beats 1 and 3 of every bar,
    voiced within the given (low, high) MIDI register."""

    def generate(song, bar_index: int) -> List[NoteEvent]:
        events: List[NoteEvent] = []
        for beat_offset in (0.0, 2.0):
            beat = bar_index * BEATS_PER_BAR + beat_offset
            chord = song.chord_at(beat)
            root = _place_in_register(chord.root, register)
            fifth_class = (chord.root + 7) % 12
            fifth = _place_in_register(fifth_class, register)
            for pitch in (root, fifth):
                events.append(
                    NoteEvent(
                        voice_id="",  # stamped by Session once the voice is known
                        pitch=pitch,
                        velocity=DEFAULT_VELOCITY,
                        start_beat=beat,
                        duration_beats=DEFAULT_DURATION_BEATS,
                    )
                )
        return events

    return generate
