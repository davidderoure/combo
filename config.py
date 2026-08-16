"""Shared configuration for combo."""

from input.sources import DIRECTOR, PERFORMER, MidiSourceConfig

# One entry per MIDI source, tagged by role (DESIGN.md §6): "performer" sources
# feed a GestureRecognizer (note-level pitch tracking -> named gestures);
# "director" sources read a live Control Change value into the shared intensity
# dial (§11), not note-level input at all. See `python listen.py --list` for
# available port indices.
#
# Defaults to a single performer on port 0 — behaviourally equivalent to the old
# single-port setup. To add a director once a second MIDI device is available:
#   MidiSourceConfig(id="director", role=DIRECTOR, port=1, cc_number=1)
MIDI_SOURCES = [
    MidiSourceConfig(id="bass", role=PERFORMER, port=0),
]

# Sonuus i2M / similar pitch-to-MIDI trackers: pitchbend range in cents.
# 400 = +-200 cents (a whole tone each way), matching AGRP's default.
PITCH_BEND_RANGE = 400
