"""Shared configuration for combo."""

from input.sources import DIRECTOR, PERFORMER, MidiSourceConfig

# One entry per MIDI source, tagged by role (DESIGN.md §6/§11). Since Phase 13,
# role determines DESTINATION, not capability: every source gets the same
# GestureRecognizer (note-level pitch tracking -> named gestures) and live-CC
# intensity tracking — a director sitting at a keyboard uses the same interface
# a performer does ("dual control car"). "performer" gestures reach
# Session.gesture_source (TransitionController); "director" gestures/intensity
# reach a DirectorSignal. See `python listen.py --list` for available port
# indices.
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

# MIDI output port for self_test.py's playback (DESIGN.md §4's playback stage,
# output/midi_output.py). Run `python self_test.py --list-out` to find the real
# index for your synth/DAW (e.g. macOS's IAC Driver into GarageBand).
MIDI_OUTPUT_PORT = 0
