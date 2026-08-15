"""Shared configuration for combo."""

# MIDI input port index (see `python listen.py --list`)
MIDI_INPUT_PORT = 0

# Sonuus i2M / similar pitch-to-MIDI trackers: pitchbend range in cents.
# 400 = +-200 cents (a whole tone each way), matching AGRP's default.
PITCH_BEND_RANGE = 400
