"""Tests for input/sources.py — logic only, no MIDI hardware needed.

(Manually verified separately against this machine's virtual IAC MIDI ports — real
rtmidi I/O, not just synthetic calls — that start_midi_sources genuinely opens a
port and that both a director CC message and a performer note-on reach their
listeners correctly. Not part of this automated suite since it depends on
macOS-specific virtual MIDI infrastructure that isn't guaranteed present elsewhere.)
"""

from ensemble.director import NEUTRAL_INTENSITY
from input.sources import (
    DIRECTOR,
    PERFORMER,
    DirectorMidiListener,
    MidiSourceConfig,
    MidiSources,
    cc_value_to_intensity,
    start_midi_sources,
)


def test_cc_value_to_intensity_boundaries():
    assert cc_value_to_intensity(0) == 0.0
    assert cc_value_to_intensity(127) == 1.0
    assert abs(cc_value_to_intensity(64) - 0.5) < 0.01


def test_midi_source_config_rejects_unknown_role():
    try:
        MidiSourceConfig(id="x", role="conductor", port=0)
        assert False, "expected a ValueError for an unknown role"
    except ValueError:
        pass

    MidiSourceConfig(id="ok1", role=PERFORMER, port=0)
    MidiSourceConfig(id="ok2", role=DIRECTOR, port=0)


def test_director_midi_listener_starts_neutral():
    listener = DirectorMidiListener(cc_number=1)
    assert listener.intensity == NEUTRAL_INTENSITY


def test_director_midi_listener_updates_intensity_from_matching_cc():
    listener = DirectorMidiListener(cc_number=1)
    listener._callback(([0xB0, 1, 127], 0.0), None)
    assert listener.intensity == 1.0
    listener._callback(([0xB0, 1, 0], 0.0), None)
    assert listener.intensity == 0.0


def test_director_midi_listener_ignores_other_cc_numbers():
    listener = DirectorMidiListener(cc_number=1)
    listener._callback(([0xB0, 2, 127], 0.0), None)  # a different CC
    assert listener.intensity == NEUTRAL_INTENSITY


def test_director_midi_listener_ignores_non_cc_messages():
    listener = DirectorMidiListener(cc_number=1)
    listener._callback(([0x90, 60, 100], 0.0), None)  # note on, not a Control Change
    assert listener.intensity == NEUTRAL_INTENSITY


def test_as_source_reflects_live_value_not_a_snapshot():
    listener = DirectorMidiListener(cc_number=1)
    source = listener.as_source()

    first = source(None, 0, None)
    assert first.intensity == NEUTRAL_INTENSITY

    listener.intensity = 0.9  # simulates a live CC update arriving
    second = source(None, 0, None)
    assert second.intensity == 0.9


def test_start_midi_sources_with_no_sources_touches_no_hardware():
    result = start_midi_sources([])
    assert result == MidiSources()
