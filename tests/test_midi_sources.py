"""Tests for input/sources.py and input/midi_listener.py's cc_number extension —
logic only, no MIDI hardware needed.

Since Phase 13: role determines destination, not capability — MidiListener is
the ONE listener type for every source, performer or director (see both
modules' docstrings). DirectorMidiListener is retired; its old tests are
replaced, not just patched, since the behaviour genuinely changed (a
cc_number-configured listener now ALSO recognises gestures, which used to be
impossible for a director-role source).

(Manually verified separately against this machine's virtual IAC MIDI ports — real
rtmidi I/O, not just synthetic calls — that start_midi_sources genuinely opens a
port and that both a director CC message and a performer note-on reach their
listeners correctly. Not part of this automated suite since it depends on
macOS-specific virtual MIDI infrastructure that isn't guaranteed present elsewhere.)
"""

from ensemble.director import DirectorSignal
from gesture.recognizer import SubGesture, SubGestureRecognizer
from gesture.vocabulary import Gesture, GestureRecognizer
from input.midi_listener import DEFAULT_INTENSITY, MidiListener, cc_value_to_intensity
from input.sources import (
    DIRECTOR,
    PERFORMER,
    MidiSourceConfig,
    MidiSources,
    _director_source,
    start_midi_sources,
)


def make_listener(cc_number=None, on_gesture=None):
    recognizer = GestureRecognizer(on_gesture=on_gesture, sub_recognizer=SubGestureRecognizer())
    return MidiListener(recognizer, cc_number=cc_number)


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


def test_midi_listener_without_cc_number_has_no_intensity():
    listener = make_listener(cc_number=None)
    assert listener.intensity is None


def test_midi_listener_with_cc_number_starts_neutral():
    listener = make_listener(cc_number=1)
    assert listener.intensity == DEFAULT_INTENSITY == 0.5


def test_midi_listener_updates_intensity_from_matching_cc():
    listener = make_listener(cc_number=1)
    listener._callback(([0xB0, 1, 127], 0.0), None)
    assert listener.intensity == 1.0
    listener._callback(([0xB0, 1, 0], 0.0), None)
    assert listener.intensity == 0.0


def test_midi_listener_ignores_other_cc_numbers():
    listener = make_listener(cc_number=1)
    listener._callback(([0xB0, 2, 127], 0.0), None)  # a different CC
    assert listener.intensity == DEFAULT_INTENSITY


def test_midi_listener_with_cc_number_still_recognises_gestures():
    """The direct, deliberate reversal of the old
    test_director_midi_listener_ignores_non_cc_messages: a cc_number-configured
    (i.e. director-role) listener now ALSO recognises gestures, same as a
    performer's — role determines destination, not capability. Drives the
    recognizer directly via feed_subgesture (same technique
    tests/test_gesture_vocabulary.py uses) rather than trying to reproduce real
    note-timing through raw MIDI byte tuples, to isolate "is the wiring intact"
    from "does real-time gesture recognition work" (already covered elsewhere)."""
    fired = []
    listener = make_listener(cc_number=1, on_gesture=fired.append)
    listener.recognizer.feed_subgesture(SubGesture(label="L", start_time=0.0, duration=1.0, n=1))
    assert [g.name for g in fired] == ["handover"]


def test_director_source_carries_both_intensity_and_a_gesture():
    listener = make_listener(cc_number=1)
    listener.intensity = 0.8
    pending = {"gesture": Gesture("toggle_singability")}

    source = _director_source(listener, pending)
    first = source(None, 0, None)
    assert first == DirectorSignal(intensity=0.8, gesture=Gesture("toggle_singability"))

    # Consumed once -- a second poll without a new gesture returns gesture=None.
    second = source(None, 0, None)
    assert second == DirectorSignal(intensity=0.8, gesture=None)


def test_start_midi_sources_with_no_sources_touches_no_hardware():
    result = start_midi_sources([])
    assert result == MidiSources()
