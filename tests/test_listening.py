"""Tests for ensemble/listening.py — no MIDI/audio needed."""

from ensemble.listening import (
    average_velocity,
    beats_of_silence,
    density,
    pitch_range,
    synthetic_varying_density_generator,
)
from ensemble.timeline import NoteEvent, Timeline


def make_timeline() -> Timeline:
    return Timeline(
        [
            NoteEvent("sax", 60, 80, 0.0, 1.0),
            NoteEvent("sax", 64, 90, 1.0, 1.0),
            NoteEvent("sax", 67, 100, 2.0, 1.0),
            NoteEvent("drums", 42, 70, 0.5, 0.1),
            NoteEvent("drums", 42, 75, 2.5, 0.1),
        ]
    )


def test_density_counts_only_the_given_voice():
    tl = make_timeline()
    assert density(tl, "sax", 0.0, 4.0) == 3 / 4.0
    assert density(tl, "drums", 0.0, 4.0) == 2 / 4.0


def test_density_is_zero_for_empty_window_or_unheard_voice():
    tl = make_timeline()
    assert density(tl, "sax", 10.0, 10.0) == 0.0  # zero-width window
    assert density(tl, "keys", 0.0, 4.0) == 0.0  # voice never played


def test_density_window_is_half_open():
    tl = make_timeline()
    # the sax note at beat 2.0 is excluded when until_beat=2.0
    assert density(tl, "sax", 0.0, 2.0) == 2 / 2.0


def test_pitch_range():
    tl = make_timeline()
    assert pitch_range(tl, "sax", 0.0, 4.0) == (60, 67)
    assert pitch_range(tl, "keys", 0.0, 4.0) is None


def test_average_velocity():
    tl = make_timeline()
    assert average_velocity(tl, "sax", 0.0, 4.0) == (80 + 90 + 100) / 3
    assert average_velocity(tl, "keys", 0.0, 4.0) is None


def test_beats_of_silence():
    tl = make_timeline()
    assert beats_of_silence(tl, "sax", 5.0) == 5.0 - 2.0  # last sax note at beat 2.0
    assert beats_of_silence(tl, "keys", 5.0) == float("inf")


def test_synthetic_varying_density_generator_cycles_busy_sparse_moderate():
    gen = synthetic_varying_density_generator(seed=1)
    empty = Timeline()
    assert len(gen(None, 0, empty)) == 8  # busy
    assert len(gen(None, 1, empty)) == 0  # sparse
    assert len(gen(None, 2, empty)) == 2  # moderate
    assert len(gen(None, 3, empty)) == 8  # cycle repeats
