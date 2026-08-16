"""Tests for ensemble/drums.py — no MIDI/audio needed."""

from pathlib import Path

from ensemble.drums import (
    ACOUSTIC_SNARE,
    BUSY,
    CLOSED_HI_HAT,
    MEDIUM,
    RIDE_CYMBAL_1,
    SPARSE,
    TIMING_JITTER_BEATS,
    _density_for_bar,
    drum_generator,
)
from ensemble.director import DirectorSignal
from ensemble.timeline import BEATS_PER_BAR, Timeline
from song import parse_chart

CHARTS_DIR = Path(__file__).resolve().parent.parent / "songs"
ALL_PITCHES = {ACOUSTIC_SNARE, CLOSED_HI_HAT, RIDE_CYMBAL_1}


def load_blues():
    return parse_chart((CHARTS_DIR / "blues_in_f.chart").read_text())


# Head x1 = bars 0-11, Solos x3 = bars 12-47 (12 bars/chorus), Out x1 = bars 48-59.


def test_density_is_sparse_under_head_and_out():
    song = load_blues()
    assert _density_for_bar(song, 0) == SPARSE
    assert _density_for_bar(song, 11) == SPARSE
    assert _density_for_bar(song, 48) == SPARSE
    assert _density_for_bar(song, 59) == SPARSE


def test_density_is_medium_for_early_solo_choruses():
    song = load_blues()
    assert _density_for_bar(song, 12) == MEDIUM  # solos chorus 1
    assert _density_for_bar(song, 24) == MEDIUM  # solos chorus 2


def test_density_is_busy_for_the_last_solo_chorus():
    song = load_blues()
    assert _density_for_bar(song, 36) == BUSY  # solos chorus 3 (last)
    assert _density_for_bar(song, 47) == BUSY


def _assert_near_beat(event, bar_index, expected_offset):
    expected = bar_index * BEATS_PER_BAR + expected_offset
    assert abs(event.start_beat - expected) <= TIMING_JITTER_BEATS


def test_sparse_bar_is_hihat_on_two_and_four_only():
    events = drum_generator(seed=1)(load_blues(), 0, Timeline(), DirectorSignal())
    assert len(events) == 2
    assert all(e.pitch == CLOSED_HI_HAT for e in events)
    _assert_near_beat(events[0], 0, 1.0)
    _assert_near_beat(events[1], 0, 3.0)


def test_medium_bar_adds_a_walking_ride_pattern():
    events = drum_generator(seed=1)(load_blues(), 12, Timeline(), DirectorSignal())
    hihat = [e for e in events if e.pitch == CLOSED_HI_HAT]
    ride = [e for e in events if e.pitch == RIDE_CYMBAL_1]
    assert len(hihat) == 2
    assert len(ride) == 4
    assert len(events) == 6


def test_busy_bar_adds_syncopated_snare_accents():
    events = drum_generator(seed=1)(load_blues(), 36, Timeline(), DirectorSignal())
    snare = [e for e in events if e.pitch == ACOUSTIC_SNARE]
    assert 1 <= len(snare) <= 2
    assert len(events) == 6 + len(snare)
    # accents land off the beat (not near an integer beat offset within the bar)
    for e in snare:
        offset = (e.start_beat - 36 * BEATS_PER_BAR) % 1.0
        assert abs(offset - 0.5) < 0.1


def test_same_seed_is_reproducible():
    song = load_blues()
    a = drum_generator(seed=7)(song, 36, Timeline(), DirectorSignal())
    b = drum_generator(seed=7)(song, 36, Timeline(), DirectorSignal())
    assert a == b


def test_no_seed_runs_without_error():
    events = drum_generator()(load_blues(), 36, Timeline(), DirectorSignal())
    assert len(events) > 0


def test_all_pitches_are_known_gm_percussion_notes():
    song = load_blues()
    gen = drum_generator(seed=3)
    for bar_index in range(60):
        for event in gen(song, bar_index, Timeline(), DirectorSignal()):
            assert event.pitch in ALL_PITCHES
