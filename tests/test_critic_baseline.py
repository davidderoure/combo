"""Tests for critic_baseline.py's rest-gap-synthesis helper (Phase 30) --
pure, hand-built note dicts, no real wjazzd.db needed. The rest of
critic_baseline.py (WJD-corpus/combo-take scoring, summarizing, printing) is
investigative-tool orchestration, not unit-tested, same precedent as
wjd_corpus.py's own --build/--benchmark (Phase 28)."""

from critic_baseline import _insert_rest_gaps
from ensemble.wolfson.phrase_generator import REST_PITCH


def _note(pitch, duration_beats, onset, beatdur=0.5):
    return {"pitch": pitch, "duration_beats": duration_beats, "onset": onset, "beatdur": beatdur}


def test_insert_rest_gaps_empty_input():
    assert _insert_rest_gaps([]) == []


def test_insert_rest_gaps_single_note_no_gap_possible():
    notes = [_note(60, 1.0, 0.0)]
    assert _insert_rest_gaps(notes) == [{"pitch": 60, "duration_beats": 1.0}]


def test_insert_rest_gaps_inserts_a_rest_for_a_real_gap():
    # note1: onset=0, dur=1 beat @ beatdur=0.5s -> offset=0.5s
    # note2: onset=1.0s -> gap = (1.0-0.5)/0.5 = 1.0 beat, >= MIN_BREATH_BEATS (0.5)
    notes = [_note(60, 1.0, 0.0, beatdur=0.5), _note(62, 1.0, 1.0, beatdur=0.5)]
    result = _insert_rest_gaps(notes)
    assert result == [
        {"pitch": 60, "duration_beats": 1.0},
        {"pitch": REST_PITCH, "duration_beats": 1.0},
        {"pitch": 62, "duration_beats": 1.0},
    ]


def test_insert_rest_gaps_no_rest_below_threshold():
    # offset = 0 + 1*0.5 = 0.5s; note2 onset=0.6s -> gap=(0.6-0.5)/0.5=0.2 beats < 0.5
    notes = [_note(60, 1.0, 0.0, beatdur=0.5), _note(62, 1.0, 0.6, beatdur=0.5)]
    result = _insert_rest_gaps(notes)
    assert result == [{"pitch": 60, "duration_beats": 1.0}, {"pitch": 62, "duration_beats": 1.0}]
    assert not any(n["pitch"] == REST_PITCH for n in result)


def test_insert_rest_gaps_overlapping_notes_insert_nothing():
    # note2 starts BEFORE note1's offset -- a negative gap, never a rest.
    notes = [_note(60, 1.0, 0.0, beatdur=0.5), _note(62, 1.0, 0.1, beatdur=0.5)]
    result = _insert_rest_gaps(notes)
    assert not any(n["pitch"] == REST_PITCH for n in result)


def test_insert_rest_gaps_exactly_at_threshold_inserts_a_rest():
    # gap = exactly 0.5 beats -- MIN_BREATH_BEATS' own >= boundary.
    notes = [_note(60, 1.0, 0.0, beatdur=0.5), _note(62, 1.0, 0.75, beatdur=0.5)]
    result = _insert_rest_gaps(notes)
    assert result[1] == {"pitch": REST_PITCH, "duration_beats": 0.5}


def test_insert_rest_gaps_multiple_notes_multiple_gaps():
    notes = [
        _note(60, 1.0, 0.0, beatdur=0.5),   # offset 0.5
        _note(62, 1.0, 0.6, beatdur=0.5),   # gap 0.2 -- no rest; offset 1.1
        _note(64, 1.0, 2.1, beatdur=0.5),   # gap (2.1-1.1)/0.5=2.0 -- rest
    ]
    result = _insert_rest_gaps(notes)
    pitches = [n["pitch"] for n in result]
    assert pitches == [60, 62, REST_PITCH, 64]
