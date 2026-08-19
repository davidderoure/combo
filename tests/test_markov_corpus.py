"""Tests for markov_corpus.py's build_tables -- pure, hand-built note lists
(matching wjd_corpus.iter_solos'/split_into_quality_runs' output shape), no
real wjazzd.db needed."""

from ensemble.wolfson.chords import QUAL_DOM
from ensemble.wolfson.encoding import dur_to_token
from markov_corpus import build_tables


def _note(pitch, duration_beats, chord_quality):
    return {"pitch": pitch, "duration_beats": duration_beats, "chord_quality": chord_quality}


def test_build_tables_order_1_counts_exact_transitions(monkeypatch):
    """pitches [60, 62, 64, 60, 62] -> intervals [2, 2, -4, 2]. Order-1
    context (2,) is followed by 2 (once) and by -4 (once); context (-4,) is
    followed by 2 (once)."""
    solo = [_note(p, 1.0, QUAL_DOM) for p in [60, 62, 64, 60, 62]]

    def fake_iter_solos(db_path):
        yield solo

    def fake_split(s):
        return [(QUAL_DOM, s)]

    monkeypatch.setattr("markov_corpus.wjd_corpus.iter_solos", fake_iter_solos)
    monkeypatch.setattr("markov_corpus.wjd_corpus.split_into_quality_runs", fake_split)

    pitch_transitions, duration_transitions, pitch_marginals, duration_marginals = build_tables(None, order=1)

    table = pitch_transitions[QUAL_DOM]
    assert table[(2,)][2] == 1
    assert table[(2,)][-4] == 1
    assert table[(-4,)][2] == 1
    assert (2,) not in pitch_transitions[QUAL_DOM] or sum(table[(2,)].values()) == 2

    # marginal counts every interval seen, regardless of context
    assert pitch_marginals[QUAL_DOM][2] == 3  # 2 appears 3 times in [2,2,-4,2]
    assert pitch_marginals[QUAL_DOM][-4] == 1


def test_build_tables_duration_tokens_use_dur_to_token(monkeypatch):
    solo = [_note(60, 1.0, QUAL_DOM), _note(62, 0.5, QUAL_DOM), _note(64, 1.0, QUAL_DOM)]

    def fake_iter_solos(db_path):
        yield solo

    def fake_split(s):
        return [(QUAL_DOM, s)]

    monkeypatch.setattr("markov_corpus.wjd_corpus.iter_solos", fake_iter_solos)
    monkeypatch.setattr("markov_corpus.wjd_corpus.split_into_quality_runs", fake_split)

    _pitch_t, duration_transitions, _pitch_m, duration_marginals = build_tables(None, order=1)

    t1, t05 = dur_to_token(1.0), dur_to_token(0.5)
    table = duration_transitions[QUAL_DOM]
    assert table[(t1,)][t05] == 1
    assert table[(t05,)][t1] == 1
    assert duration_marginals[QUAL_DOM][t1] == 2
    assert duration_marginals[QUAL_DOM][t05] == 1


def test_build_tables_separate_qualities_do_not_mix(monkeypatch):
    solo = [_note(60, 1.0, None), _note(62, 1.0, None)]  # chord_quality field unused by fake_split here

    def fake_iter_solos(db_path):
        yield solo

    def fake_split(s):
        from ensemble.wolfson.chords import QUAL_MAJOR
        # two runs, different qualities, each a single interval
        return [(QUAL_DOM, [_note(60, 1.0, QUAL_DOM), _note(63, 1.0, QUAL_DOM)]),
                (QUAL_MAJOR, [_note(60, 1.0, QUAL_MAJOR), _note(67, 1.0, QUAL_MAJOR)])]

    monkeypatch.setattr("markov_corpus.wjd_corpus.iter_solos", fake_iter_solos)
    monkeypatch.setattr("markov_corpus.wjd_corpus.split_into_quality_runs", fake_split)

    from ensemble.wolfson.chords import QUAL_MAJOR
    _pt, _dt, pitch_marginals, _dm = build_tables(None, order=1)

    assert pitch_marginals[QUAL_DOM][3] == 1  # 63-60
    assert pitch_marginals[QUAL_MAJOR][7] == 1  # 67-60
    assert pitch_marginals[QUAL_DOM][7] == 0  # no leakage across qualities
