"""Tests for ensemble/markov_sax.py -- the chain-walking sampler
(_generate_markov_phrase) is pure, tested with hand-built MarkovTables via
tmp_path, no real wjazzd.db needed. markov_sax_generator itself (the real-
consumer, selection-key proof) is tested against a REAL built cache below,
skipif-gated the same way tests/test_sax_wolfson_integration.py gates on
sax_best.pt -- but this needs no LSTM weights at all (the Markov generator
never touches PhraseGenerator), so it lives in its own file rather than
being needlessly bundled behind that file's weights-gate."""

import json
import random

import pytest

import self_test as st
from ensemble import MACHINE_SPEED, Session, Voice
from ensemble.critic import dissonance, motif_adherence, musicality_score
from ensemble.generators import chord_tone_generator
from ensemble.markov_sax import _generate_markov_phrase, markov_sax_generator
from ensemble.markov_tables import MarkovTables
from ensemble.sax import chord_to_wolfson_index
from ensemble.timeline import BEATS_PER_BAR
from ensemble.wolfson.encoding import dur_to_token, token_to_dur
from song import parse_chart
from song.chord import Chord

import markov_corpus

needs_markov_cache = pytest.mark.skipif(
    not markov_corpus.CACHE_PATH.exists(),
    reason=f"Markov tables cache not present at {markov_corpus.CACHE_PATH} — run 'python markov_corpus.py --build' first",
)


def _write_cache(path, pitch_transitions=None, duration_transitions=None,
                  pitch_marginals=None, duration_marginals=None):
    raw = {
        "order": 1,
        "pitch_transitions": pitch_transitions or {},
        "duration_transitions": duration_transitions or {},
        "pitch_marginals": pitch_marginals or {},
        "duration_marginals": duration_marginals or {},
    }
    path.write_text(json.dumps(raw))


def test_deterministic_chain_produces_a_predictable_phrase(tmp_path):
    """A quality-1 chain where every context has exactly one possible
    outcome: interval always +2, duration token always token_of(1.0). No
    seed -- starts at the middle of a (60, 72) register."""
    t1 = dur_to_token(1.0)
    path = tmp_path / "cache.json"
    _write_cache(
        path,
        pitch_transitions={"1": [[[2], {"2": 1}]]},
        pitch_marginals={"1": {"2": 1}},
        duration_transitions={"1": [[[t1], {str(t1): 1}]]},
        duration_marginals={"1": {str(t1): 1}},
    )
    tables = MarkovTables(path)
    rng = random.Random(0)

    notes = _generate_markov_phrase(
        tables, quality=1, seed_phrase=[], register=(60, 72), max_phrase_beats=4.0, order=1, rng=rng,
    )

    start_pitch = (60 + 72) // 2
    expected_pitches = [start_pitch + 2 * (i + 1) for i in range(len(notes))]
    assert [n["pitch"] for n in notes] == expected_pitches
    assert all(n["duration_beats"] == token_to_dur(t1) for n in notes)
    # every note is exactly 1 beat (token_to_dur(t1)) -- should stop once
    # total reaches or exceeds max_phrase_beats=4.0
    assert sum(n["duration_beats"] for n in notes) >= 4.0
    assert sum(n["duration_beats"] for n in notes) - notes[-1]["duration_beats"] < 4.0


def test_seed_phrase_provides_the_starting_pitch_and_context(tmp_path):
    """With a real seed, generation continues from the seed's own last
    pitch, and the first sampled interval is looked up using the seed's own
    trailing interval as context."""
    t1 = dur_to_token(1.0)
    path = tmp_path / "cache.json"
    _write_cache(
        path,
        # context (3,) [the seed's own trailing interval] -> always +5
        pitch_transitions={"1": [[[3], {"5": 1}]]},
        pitch_marginals={"1": {"5": 1}},
        duration_transitions={"1": [[[t1], {str(t1): 1}]]},
        duration_marginals={"1": {str(t1): 1}},
    )
    tables = MarkovTables(path)
    rng = random.Random(0)

    # seed: pitches 60 -> 63 (interval +3), matching duration 1.0 beat each
    seed_phrase = [
        {"pitch": 60, "onset": 0.0, "offset": 1.0},
        {"pitch": 63, "onset": 1.0, "offset": 2.0},
    ]
    notes = _generate_markov_phrase(
        tables, quality=1, seed_phrase=seed_phrase, register=(50, 80), max_phrase_beats=1.0, order=1, rng=rng,
    )
    assert notes[0]["pitch"] == 63 + 5  # continues from the seed's last pitch (63), using context (3,) -> +5


def test_max_phrase_beats_respected_with_varying_durations(tmp_path):
    t_short = dur_to_token(0.5)
    path = tmp_path / "cache.json"
    _write_cache(
        path,
        pitch_transitions={"1": [[[1], {"1": 1}]]},
        pitch_marginals={"1": {"1": 1}},
        duration_transitions={"1": [[[t_short], {str(t_short): 1}]]},
        duration_marginals={"1": {str(t_short): 1}},
    )
    tables = MarkovTables(path)
    rng = random.Random(0)
    notes = _generate_markov_phrase(
        tables, quality=1, seed_phrase=[], register=(60, 72), max_phrase_beats=2.0, order=1, rng=rng,
    )
    total = sum(n["duration_beats"] for n in notes)
    assert total >= 2.0
    assert total - notes[-1]["duration_beats"] < 2.0


@needs_markov_cache
def test_markov_sax_generator_real_selection_matches_the_recomputed_key():
    """Real-consumer proof, same spy-and-recompute discipline as every real
    consumer test in tests/test_sax_wolfson_integration.py: run a real
    Session over blues_in_f.chart with a real built MarkovTables cache,
    spy on _generate_markov_phrase to capture every candidate, independently
    recompute the (-dissonance, motif_adherence, overall) key sax_generator
    and markov_sax_generator both use, and confirm the dispensed notes match
    the recomputed winner -- not just that generation runs without error."""
    import ensemble.markov_sax as markov_sax_module

    original = markov_sax_module._generate_markov_phrase
    candidates = []  # (chord_quality, seed_phrase, notes) per call

    def recording_generate(tables, quality, seed_phrase, register, max_phrase_beats, order, rng):
        notes = original(tables, quality, seed_phrase, register, max_phrase_beats, order, rng)
        candidates.append((quality, seed_phrase, notes))
        return notes

    markov_sax_module._generate_markov_phrase = recording_generate
    try:
        tables = MarkovTables(markov_corpus.CACHE_PATH)
        song = parse_chart(st.DEFAULT_CHART.read_text())
        bass = Voice(id="bass", instrument="bass", register=st.BASS_REGISTER, source="ai",
                     generator=chord_tone_generator(st.BASS_REGISTER))
        sax_gen = markov_sax_generator(st.SAX_REGISTER, target_voice_id="bass", tables=tables, n_candidates=5, seed=7)
        sax = Voice(id="sax", instrument="sax", register=st.SAX_REGISTER, source="ai", generator=sax_gen)
        timeline = Session(song=song, voices=[bass, sax]).generate(mode=MACHINE_SPEED)
    finally:
        markov_sax_module._generate_markov_phrase = original

    assert len(candidates) > 0
    assert len(sax_gen.winning_score_log) > 0

    # Reconstruct chunk boundaries the same way the real loop does: group
    # consecutive candidates by (quality, seed_phrase identity) into chunks
    # of 5 (n_candidates), matching one chunk-build each.
    first_chunk = candidates[:5]
    quality = first_chunk[0][0]
    chord_idx = chord_to_wolfson_index(song.chord_at(0.0))
    assert quality == chord_idx % 4

    recomputed = [
        (
            -dissonance(notes, chord_idx),
            motif_adherence(notes, []),
            musicality_score(notes, chord_idx, seed_phrase, st.SAX_REGISTER).overall,
        )
        for _q, seed_phrase, notes in first_chunk
    ]
    best_key = max(recomputed)
    winner_notes = first_chunk[recomputed.index(best_key)][2]
    winner_pitches = sorted(n["pitch"] for n in winner_notes)

    dispensed_pitches = sorted(
        e.pitch for e in timeline if e.voice_id == "sax" and 0.0 <= e.start_beat < BEATS_PER_BAR
    )
    assert set(dispensed_pitches).issubset(set(winner_pitches))


@needs_markov_cache
def test_markov_sax_generator_own_pitch_range_matches_real_dispensed_range():
    tables = MarkovTables(markov_corpus.CACHE_PATH)
    song = parse_chart(st.DEFAULT_CHART.read_text())
    bass = Voice(id="bass", instrument="bass", register=st.BASS_REGISTER, source="ai",
                 generator=chord_tone_generator(st.BASS_REGISTER))
    sax_gen = markov_sax_generator(st.SAX_REGISTER, target_voice_id="bass", tables=tables, n_candidates=3, seed=11)
    sax = Voice(id="sax", instrument="sax", register=st.SAX_REGISTER, source="ai", generator=sax_gen)
    timeline = Session(song=song, voices=[bass, sax]).generate(mode=MACHINE_SPEED)

    all_sax_pitches = [e.pitch for e in timeline if e.voice_id == "sax"]
    assert all_sax_pitches
    assert sax_gen.own_pitch_range == {"low": min(all_sax_pitches), "high": max(all_sax_pitches)}
