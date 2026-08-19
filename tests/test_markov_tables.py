"""Tests for ensemble/markov_tables.py (MarkovTables) -- pure, pytest's
tmp_path fixture with a small hand-built cache JSON (same shape
markov_corpus.py --build produces), no real wjazzd.db needed."""

import json
import random

import pytest

from ensemble.markov_tables import MarkovTables


def _write_cache(path, order=1, pitch_transitions=None, duration_transitions=None,
                  pitch_marginals=None, duration_marginals=None):
    raw = {
        "order": order,
        "pitch_transitions": pitch_transitions or {},
        "duration_transitions": duration_transitions or {},
        "pitch_marginals": pitch_marginals or {},
        "duration_marginals": duration_marginals or {},
    }
    path.write_text(json.dumps(raw))


def test_loads_order(tmp_path):
    path = tmp_path / "cache.json"
    _write_cache(path, order=2)
    tables = MarkovTables(path)
    assert tables.order == 2


def test_sample_next_interval_single_outcome_context_is_deterministic(tmp_path):
    path = tmp_path / "cache.json"
    # quality 1 (dominant): context (3,) always leads to outcome 2
    _write_cache(path, pitch_transitions={"1": [[[3], {"2": 10}]]})
    tables = MarkovTables(path)
    rng = random.Random(0)
    for _ in range(20):
        assert tables.sample_next_interval(1, (3,), rng) == 2


def test_sample_next_interval_respects_weighted_proportions(tmp_path):
    path = tmp_path / "cache.json"
    # context (3,) -> outcome 2 nine times as often as outcome -5
    _write_cache(path, pitch_transitions={"1": [[[3], {"2": 900, "-5": 100}]]})
    tables = MarkovTables(path)
    rng = random.Random(42)
    draws = [tables.sample_next_interval(1, (3,), rng) for _ in range(2000)]
    fraction_2 = sum(1 for d in draws if d == 2) / len(draws)
    assert fraction_2 == pytest.approx(0.9, abs=0.03)


def test_sample_next_interval_unseen_context_falls_back_to_marginal(tmp_path):
    path = tmp_path / "cache.json"
    _write_cache(
        path,
        pitch_transitions={"1": [[[3], {"2": 10}]]},
        pitch_marginals={"1": {"7": 5}},
    )
    tables = MarkovTables(path)
    rng = random.Random(0)
    # context (99,) was never observed for quality 1 -- must fall back to the
    # marginal distribution ({7: 5}), not the (3,)-context's outcome (2).
    assert tables.sample_next_interval(1, (99,), rng) == 7


def test_sample_next_interval_none_context_uses_marginal_directly(tmp_path):
    path = tmp_path / "cache.json"
    _write_cache(
        path,
        pitch_transitions={"1": [[[3], {"2": 10}]]},
        pitch_marginals={"1": {"7": 5}},
    )
    tables = MarkovTables(path)
    rng = random.Random(0)
    assert tables.sample_next_interval(1, None, rng) == 7


def test_sample_next_interval_no_data_at_all_raises(tmp_path):
    path = tmp_path / "cache.json"
    _write_cache(path)
    tables = MarkovTables(path)
    rng = random.Random(0)
    with pytest.raises(ValueError):
        tables.sample_next_interval(1, (3,), rng)


def test_sample_next_duration_token_independent_of_pitch_tables(tmp_path):
    path = tmp_path / "cache.json"
    _write_cache(
        path,
        pitch_transitions={"1": [[[3], {"2": 10}]]},
        duration_transitions={"1": [[[10], {"12": 10}]]},
    )
    tables = MarkovTables(path)
    rng = random.Random(0)
    assert tables.sample_next_duration_token(1, (10,), rng) == 12
