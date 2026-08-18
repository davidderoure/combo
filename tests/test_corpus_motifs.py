"""Tests for ensemble/corpus_motifs.py (CorpusMotifs) -- pure, pytest's
tmp_path fixture with a small hand-built cache JSON (same shape wjd_corpus.py
--build produces), no real wjazzd.db needed."""

import json

import pytest

from ensemble.corpus_motifs import CorpusMotifs


def _write_cache(path, pitch_motifs=None, duration_motifs=None):
    raw = {
        "n_solos": 1,
        "n_notes": 4,
        "pitch_motifs": pitch_motifs or {},
        "duration_motifs": duration_motifs or {},
    }
    path.write_text(json.dumps(raw))


def test_has_pitch_motif_true_for_a_seen_motif(tmp_path):
    path = tmp_path / "cache.json"
    _write_cache(path, pitch_motifs={"1": [[[3, 2], 5]]})  # quality 1 (dominant): motif (3,2) seen 5x
    corpus = CorpusMotifs(path)
    assert corpus.has_pitch_motif((3, 2), chord_quality=1) is True


def test_has_pitch_motif_false_for_an_unseen_motif(tmp_path):
    path = tmp_path / "cache.json"
    _write_cache(path, pitch_motifs={"1": [[[3, 2], 5]]})
    corpus = CorpusMotifs(path)
    assert corpus.has_pitch_motif((9, 9), chord_quality=1) is False


def test_has_pitch_motif_false_for_an_unseen_quality(tmp_path):
    path = tmp_path / "cache.json"
    _write_cache(path, pitch_motifs={"1": [[[3, 2], 5]]})
    corpus = CorpusMotifs(path)
    # motif (3,2) exists under quality 1 but this asks about quality 2 (minor) -- no crash, just False
    assert corpus.has_pitch_motif((3, 2), chord_quality=2) is False


def test_has_duration_motif_true_for_a_seen_motif(tmp_path):
    path = tmp_path / "cache.json"
    _write_cache(path, duration_motifs={"0": [[[10, 10], 3]]})
    corpus = CorpusMotifs(path)
    assert corpus.has_duration_motif((10, 10), chord_quality=0) is True


def test_has_duration_motif_false_for_an_unseen_motif(tmp_path):
    path = tmp_path / "cache.json"
    _write_cache(path, duration_motifs={"0": [[[10, 10], 3]]})
    corpus = CorpusMotifs(path)
    assert corpus.has_duration_motif((1, 1), chord_quality=0) is False


def test_missing_cache_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        CorpusMotifs(tmp_path / "does_not_exist.json")


def test_loads_multiple_qualities_independently(tmp_path):
    path = tmp_path / "cache.json"
    _write_cache(
        path,
        pitch_motifs={"0": [[[1, 1], 1]], "1": [[[2, 2], 1]], "2": [[[3, 3], 1]], "3": [[[4, 4], 1]]},
    )
    corpus = CorpusMotifs(path)
    assert corpus.has_pitch_motif((1, 1), chord_quality=0) is True
    assert corpus.has_pitch_motif((2, 2), chord_quality=1) is True
    assert corpus.has_pitch_motif((3, 3), chord_quality=2) is True
    assert corpus.has_pitch_motif((4, 4), chord_quality=3) is True
    # motifs don't leak across quality
    assert corpus.has_pitch_motif((1, 1), chord_quality=1) is False
