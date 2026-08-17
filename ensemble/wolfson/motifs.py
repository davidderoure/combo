"""Interval-motif extraction — ported from wolfson's input/phrase_analyzer.py
(extract_interval_motifs only; that file's other feature-extraction functions
depend on phrase-detector/beat-estimator context combo doesn't have and weren't
ported). See ensemble/wolfson/__init__.py for provenance."""


def extract_interval_motifs(phrase: list[dict]) -> list:
    """
    Extract all 2-, 3-, and 4-note interval n-grams from a phrase.

    Returns a list of tuples of signed semitone intervals — transposition-invariant
    so the same melodic shape is recognised regardless of key.

    e.g. pitches [62, 65, 67, 69] → intervals [3, 2, 2]
         2-grams: [(3,2), (2,2)]
         3-grams: [(3,2,2)]
    """
    pitches = [n["pitch"] for n in phrase if "pitch" in n]
    if len(pitches) < 2:
        return []
    intervals = [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]
    motifs = []
    for length in (2, 3, 4):
        for start in range(len(intervals) - length + 1):
            motifs.append(tuple(intervals[start : start + length]))
    return motifs
