"""Duration-motif extraction — combo-authored, not a wolfson port (see module
docstring below for why it doesn't live under ensemble/wolfson/). Sibling to
ensemble/wolfson/motifs.py's extract_interval_motifs, built for Phase 28's WJD
corpus-feasibility work (DESIGN.md §13) and reusable wherever a rhythmic-motif
signal is needed later, the same way extract_interval_motifs already is by
ensemble/memory.py.
"""

from .wolfson.encoding import dur_to_token


def extract_duration_motifs(phrase: list) -> list:
    """All 2-, 3-, and 4-note duration n-grams — sequences of quantized duration
    TOKENS (dur_to_token, ensemble/wolfson/encoding.py — the same quantization
    the model's own output is already expressed in), not differences between
    consecutive durations.

    Unlike pitch, where a transposition-invariant INTERVAL is the natural unit
    of melodic "shape", a rhythmic figure's identity is usually the actual
    sequence of note values (e.g. "quarter, eighth, eighth"), not a relative
    delta between them — so this deliberately n-grams the token SEQUENCE
    directly (one token per note, same length as the phrase) rather than a
    sequence of differences (one shorter than the phrase, extract_interval_
    motifs' own shape). Structurally different from extract_interval_motifs
    for this reason, not an inconsistency.

    Whether a tempo-invariant/relative variant is ALSO needed (e.g. "same
    rhythmic shape, played faster") is a real, open question for whenever this
    feeds an actual critic or RehearsalMemory-style recall — deliberately not
    resolved in this feasibility phase.

    Like extract_interval_motifs, this does NOT filter wolfson's REST_PITCH
    sentinel — callers filter real notes first (see ensemble/memory.py's own
    store(), which does exactly this before calling extract_interval_motifs).
    Notes without a "duration_beats" key are skipped, matching extract_interval_
    motifs' own "pitch" in n convention.
    """
    tokens = [dur_to_token(n["duration_beats"]) for n in phrase if "duration_beats" in n]
    if len(tokens) < 2:
        return []
    motifs = []
    for length in (2, 3, 4):
        for start in range(len(tokens) - length + 1):
            motifs.append(tuple(tokens[start : start + length]))
    return motifs
