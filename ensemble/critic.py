"""A musicality critic for completed sax phrases — DESIGN.md §11/§12, Phase 12.

Named as the natural next step because it unblocks two things already decided
rather than opening a third front: ensemble/memory.py's own docstring names "no
evaluation of what's worth remembering" as a deliberate gap needing "an actual
critic... real, separate work"; the "chess" search-and-evaluate idea (DESIGN.md
§13) needs something to score candidates against too. This phase builds the critic
and wires it into the gap that's already open (RehearsalMemory) — search/evaluate
stays a separate, still-deferred direction, not attempted here.

Grounded in two real sources, not first-principles guessing:

1. David's unrelated prior work measuring "musicality" for LSTM-generated piano
   sight-reading pieces (a Colab notebook, Copy_of_G1.ipynb). Its actual method is
   corpus-similarity, not a hand-authored score: build a "truth" set from real
   Grade 1 MIDI specimens, extract features (interval histograms, and melodic
   contour reduced to a string of U/D/S -- up/down/same, via a per-pair cs(a,b)
   function joined into a string), then compare a candidate by EDIT DISTANCE --
   both on raw pitch sequences and on the U/D/S contour strings. A striking,
   unplanned overlap: gesture/recognizer.py's sub-gesture alphabet (R, X, U, D, S,
   T, L) already contains this exact U/D/S vocabulary, independently arrived at
   for a different purpose (live sub-gesture segmentation, not whole-phrase shape
   comparison) -- not shared code, shared vocabulary. The corpus-similarity METHOD
   doesn't transfer directly here -- combo has no "truth set" of real jazz phrases
   to compare against (WJazzD is baked into Wolfson's trained weights, not
   available as raw MIDI), and Grade 1 piano's interval/contour norms are the
   wrong reference for idiomatic jazz -- but the TECHNIQUE (interval sequences;
   U/D/S contour strings compared by edit distance) is reused directly below.

2. Wolfson's own ported bias layers (ensemble/wolfson/phrase_generator.py,
   ensemble/wolfson/scales.py) already encode real, previously-tuned musical
   judgments as GENERATION-TIME SAMPLING BIASES. Three are repurposed here as
   RETROSPECTIVE SCORING FUNCTIONS over completed output instead -- reusing
   already-vetted musical reasoning rather than inventing new theory.

Every function below is a pure, deterministic function over already-generated
data -- no model inference anywhere in this file, unlike everything else that
touches PhraseGenerator. All weights/thresholds (DEFAULT_WEIGHTS,
SMOOTH_INTERVAL_MAX_SEMITONES, NEAR_REPEAT_MAX_DISTANCE) are explicit
placeholders, same honest status as INTENSITY_SPREAD (comping.py),
REFERENCE_MAX_DENSITY (director.py), and DEFAULT_MOTIF_STRENGTH (sax.py) --
needing real tuning once there's a way to listen and compare, not asserted as
musically validated.
"""

import math
from collections import Counter
from dataclasses import dataclass

from .wolfson.encoding import dur_to_token
from .wolfson.motifs import extract_interval_motifs
from .wolfson.phrase_generator import REST_PITCH, SINGABLE_DUR_CENTER, SINGABLE_DUR_WIDTH
from .wolfson.scales import chord_root, chord_to_mode, chord_tones, scale_pitch_classes

TONAL_RESOLUTION_WEIGHT = 0.3  # blend weight for "does the last note land on a chord tone"
SMOOTH_INTERVAL_MAX_SEMITONES = 4  # placeholder -- Grade 1 piano's own norms are far more
                                    # conservative than idiomatic jazz and aren't a valid
                                    # reference; needs real jazz-appropriate tuning.
NEAR_REPEAT_WINDOW = 4         # contour-string window length for near-repeat comparison
NEAR_REPEAT_MAX_DISTANCE = 1   # placeholder: windows within this edit distance count as near-repeats

DEFAULT_WEIGHTS = {
    "tonal_conformity": 0.25,
    "contour_smoothness": 0.2,
    "repetition": 0.2,
    "call_response_relatedness": 0.15,
    "singability": 0.2,
}


def _real_notes(notes: list) -> list:
    return [n for n in notes if n.get("pitch") != REST_PITCH]


def _cs(a: int, b: int) -> str:
    """One pairwise contour symbol -- up/down/same. Same U/D/S alphabet as
    gesture/recognizer.py's sub-gestures, independently arrived at here for
    whole-phrase melodic shape rather than live sub-gesture segmentation --
    see module docstring."""
    if a > b:
        return "D"
    if b > a:
        return "U"
    return "S"


def _contour_string(pitches: list) -> str:
    return "".join(_cs(pitches[i], pitches[i + 1]) for i in range(len(pitches) - 1))


def _levenshtein(a: str, b: str) -> int:
    """Standard edit distance, in-house rather than a new dependency -- the
    notebook used the `editdistance` package for this; not worth adding a pip
    dependency to this codebase for one ~15-line algorithm."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def tonal_conformity(notes: list, chord_idx: int) -> float:
    """Fraction of real notes whose pitch class is in the active chord's scale
    (chord_to_mode/scale_pitch_classes, ensemble/wolfson/scales.py -- ported
    Phase 8, previously unused beyond chord_tones()), blended with a bonus for
    the phrase's FINAL note landing on an actual chord tone -- mirrors how
    Wolfson's own voice-leading bias specifically ramps up at the end of a
    phrase, not uniformly across it (phrase_generator.py's arc_position-scaled
    chord-tone targeting). No real notes -> 0.0: nothing to be in- or
    out-of-key about."""
    real = _real_notes(notes)
    if not real:
        return 0.0
    scale = scale_pitch_classes(chord_root(chord_idx), chord_to_mode(chord_idx))
    scale_fraction = sum(1 for n in real if n["pitch"] % 12 in scale) / len(real)

    tones = chord_tones(chord_idx)
    resolves = 1.0 if (tones and real[-1]["pitch"] % 12 in tones) else 0.0

    return (1.0 - TONAL_RESOLUTION_WEIGHT) * scale_fraction + TONAL_RESOLUTION_WEIGHT * resolves


def contour_smoothness(notes: list) -> float:
    """Fraction of consecutive-note intervals within SMOOTH_INTERVAL_MAX_SEMITONES
    -- same core computation as the notebook's intervals() (signed semitone
    differences between consecutive notes). Fewer than 2 real notes -> 1.0:
    vacuously smooth, nothing observed to be a leap."""
    pitches = [n["pitch"] for n in _real_notes(notes)]
    if len(pitches) < 2:
        return 1.0
    intervals = [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]
    smooth = sum(1 for iv in intervals if abs(iv) <= SMOOTH_INTERVAL_MAX_SEMITONES)
    return smooth / len(intervals)


def repetition(notes: list) -> float:
    """1.0 if the phrase shows evidence of a repeated pattern, else 0.0 --
    combines exact n-gram repetition (extract_interval_motifs, ported Phase 11:
    does any motif recur 2+ times) with NEAR-repetition via edit distance on the
    phrase's own U/D/S contour string (new here, inspired directly by the
    notebook's cs()/contour()/editdistance approach) -- catches a varied
    restatement of a lick that exact tuple-matching misses. Fewer than 3 real
    notes -> 0.0: too short for anything to recur."""
    real = _real_notes(notes)
    pitches = [n["pitch"] for n in real]
    if len(pitches) < 3:
        return 0.0

    motifs = extract_interval_motifs(real)
    exact = 1.0 if any(count >= 2 for count in Counter(motifs).values()) else 0.0

    contour = _contour_string(pitches)
    near = 0.0
    if len(contour) >= NEAR_REPEAT_WINDOW * 2:
        windows = [contour[i : i + NEAR_REPEAT_WINDOW] for i in range(len(contour) - NEAR_REPEAT_WINDOW + 1)]
        for i in range(len(windows)):
            for j in range(i + NEAR_REPEAT_WINDOW, len(windows)):
                if _levenshtein(windows[i], windows[j]) <= NEAR_REPEAT_MAX_DISTANCE:
                    near = 1.0
                    break
            if near:
                break

    return max(exact, near)


def call_response_relatedness(seed_phrase: list, response_notes: list) -> float:
    """U/D/S contour-string similarity between what the target voice played
    (the seed) and the phrase generated in response -- did the response relate
    to what it heard?

    Measures RELATEDNESS, not goodness: real call-and-response sometimes
    deliberately CONTRASTS rather than mirrors (Wolfson's own
    register_contrast_str bias values contrast over similarity elsewhere in
    this same ported code) -- a deliberately contrasting response can
    legitimately score low here, which isn't the same as a bad response. Said
    plainly rather than overclaimed."""
    seed_pitches = [n["pitch"] for n in seed_phrase]
    response_pitches = [n["pitch"] for n in _real_notes(response_notes)]
    seed_contour = _contour_string(seed_pitches)
    response_contour = _contour_string(response_pitches)

    if not seed_contour and not response_contour:
        return 1.0  # both trivially shapeless -- vacuously identical
    if not seed_contour or not response_contour:
        return 0.0  # one has shape, the other doesn't -- can't be related

    distance = _levenshtein(seed_contour, response_contour)
    max_len = max(len(seed_contour), len(response_contour))
    return 1.0 - (distance / max_len)


def singability(notes: list) -> float:
    """Retrospective version of Wolfson's own SINGABLE_DUR_CENTER/
    SINGABLE_DUR_WIDTH bell curve (phrase_generator.py), imported directly (not
    redefined) and applied to actual durations via dur_to_token
    (ensemble/wolfson/encoding.py) instead of as a sampling bias -- the exact
    formula the model already uses to nudge toward "sustained, singable"
    duration, now used to measure it instead. No real notes -> 0.0."""
    real = _real_notes(notes)
    if not real:
        return 0.0
    scores = []
    for n in real:
        token = dur_to_token(n["duration_beats"])
        dist = token - SINGABLE_DUR_CENTER
        scores.append(math.exp(-0.5 * (dist / SINGABLE_DUR_WIDTH) ** 2))
    return sum(scores) / len(scores)


@dataclass(frozen=True)
class MusicalityScore:
    tonal_conformity: float
    contour_smoothness: float
    repetition: float
    call_response_relatedness: float
    singability: float
    overall: float


def musicality_score(notes: list, chord_idx: int, seed_phrase: list, weights: dict = None) -> MusicalityScore:
    """weights, if given, overrides DEFAULT_WEIGHTS for the overall combination
    only -- every sub-score is still computed and reported regardless (a metric
    "turned off" by setting its weight to 0.0 is still visible on the returned
    MusicalityScore, just not counted toward overall). Phase 13, DESIGN.md §11:
    the first per-session/per-gesture configuration point for the critic --
    see ensemble/sax.py's sax_generator for a live consumer (a director's
    toggle_singability gesture zeroing this at runtime)."""
    weights = weights if weights is not None else DEFAULT_WEIGHTS
    scores = {
        "tonal_conformity": tonal_conformity(notes, chord_idx),
        "contour_smoothness": contour_smoothness(notes),
        "repetition": repetition(notes),
        "call_response_relatedness": call_response_relatedness(seed_phrase, notes),
        "singability": singability(notes),
    }
    overall = sum(scores[key] * weights.get(key, 0.0) for key in scores)
    return MusicalityScore(overall=overall, **scores)
