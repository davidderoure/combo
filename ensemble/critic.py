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
   doesn't transfer directly here -- at the time this was written combo had no
   "truth set" of real jazz phrases to compare against (WJazzD was baked into
   Wolfson's trained weights only, not available as raw MIDI to combo), and
   Grade 1 piano's interval/contour norms are the wrong reference for
   idiomatic jazz -- but the TECHNIQUE (interval sequences; U/D/S contour
   strings compared by edit distance) is reused directly below. (Phase 28
   later did obtain real WJD data as a downloaded SQLite database,
   wjd_data/wjazzd.db -- see corpus_familiarity below and wjd_corpus.py --
   this paragraph is left as the honest record of the constraint at the time
   this module was first written.)

2. Wolfson's own ported bias layers (ensemble/wolfson/phrase_generator.py,
   ensemble/wolfson/scales.py) already encode real, previously-tuned musical
   judgments as GENERATION-TIME SAMPLING BIASES. Three are repurposed here as
   RETROSPECTIVE SCORING FUNCTIONS over completed output instead -- reusing
   already-vetted musical reasoning rather than inventing new theory.

Every function below is a pure, deterministic function over already-generated
data -- no model inference anywhere in this file, unlike everything else that
touches PhraseGenerator. All weights/thresholds (DEFAULT_WEIGHTS,
SMOOTH_INTERVAL_MAX_SEMITONES, AUTOCORRELATION_MAX_LAG) are explicit
placeholders, same honest status as INTENSITY_SPREAD (comping.py),
REFERENCE_MAX_DENSITY (director.py), and DEFAULT_MOTIF_STRENGTH (sax.py) --
needing real tuning once there's a way to listen and compare, not asserted as
musically validated.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from .corpus_motifs import CorpusMotifs
from .rhythm_motifs import extract_duration_motifs
from .wolfson.chords import N_QUALITIES, QUAL_DOM
from .wolfson.encoding import dur_to_token
from .wolfson.motifs import extract_interval_motifs
from .wolfson.phrase_generator import REST_PITCH, SINGABLE_DUR_CENTER, SINGABLE_DUR_WIDTH
from .wolfson.scales import chord_root, chord_to_mode, chord_tones, scale_pitch_classes

TONAL_RESOLUTION_WEIGHT = 0.3  # blend weight for "does the last note land on a chord tone"
SMOOTH_INTERVAL_MAX_SEMITONES = 4  # placeholder -- Grade 1 piano's own norms are far more
                                    # conservative than idiomatic jazz and aren't a valid
                                    # reference; needs real jazz-appropriate tuning.
MODAL_LEAP_SEMITONES = frozenset({5, 7})  # P4, P5 -- the quartal leaps Wolfson's own
                                            # modal_strength generation bias targets
                                            # (phrase_generator.py's MODAL_P4_BONUS/
                                            # MODAL_P5_BONUS); tolerated by
                                            # contour_smoothness only when modal=True.
AUTOCORRELATION_MAX_LAG = 4  # Phase 34 -- matches extract_interval_motifs' own longest
                               # n-gram (2, 3, 4); the longest repeat period repetition()
                               # checks for.
DISSONANT_SEMITONE_DISTANCE = 1  # a note exactly this far from the nearest in-scale pitch
                                   # class -- the "minor 9th"/major-7th-clash relationship,
                                   # judged as the harshest dissonance in ordinary melodic
                                   # playing, worse than landing further outside the scale
                                   # (read as deliberately "outside" rather than a clash) --
                                   # David's own musical judgment, not derived from anything
                                   # else in this codebase.
PASSING_TONE_MAX_STEP = 2  # semitones -- standard tonal-theory "step" (minor or major
                             # 2nd), shared by both _is_passing_tone (either side of a
                             # passing tone) and _is_resolved_tension (Phase 22, the
                             # resolution side of a tension-and-release) -- one "step"
                             # concept, not two constants for it. A placeholder like
                             # every other constant here, not asserted as musically final.
MIN_BREATH_BEATS = 0.5  # shortest duration counted as a genuine "breath"/sentence
                          # boundary (Phase 23) -- matches phrase_generator.py's own
                          # shortest producible rest (_REST_DURATIONS = [0.5, 1.0]),
                          # so nothing the model can currently produce is excluded; a
                          # forward-looking definition more than an active filter
                          # today, same honesty convention as every constant here.
TARGET_BREATH_FRACTION = 0.15  # placeholder, empirically grounded: 40 real 4-bar
                                 # chunks had mean breath fraction 0.129, median
                                 # 0.124 -- this target sits close to that natural
                                 # centre, nudged slightly higher to push selection
                                 # toward a bit MORE breath than the unweighted
                                 # default, matching what David actually asked for.
BREATH_FRACTION_WIDTH = 0.08   # placeholder -- same bell-curve treatment as
                                 # SINGABLE_DUR_CENTER/WIDTH below; tight enough that
                                 # a zero-breath chunk (David's actual complaint --
                                 # solos running on with no gaps) scores clearly
                                 # lower (~0.17) while chunks near the natural mean
                                 # score highly.

DEFAULT_WEIGHTS = {
    "tonal_conformity": 0.2,
    "contour_smoothness": 0.1,
    # POSITIVE again (Phase 34) -- Phase 33 flipped this negative after
    # Phase 30 found combo showed the OLD binary "any exact/near-repeat"
    # detector far more than real WJD solos (64.8% vs 29.4%). David's
    # correction: that conflated literal repetition with the real, positive
    # thing -- deliberate repeating PATTERN FRAGMENTS, which real solos use
    # to build longer structures. repetition() is now a continuous
    # autocorrelation-based measure of exactly that (Phase 34), so a real
    # positive quality is being rewarded again, not the old binary
    # literal-repeat flag. Magnitude restored to the original pre-Phase-33
    # value as the smallest deliberate starting point -- whether it's the
    # right size (or whether combo still over-shows even this better-
    # grounded measure relative to WJD) is checked empirically in Phase 34's
    # own verification, not assumed correct just because the polarity now
    # makes more sense.
    "repetition": 0.1,
    "call_response_relatedness": 0.1,
    "singability": 0.15,
    "phrasing": 0.15,
    "register_usage": 0.2,
    # Phase 36 -- a new placeholder weight, same honest not-yet-tuned status as
    # every other weight here. register_usage (span) and register_balance
    # (distribution) are genuinely different signals, both worth keeping --
    # this doesn't steal from register_usage's own weight. Weights already
    # don't sum to 1.0 (never a strict invariant, per Phase 33's own note);
    # this just adds one more term.
    "register_balance": 0.15,
    # Phase 40 -- a new placeholder weight, same status as every other. A
    # real, orthogonal signal to tonal_conformity (which doesn't weight by
    # duration at all): does time spent HOLDING a note land on a genuine
    # chord/quartal tone, not just any in-scale pitch class. Real empirical
    # check before picking this weight: 15 real one-shot generations over F7
    # showed sustain_quality varying 0.17-0.60 (mean 0.42) -- not degenerate,
    # a genuine discriminating signal.
    "sustain_quality": 0.15,
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


def tonal_conformity(
    notes: list, chord_idx: int, extra_tolerated: frozenset = frozenset(), credit_resolved_tension: bool = False
) -> float:
    """Fraction of real notes that are in-scale-or-excused, blended with a
    bonus for the phrase's FINAL note landing on an actual chord tone --
    mirrors how Wolfson's own voice-leading bias specifically ramps up at the
    end of a phrase, not uniformly across it (phrase_generator.py's
    arc_position-scaled chord-tone targeting). No real notes -> 0.0: nothing
    to be in- or out-of-key about.

    Phase 27: uses the SAME scale reference and exemptions as dissonance()
    (dissonance_scale, extra_tolerated, _is_passing_tone, and -- when
    credit_resolved_tension -- _is_resolved_tension) rather than the plain
    chord_to_mode scale it used through Phase 26. Checked directly before
    making this change, not assumed: every note that clears dissonance()'s
    badness gate as "not a clash" -- a bebop passing tone, a b5-substitution
    tritone, a functional-context ii-V-I note, a deliberately resolved
    tension -- was, until this phase, STILL counted against tonal_conformity,
    quietly penalising exactly the "advanced" playing Phases 19-22 were built
    to allow. Two real consequences this fixes: in n_candidates search, a bold
    candidate tied with a safe one on dissonance no longer automatically loses
    the overall tie-break; in RehearsalMemory, quality-weighted recall no
    longer systematically favours the safe phrasing over the bold one. This is
    a genuine, deliberate behaviour change for the scale reference itself
    (always the widened one now, matching dissonance()'s own baseline, not
    gated behind a flag) -- extra_tolerated/credit_resolved_tension are the
    only "extend on integration" parts (default empty/False)."""
    real = _real_notes(notes)
    if not real:
        return 0.0
    scale = dissonance_scale(chord_idx) | extra_tolerated
    tones = chord_tones(chord_idx)

    in_scale_count = 0
    for i, n in enumerate(real):
        pc = n["pitch"] % 12
        if pc in scale:
            in_scale_count += 1
        elif _is_passing_tone(real, i):
            in_scale_count += 1
        elif credit_resolved_tension and _is_resolved_tension(real, i, scale, tones):
            in_scale_count += 1
    scale_fraction = in_scale_count / len(real)

    resolves = 1.0 if (tones and real[-1]["pitch"] % 12 in tones) else 0.0

    return (1.0 - TONAL_RESOLUTION_WEIGHT) * scale_fraction + TONAL_RESOLUTION_WEIGHT * resolves


def _semitones_to_scale(pitch_class: int, scale: frozenset) -> int:
    """Shortest distance (0-6) from pitch_class to the nearest pitch class in
    scale, wrapping mod 12 in whichever direction is shorter."""
    return min((pitch_class - s) % 12 if (pitch_class - s) % 12 <= 6 else 12 - (pitch_class - s) % 12 for s in scale)


_RICHER_MODE_FOR = {"ionian": "bebop_major", "mixolydian": "bebop_dom"}
_MINOR_APPROACH_TONE_INTERVAL = 11  # semitones above root -- a chromatic
                                       # leading/approach tone a half-step BELOW
                                       # the root (mod 12); the identical
                                       # construction bebop_dom already applies
                                       # to mixolydian (mixolydian + interval 11),
                                       # applied here to dorian instead (Phase
                                       # 36). Grounded in a real, recurring
                                       # example from a listening-test MIDI
                                       # analysis: a chromatic approach tone
                                       # (pc 1, C#/Db) into Dm7's root D, found
                                       # three times. Minor has no comparably-
                                       # named MODES entry to look up the way
                                       # ionian/mixolydian do (Phase 20's own
                                       # named scope-cut) -- implemented as a
                                       # single pitch-class union instead, the
                                       # same pattern dissonance_scale already
                                       # uses for the dominant tritone-sub
                                       # color tone below, not a new mechanism.
                                       # Diminished is deliberately left alone:
                                       # its own base scale ([0,2,3,5,6,8,9,11])
                                       # already includes interval 11 and is far
                                       # richer than dorian's 7 notes, and the
                                       # real listening-test evidence pointed at
                                       # minor specifically, not diminished.


def _widened_mode_scale(chord_idx: int) -> frozenset:
    """Phase 20, Lever A: the plain chord_to_mode scale UNIONED with a named
    jazz-standard "richer" variant when one exists (nothing previously
    in-scale is lost, only real, named tensions gain tolerance) rather than
    replacing it -- e.g. mixolydian -> mixolydian | bebop_dom, recognising
    the maj7-as-passing-tone vocabulary over a dominant chord (the original
    "E natural over F7" example that started this). Diminished has no
    comparably-named richer variant in ensemble/wolfson/scales.py's MODES
    table -- not invented, a named scope-cut, not an oversight (see
    _MINOR_APPROACH_TONE_INTERVAL's own comment for why diminished stays
    untouched). Minor (dorian) gets a single extra chromatic approach-tone
    pitch class instead of a whole named scale (Phase 36) -- ported
    ensemble/wolfson/scales.py can't be edited to add a new MODES key, and
    real listening-test evidence pointed at a specific, single recurring
    color tone, not a whole richer scale. Factored out from dissonance_scale
    (Phase 21) so the tritone/b5 substitution term below can be layered on
    top without recursing into itself."""
    root = chord_root(chord_idx)
    mode = chord_to_mode(chord_idx)
    scale = scale_pitch_classes(root, mode)
    richer = _RICHER_MODE_FOR.get(mode)
    if richer:
        return scale | scale_pitch_classes(root, richer)
    if mode == "dorian":
        return scale | {(root + _MINOR_APPROACH_TONE_INTERVAL) % 12}
    return scale


def dissonance_scale(chord_idx: int) -> frozenset:
    """The scale reference dissonance() judges against. Public (not
    _dissonance_scale) since Phase 21: ensemble/sax.py now calls this
    directly (to compute a ii-V-I target's own widened scale for
    _functional_tonic_scale) -- the first time a critic.py helper is needed
    by production code outside this module, not just tests/tooling, so the
    leading underscore (which signals "no external consumers expected") is
    no longer honest.

    Starts from _widened_mode_scale (Phase 20, Lever A) above. For a
    DOMINANT chord, ALSO tolerates the single pitch class a tritone from the
    root -- tritone/b5 substitution (Phase 21, Lever D): the specific,
    named color tone real players use to evoke "playing the substitute"
    (e.g. Gb over F7), not the substitute chord's whole scale. Checked
    directly before choosing this, not assumed: unioning the FULL widened
    scale of the tritone-substitute chord saturates the metric almost
    completely -- F7's own widened scale is {0,2,3,4,5,7,9,10} (8 notes),
    its substitute B7's widened scale is {1,3,4,6,8,9,10,11} (8 notes),
    and their union is all 12 pitch classes, since a tritone is the most
    harmonically distant interval -- two mixolydian-family scales that far
    apart share almost nothing, so dissonance would never fire on a
    dominant chord again. A single extra pitch class avoids that entirely
    while still directly naming the technique David asked about ("a b5
    substitution" -- a specific color tone, not "the whole substitute chord
    is valid"). Doesn't touch scales.py (the ported file) or
    generation-time bias -- see _widened_mode_scale's own note on that."""
    scale = _widened_mode_scale(chord_idx)
    if chord_idx % N_QUALITIES == QUAL_DOM:
        scale = scale | {(chord_root(chord_idx) + 6) % 12}
    return scale


def _is_passing_tone(real_notes: list, i: int) -> bool:
    """True if real_notes[i] is approached AND left by step (<=
    PASSING_TONE_MAX_STEP semitones), continuing in the SAME direction on both
    sides -- connects two flanking pitches as a genuine melodic passing tone,
    the classical tonal-theory treatment of a dissonance (David's own example:
    a chromatically descending bass line "justifies" the semitone deltas along
    the way). Contrast a NEIGHBOUR tone, approached and left in OPPOSITE
    directions (e.g. C-D-C) -- a related but distinct device, deliberately not
    covered here. False at the very start/end of a phrase (no flanking note on
    one side -- nothing to pass between). Operates on the same
    _real_notes-filtered sequence every other function in this module already
    uses -- rests are already ignored for interval/contour purposes elsewhere
    (repetition, contour_smoothness), so this follows the same existing
    convention, not a new inconsistency."""
    if i == 0 or i == len(real_notes) - 1:
        return False
    prev_interval = real_notes[i]["pitch"] - real_notes[i - 1]["pitch"]
    next_interval = real_notes[i + 1]["pitch"] - real_notes[i]["pitch"]
    if abs(prev_interval) > PASSING_TONE_MAX_STEP or abs(next_interval) > PASSING_TONE_MAX_STEP:
        return False
    return (prev_interval > 0 and next_interval > 0) or (prev_interval < 0 and next_interval < 0)


def _is_resolved_tension(real_notes: list, i: int, scale: frozenset, tones: frozenset) -> bool:
    """True if real_notes[i] -- a clashing note -- reads as a deliberate,
    RESOLVED tension rather than an unresolved clash (Phase 22, prompted
    directly by David's own listening-test question: "I can hear the
    difference between conscious use of discordant intervals and use due to
    getting lost, panic, or playing randomly"): approached FROM an in-scale
    note (a single isolated reach outward, not part of a longer excursion
    that would read as "lost") and LEFT by step (<= PASSING_TONE_MAX_STEP
    semitones) onto an actual CHORD TONE (tones -- chord_tones(), not just
    any in-scale note: a genuine harmonic landing point, e.g. a b9
    resolving down a half-step to the root). Distinct from
    _is_passing_tone: that connects two flanking pitches by continuing
    THROUGH in one direction; this LANDS deliberately and then resolves.
    False at the very start/end of a phrase (nothing to approach from, or
    resolve into -- no way to tell intent from a fragment)."""
    if i == 0 or i == len(real_notes) - 1:
        return False
    if real_notes[i - 1]["pitch"] % 12 not in scale:
        return False
    next_interval = real_notes[i + 1]["pitch"] - real_notes[i]["pitch"]
    if abs(next_interval) > PASSING_TONE_MAX_STEP:
        return False
    return real_notes[i + 1]["pitch"] % 12 in tones


def dissonance(
    notes: list, chord_idx: int, extra_tolerated: frozenset = frozenset(), credit_resolved_tension: bool = False
) -> float:
    """Fraction of real notes that CLASH with the chord: pitch class out of
    scale and exactly DISSONANT_SEMITONE_DISTANCE semitones from the nearest
    in-scale pitch class -- found empirically, not assumed, by checking real
    generated output directly: every out-of-key note self_test.py produced
    landed exactly 1 semitone from the scale, never further. Deliberately
    distinct from tonal_conformity's plain in/out-of-scale fraction: a note
    further than DISSONANT_SEMITONE_DISTANCE from the scale isn't counted
    here at all (reads as deliberately "outside" rather than a clash, David's
    own judgment -- being further from the scale is not treated as worse).

    A genuine chromatic PASSING tone (_is_passing_tone above) is excused, not
    counted -- a real, distinct exception, not a loophole: the note still
    clashes with the chord in isolation, but its melodic context (approached
    and left by step, continuing through) is what tonal theory itself uses to
    justify a dissonance. A NEIGHBOUR tone (approached/left in opposite
    directions) is a related, still-uncovered case, named as a deliberate
    scope-cut, not an oversight.

    extra_tolerated (Phase 21): additional pitch classes to treat as in-scale,
    on top of dissonance_scale(chord_idx)'s own chord-local widening (Lever
    A/D). Default empty reproduces exactly Phase 20's behaviour -- this
    module stays deliberately Song-agnostic (see module docstring), so it's
    the CALLER's job to decide what extra context justifies tolerance (e.g.
    ensemble/sax.py's _functional_tonic_scale, informed by the surrounding
    chord sequence, which this module has no access to).

    credit_resolved_tension (Phase 22): when True, also excuses a clash that
    reads as a deliberate, resolved tension (_is_resolved_tension above) --
    "advanced" playing's use of #11/b9/#5-style color tones that resolve to
    consonance, distinct from extra_tolerated's scale-membership widening.
    Default False, unlike the always-on passing-tone exception: this isn't
    universally uncontroversial the way a passing tone is, it's the
    "advanced" behaviour itself, so a "beginner" default leaves it off.

    Higher is worse, unlike every other function in this module -- this is a
    badness signal for selection to minimise (ensemble/sax.py), not a
    goodness signal to blend into MusicalityScore/DEFAULT_WEIGHTS below,
    matching how motif_adherence is also kept separate. No real notes -> 0.0
    (nothing to clash)."""
    real = _real_notes(notes)
    if not real:
        return 0.0
    scale = dissonance_scale(chord_idx) | extra_tolerated
    tones = chord_tones(chord_idx) if credit_resolved_tension else frozenset()
    clashes = 0
    for i, n in enumerate(real):
        pc = n["pitch"] % 12
        if pc in scale:
            continue
        if _semitones_to_scale(pc, scale) != DISSONANT_SEMITONE_DISTANCE:
            continue
        if _is_passing_tone(real, i):
            continue
        if credit_resolved_tension and _is_resolved_tension(real, i, scale, tones):
            continue
        clashes += 1
    return clashes / len(real)


def contour_smoothness(notes: list, modal: bool = False) -> float:
    """Fraction of consecutive-note intervals within SMOOTH_INTERVAL_MAX_SEMITONES
    -- same core computation as the notebook's intervals() (signed semitone
    differences between consecutive notes). Fewer than 2 real notes -> 1.0:
    vacuously smooth, nothing observed to be a leap.

    modal (Phase 27): when True, a P4 or P5 leap (MODAL_LEAP_SEMITONES) also
    counts as smooth, on top of the existing <=4-semitone rule. Wolfson's own
    modal_strength generation parameter biases toward exactly these two
    intervals ("quartal/pentatonic character of modal jazz stages") -- without
    this, contour_smoothness would mark down the very leaps modal_strength is
    trying to produce. Default False reproduces today's behaviour exactly for
    every existing call site. A leap wider than P5 (e.g. a 9th) still counts
    as unsmooth regardless -- this widens the tolerance for a specific,
    named vocabulary, not a general loosening."""
    pitches = [n["pitch"] for n in _real_notes(notes)]
    if len(pitches) < 2:
        return 1.0
    intervals = [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]
    smooth = sum(
        1 for iv in intervals
        if abs(iv) <= SMOOTH_INTERVAL_MAX_SEMITONES or (modal and abs(iv) in MODAL_LEAP_SEMITONES)
    )
    return smooth / len(intervals)


def _autocorrelation(seq: list, lag: int) -> float:
    """Normalized (Pearson-style) autocorrelation of seq against itself
    shifted by `lag` steps -- a value near 1.0 means the sequence tends to
    repeat with that period. A constant sequence (zero variance -- e.g. a
    scale run, the same interval every step) returns 0.0: trivially
    self-similar in a way contour_smoothness/tonal_conformity already cover,
    not a "pattern fragment" in the motivic sense repetition() below cares
    about. 0.0 for a nonsensical lag (< 1, or >= the sequence length)."""
    n = len(seq)
    if lag < 1 or lag >= n:
        return 0.0
    mean = sum(seq) / n
    variance = sum((x - mean) ** 2 for x in seq)
    if variance == 0:
        return 0.0
    covariance = sum((seq[i] - mean) * (seq[i + lag] - mean) for i in range(n - lag))
    return covariance / variance


def repetition(notes: list) -> float:
    """How strongly this phrase shows a REPEATING PATTERN FRAGMENT -- Phase
    34, replacing an earlier binary "does any exact/near-repeat exist"
    detector. David's own correction to an even earlier fix (Phase 33, which
    treated all detected repetition as bad): literal, verbatim repetition
    isn't the thing that matters -- real solos build longer structures out
    of repeating FRAGMENTS, shapes rather than necessarily identical notes.
    Measured via autocorrelation (David's own suggestion) of the phrase's
    interval sequence AND its U/D/S contour sequence (numeric-encoded), at
    lags 1..AUTOCORRELATION_MAX_LAG -- transposition-invariant by
    construction (an interval/contour sequence doesn't care about absolute
    pitch), so this measures shape recurrence, not literal note repetition,
    directly matching "pattern fragment repeat, not literal repeat". Both
    sequences are checked, not just intervals: contour is a coarser,
    magnitude-blind reduction, so it catches "the same shape of ups/downs,
    different-sized steps each time" -- a genuine pattern-fragment case
    interval-only autocorrelation would miss.

    Deliberately does NOT try to distinguish deliberate development from
    being stuck in a loop -- a stuck, verbatim loop autocorrelates AT LEAST
    as strongly as a varied, developing restatement, so this metric can't
    make that distinction on its own, and isn't asked to (David's own call):
    other metrics (tonal_conformity, contour_smoothness, phrasing,
    register_usage, call_response_relatedness) are expected to catch
    genuinely stuck/dull playing through their own signals; a dedicated
    "stuck in a loop" or "dull solo" detector is a real, separate,
    explicitly deferred future idea if that turns out not to be enough.

    Score is the highest POSITIVE autocorrelation found across both
    sequences and all checked lags, clipped to [0, 1] -- negative
    (anti-)correlation is alternation, not a repeating fragment, and isn't
    counted. Fewer than 3 real notes -> 0.0: too short for any lag to be
    meaningful."""
    real = _real_notes(notes)
    pitches = [n["pitch"] for n in real]
    if len(pitches) < 3:
        return 0.0

    intervals = [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]
    contour_numeric = [{"U": 1, "D": -1, "S": 0}[c] for c in _contour_string(pitches)]

    best = 0.0
    for seq in (intervals, contour_numeric):
        for lag in range(1, min(AUTOCORRELATION_MAX_LAG, len(seq) - 1) + 1):
            if len(seq) - lag >= 2:  # at least 2 overlapping pairs -- a single
                                       # pair would make correlation trivially
                                       # perfect (or perfectly anti-) by chance
                best = max(best, _autocorrelation(seq, lag))
    return max(0.0, min(1.0, best))


def motif_adherence(notes: list, motif_targets: list) -> float:
    """1.0 if any of motif_targets appears among this phrase's OWN interval
    n-grams (extract_interval_motifs), else 0.0. Deliberately distinct from
    repetition() above: that measures whether a phrase repeats a pattern WITHIN
    ITSELF; this measures whether it echoes a SPECIFIC externally-supplied
    target (ensemble/memory.py's RehearsalMemory recall) -- a phrase that uses
    the target motif exactly once, without also repeating it again inside this
    same short chunk, scores 0.0 on repetition() but should score 1.0 here.
    Not part of MusicalityScore/DEFAULT_WEIGHTS below -- used only for
    candidate selection when a chunk actually has a target to aim for
    (ensemble/sax.py), not as a general-purpose quality signal. Empty
    motif_targets (nothing recalled) -> 0.0."""
    if not motif_targets:
        return 0.0
    real = _real_notes(notes)
    if len(real) < 2:
        return 0.0
    phrase_motifs = set(extract_interval_motifs(real))
    return 1.0 if any(target in phrase_motifs for target in motif_targets) else 0.0


def corpus_familiarity(notes: list, chord_quality: int, corpus: CorpusMotifs) -> float:
    """Fraction of this candidate's own pitch+duration motifs (pooled) that
    were actually seen in the Weimar Jazz Database under the SAME chord
    quality (CorpusMotifs, ensemble/corpus_motifs.py; quality is Wolfson's
    QUAL_MAJOR/QUAL_DOM/QUAL_MINOR/QUAL_DIM, matching what the LSTM itself
    conditions on) -- a corpus-grounded plausibility check, distinct from
    motif_adherence above (which checks a specific RECALLED target, not
    general corpus familiarity) and distinct from every other function in
    this module (none reference an external dataset).

    Deliberately NOT part of MusicalityScore/DEFAULT_WEIGHTS below, same
    precedent as motif_adherence and dissonance -- a standalone function for
    ensemble/sax.py's selection to use directly, and ONLY for chunks where
    generation is already pushed off the model's natural distribution
    (a recalled motif target or modal_strength) -- applying this to every
    candidate regardless would mostly just re-reward what the LSTM already
    learned to prefer from training on this exact corpus, and risks
    systematically favouring "sounds like average WJD" over the deliberate,
    bold playing Phases 17-24 pushed generation TOWARD, not away from. See
    DESIGN.md/README's Phase 29 paragraph for the full reasoning.

    No notes, or no extracted motifs at all -> 0.0 (nothing to be familiar
    or unfamiliar about, not vacuously "fully familiar")."""
    real = _real_notes(notes)
    pitch_motifs = extract_interval_motifs(real)
    duration_motifs = extract_duration_motifs(real)
    all_motifs = [("pitch", m) for m in pitch_motifs] + [("duration", m) for m in duration_motifs]
    if not all_motifs:
        return 0.0
    found = sum(
        1
        for kind, m in all_motifs
        if (corpus.has_pitch_motif(m, chord_quality) if kind == "pitch" else corpus.has_duration_motif(m, chord_quality))
    )
    return found / len(all_motifs)


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


def phrasing(notes: list) -> float:
    """Fraction of the chunk's total duration spent in a genuine breath, scored
    via a bell curve around TARGET_BREATH_FRACTION -- same style as singability()
    above. Prompted directly by a listening-test observation: "the solos are not
    speaking in 'sentences' with gaps in between." The only function in this
    module that looks at raw `notes` (including REST_PITCH sentinels) rather than
    _real_notes()-filtering them out first -- the "gaps" this measures ARE those
    sentinels, already spliced in by phrase_generator.py's own _inject_rests
    (bell-curve-weighted toward the phrase midpoint, capped at
    REST_MAX_PROBABILITY=0.15, each rest 0.5 or 1.0 beats -- naturally sparse,
    which is exactly what the listening test caught; no other function here ever
    looks at this).

    Too little breath (fraction near 0 -- a phrase that never pauses) reads as
    running on with no sentence structure; too much reads as fragmented,
    disjointed silence rather than phrasing -- the bell curve penalises both
    directions symmetrically, same treatment as singability's duration curve.
    Deliberately not a separate "sentence count" measure on top of this -- one
    fraction captures both space (breath amount) and, indirectly, phrasing (more
    breath in the target range naturally means more sentence-like division); a
    more explicit segmentation measure is a possible future refinement, not
    attempted here. No notes at all -> 0.0."""
    if not notes:
        return 0.0
    total_beats = sum(n["duration_beats"] for n in notes)
    if total_beats <= 0:
        return 0.0
    breath_beats = sum(
        n["duration_beats"] for n in notes
        if n["pitch"] == REST_PITCH and n["duration_beats"] >= MIN_BREATH_BEATS
    )
    fraction = breath_beats / total_beats
    dist = fraction - TARGET_BREATH_FRACTION
    return math.exp(-0.5 * (dist / BREATH_FRACTION_WIDTH) ** 2)


def register_usage(notes: list, register: Tuple[int, int], prior_range: Optional[Tuple[int, int]] = None) -> float:
    """Fraction of `register`'s width spanned by this chunk's IN-REGISTER real
    notes -- prompted directly by a listening-test observation ("not much range
    of the instrument is being used"). Deliberately measured over in-register
    notes only, not every real note candidate generation produces: candidate_notes
    (what musicality_score scores) are the model's RAW output, before
    _split_phrase_into_bars clips out-of-register pitches for playback -- checked
    directly, not assumed: 13 of 40 real 4-bar chunks had at least one
    out-of-register note (Wolfson's own trained pitch vocabulary is MIDI 44-93,
    wider than any `register` passed in), so a raw-span measurement can reward
    spread that never actually sounds.

    Deliberately NOT a bell curve around a target (unlike phrasing/singability):
    there's no clear "too much range" complaint the way there is for breath --
    `register` itself already caps what's appropriate for the chosen skill level,
    and contour_smoothness already penalises erratic wide leaps separately
    (SMOOTH_INTERVAL_MAX_SEMITONES). This is a monotonic "use more of what
    you're given" reward instead, balanced against the other six blended
    metrics rather than self-penalising at the top end.

    No separate beginner/advanced mode: the skill-level distinction lives
    entirely in which `register` the CALLER passes to sax_generator -- a
    narrow register already caps how much span is possible, so this metric
    doesn't need to know which mode is active.

    prior_range (Phase 32), if given: the (low, high) pitch bounds already
    explored by this SAME voice earlier in this performance (ensemble/sax.py
    tracks this across chunk-builds). Widens what the candidate is judged
    against from "just this chunk's own notes" to "everything this voice has
    played so far, including this candidate" -- checked directly against
    Phase 30's own finding that per-chunk-only measurement scored WJD's real,
    wide natural register (44-93) LOWER than combo's narrower SAX_REGISTER
    (55-79): backwards, because a short excerpt can't reflect a whole solo's
    real range regardless of how far the player actually goes. A candidate
    that only replays already-explored territory can't raise the combined
    span above prior_range's own (it's already counted); only a genuine
    excursion beyond what's already been played does -- directly rewarding
    "occasional excursions", not just "wide notes within one candidate".
    Default None reproduces exactly today's per-chunk-only behaviour.

    Fewer than 2 in-register real notes, and no prior_range -> 0.0 (nothing to
    span). With a real prior_range, even zero in-register candidate notes
    still reports prior_range's own span -- "this candidate contributed
    nothing new" is a real, honest answer, not forced to 0.0."""
    low, high = register
    width = high - low
    if width <= 0:
        return 0.0
    in_register = [n["pitch"] for n in _real_notes(notes) if low <= n["pitch"] <= high]
    if prior_range is None:
        if len(in_register) < 2:
            return 0.0
        return (max(in_register) - min(in_register)) / width
    p_low, p_high = prior_range
    combined_low = min([p_low] + in_register)
    combined_high = max([p_high] + in_register)
    return (combined_high - combined_low) / width


def register_balance(
    notes: list, register: Tuple[int, int], prior_mean_beats: Optional[Tuple[float, float]] = None
) -> float:
    """How close this voice's CUMULATIVE, duration-weighted mean pitch sits to
    register's own midpoint -- the DISTRIBUTION counterpart to register_usage's
    SPAN (Phase 36). register_usage rewards having touched the register's
    boundary at least once; nothing in it then pulls a voice back toward
    spending TIME away from wherever it's been sitting -- this is that missing
    pressure.

    Found necessary by a real listening-test MIDI analysis: register
    mean/median sat near the BOTTOM of SAX_REGISTER for two entire real
    recorded performances (blues_in_f.chart and songs/ii_v_i.chart) despite
    the full span being touched early, with no progressive drift across the
    performance -- span alone was cheaply satisfied once and never revisited.

    Duration-weighted, not note-count-weighted: a long held note should count
    for more than a quick passing one, matching how a listener actually
    perceives "where the solo is sitting".

    prior_mean_beats, if given: (weighted_pitch_sum, total_beats) already
    accumulated by THIS voice earlier this performance (ensemble/sax.py
    tracks this across chunk-builds, mirroring register_usage's own
    prior_range -- Phase 32). Combined with this candidate's own in-register
    notes before computing the mean. None (every pre-Phase-36 caller and any
    bar-0 call) means only this candidate's own short chunk is judged -- a
    weak, noisy signal alone; the real, intended use is always with a real
    prior_mean_beats once a performance is under way.

    Measured over IN-REGISTER real notes only, same reasoning as
    register_usage (candidate_notes are the model's raw output before
    out-of-register pitches get clipped at playback).

    No in-register notes and no prior_mean_beats -> 0.0. No in-register
    candidate notes but a real prior_mean_beats -> scores prior_mean_beats'
    own mean unchanged -- "contributed nothing new" is a real answer, not
    forced to 0.0, same honesty convention as register_usage."""
    low, high = register
    width = high - low
    if width <= 0:
        return 0.0
    center = (low + high) / 2.0
    in_register = [n for n in _real_notes(notes) if low <= n["pitch"] <= high]
    candidate_sum = sum(n["pitch"] * n["duration_beats"] for n in in_register)
    candidate_beats = sum(n["duration_beats"] for n in in_register)
    if prior_mean_beats is not None:
        prior_sum, prior_beats = prior_mean_beats
        combined_sum, combined_beats = prior_sum + candidate_sum, prior_beats + candidate_beats
    else:
        combined_sum, combined_beats = candidate_sum, candidate_beats
    if combined_beats <= 0:
        return 0.0
    combined_mean = combined_sum / combined_beats
    return max(0.0, 1.0 - abs(combined_mean - center) / (width / 2.0))


def _quartal_tones(root: int) -> frozenset:
    """A basic quartal triad on root -- two stacked perfect 4ths (root, +5,
    +10), e.g. C: C-F-Bb -- the real "So What" voicing shape (root E: E-A-D
    is root+5+10 from E). Combo-authored: chord_tones() (ported, Wolfson) has
    no quartal concept at all, always quality-conditioned tertian root/3rd/7th
    -- quality-NEUTRAL by design here, matching how real quartal harmony is
    characterized by interval structure, not major/minor quality."""
    return frozenset({root % 12, (root + 5) % 12, (root + 10) % 12})


def sustain_quality(notes: list, chord_idx: int, modal: bool = False) -> float:
    """Duration-weighted fraction of a phrase's real note-TIME spent on a
    genuine chord tone -- tertian (chord_tones()) normally, or a quartal
    stack (_quartal_tones) when modal=True -- rather than merely any
    in-scale pitch class (Phase 40).

    Prompted directly by a listening-test question: "are we giving credit
    for holding notes that are in the triad/quartal vs those that are just
    in the scale?" Checked the existing code before answering: no, on both
    counts. tonal_conformity() counts every in-scale note equally toward
    in_scale_count regardless of duration -- a whole-note 9th scores
    identically, per note, to a whole-note root. The only chord-tone-aware
    check anywhere is a single binary bonus for whether the phrase's LAST
    note resolves onto one; nothing rewards a note held MID-phrase for
    landing on something harmonically strong. This is a new, standalone
    metric for exactly that gap -- a sibling to tonal_conformity (scale
    membership), not a replacement: tonal_conformity/dissonance already own
    "is this in scale / a clash"; this measures something orthogonal, "when
    you DO sit on a note, is it a strong one."

    Duration-weighted, with no arbitrary "sustained" threshold: a note's own
    duration is simply its weight in the fraction, so holding a chord tone
    longer earns more credit and holding a non-chord-tone longer costs more
    -- the same "weight by real elapsed time" idea register_balance (Phase
    36) already uses, not a new pattern. Doesn't require scale membership
    either -- an out-of-scale held note still counts against this (0
    contribution to the numerator, full weight in the denominator), since
    that's a real, if different, way of "not sitting on a strong tone".

    No real notes, or no chord tones for this chord_idx (NC) -> 0.0."""
    real = _real_notes(notes)
    if not real:
        return 0.0
    tones = _quartal_tones(chord_root(chord_idx)) if modal else chord_tones(chord_idx)
    if not tones:
        return 0.0
    on_tone_beats = sum(n["duration_beats"] for n in real if n["pitch"] % 12 in tones)
    total_beats = sum(n["duration_beats"] for n in real)
    return on_tone_beats / total_beats if total_beats > 0 else 0.0


@dataclass(frozen=True)
class MusicalityScore:
    tonal_conformity: float
    contour_smoothness: float
    repetition: float
    call_response_relatedness: float
    singability: float
    phrasing: float
    register_usage: float
    register_balance: float
    sustain_quality: float
    overall: float


def musicality_score(
    notes: list, chord_idx: int, seed_phrase: list, register: Tuple[int, int], weights: dict = None,
    extra_tolerated: frozenset = frozenset(), credit_resolved_tension: bool = False, modal: bool = False,
    prior_range: Optional[Tuple[int, int]] = None, prior_mean_beats: Optional[Tuple[float, float]] = None,
) -> MusicalityScore:
    """weights, if given, overrides DEFAULT_WEIGHTS for the overall combination
    only -- every sub-score is still computed and reported regardless (a metric
    "turned off" by setting its weight to 0.0 is still visible on the returned
    MusicalityScore, just not counted toward overall). Phase 13, DESIGN.md §11:
    the first per-session/per-gesture configuration point for the critic --
    see ensemble/sax.py's sax_generator for a live consumer (a director's
    toggle_singability gesture zeroing this at runtime).

    register (Phase 24): required, no sensible default exists (same reasoning
    as chord_idx/seed_phrase having none) -- the active register bound
    register_usage() needs, which this function never received before.

    extra_tolerated/credit_resolved_tension (Phase 27): passed straight through
    to tonal_conformity, the SAME context dissonance() is judged against
    elsewhere in ensemble/sax.py's selection loop -- keeps the badness gate and
    this positive quality signal in agreement about what counts as legitimate.
    modal (Phase 27): passed straight through to contour_smoothness, matching
    whichever modal_strength was used for this same candidate's generation --
    and, since Phase 40, to sustain_quality too (quartal tones instead of
    tertian when True).
    prior_range (Phase 32): passed straight through to register_usage -- see
    its own docstring. prior_mean_beats (Phase 36): passed straight through to
    register_balance -- see its own docstring. All five default to their
    respective pre-existing behaviour."""
    weights = weights if weights is not None else DEFAULT_WEIGHTS
    scores = {
        "tonal_conformity": tonal_conformity(notes, chord_idx, extra_tolerated, credit_resolved_tension),
        "contour_smoothness": contour_smoothness(notes, modal),
        "repetition": repetition(notes),
        "call_response_relatedness": call_response_relatedness(seed_phrase, notes),
        "singability": singability(notes),
        "phrasing": phrasing(notes),
        "register_usage": register_usage(notes, register, prior_range),
        "register_balance": register_balance(notes, register, prior_mean_beats),
        "sustain_quality": sustain_quality(notes, chord_idx, modal),
    }
    overall = sum(scores[key] * weights.get(key, 0.0) for key in scores)
    return MusicalityScore(overall=overall, **scores)
