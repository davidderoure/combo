"""Real generation for the sax voice — DESIGN.md §12, Phases 8-10 of the build plan.

The first voice to move off ensemble/generators.py's chord_tone_generator stub.
Wraps ensemble/wolfson/'s ported PhraseGenerator (an LSTM adapted from David's
earlier system, Wolfson) behind combo's ordinary Generator interface. Everything
here is combo-authored glue, not ported — structured the same context-driven shape
as ensemble/drums.py and ensemble/comping.py, though determinism works differently
here (see sax_generator's docstring: a global torch seed, not a local RNG), with
four real integration seams:

  1. chord_to_wolfson_index — combo's Chord -> Wolfson's chord vocabulary.
  2. _build_seed_phrase — a target voice's recent notes -> Wolfson's seed_phrase,
     mirroring comping_generator's lookback-window pattern exactly.
  3. _bars_until_chord_change — how far ahead a plan can safely span without
     crossing a chord change (PhraseGenerator.generate() only accepts one chord,
     broadcast across the whole call — it can't represent a chord change mid-call).
  4. _split_phrase_into_bars — splits a single multi-bar generate() call's output
     across the bars it spans. Necessary because max_phrase_beats does NOT bound
     the returned phrase's span (verified empirically: rest injection runs after
     the beat cap and isn't counted against it) — see
     ensemble/wolfson/phrase_generator.py's generate() docstring.

The director's aggregated intensity (DESIGN.md §11) drives rhythmic_density —
PhraseGenerator.generate()'s own docstring frames that parameter as "0-1 busyness
... 0=lyrical/slow, 1=bebop/fast", the one bias knob the model itself already
treats as a general busyness dial, so no translation function is needed (both
values are already 0-1 in the same direction, and generate() clamps internally).
Intensity is captured once per planned chunk, not re-read bar to bar within a
chunk — a deliberate simplification (see sax_generator's docstring), not a
structural necessity like the chord case.

Multi-bar planning (Phase 10, DESIGN.md §12): sax_generator plans `plan_bars`
bars ahead in ONE continuous generate() call whenever its buffer runs out, rather
than one independent call per bar. This is what "planning ahead" mechanically
means here — the model's own arc_position-driven bias layers (voice-leading,
contour) now sweep across the real planned span instead of resetting every bar.
No revision-on-mismatch mechanism exists, or is needed: TransitionController's
effective_song only ever replaces Song.form, and Song.chord_at only ever reads
Song.changes — a handover can never change what chord a given bar resolves to,
and Session.generate()'s bar_index always advances by exactly +1, so a plan's
chord assumptions can never go stale between when it's built and when it's
dispensed. Verified directly in ensemble/transitions.py and song/song.py, not
assumed — building a revision mechanism anyway would be untested, unreachable
code.

Rehearsal memory (Phase 11, DESIGN.md §12): sax_generator optionally takes a
RehearsalMemory (ensemble/memory.py) — the first thing in combo that persists
across separate Session.generate() calls, on purpose, unlike everything else here.
Read and written at exactly the point a new plan chunk is built, which gives two
kinds of persistence from one piece of code: within-run (chunk 2 can draw on what
chunk 1 just played, same Session.generate() call) and cross-run (chunk 1 of a
*new* Session.generate() call can draw on the last chunk of a *previous* one, if
the same RehearsalMemory object is passed to both — the rehearsal-informs-the-gig
idea this was built for). Since Phase 12 (ensemble/critic.py), what's remembered
is quality-weighted, not pure frequency: each chunk's musicality_score is computed
right here (chord_idx and seed_phrase are already local at this exact point, no
new data plumbing) and passed to memory.store() alongside the notes, so
recall_motifs() favours motifs from higher-scoring phrases — see
ensemble/critic.py's module docstring for how "quality" is measured and how
honestly-placeholder that is. Recall is also chord-quality-aware (Phase 25):
chord_quality = chord_idx % N_QUALITIES is computed once per chunk (same
timing as functional_scale) and passed to both memory.recall_motifs() and
memory.store(), so a chunk over a dominant chord only ever recalls motifs
this run stored over other dominant chords — "what worked here", not "what
worked anywhere". Tagged by Wolfson's 4-class quality, not root or full
chord_idx — extract_interval_motifs is already transposition-invariant, so a
shape is equally valid over any root of the same quality; see
ensemble/memory.py's module docstring for the full reasoning. Strict, no
cross-quality fallback: a quality with no history yet in this run simply
recalls nothing, same as memory=None.

Director gesture toggle (Phase 13, DESIGN.md §11): the first real consumer of
DirectorSignal.gesture since the dial channel was built (Phase 5) — every phase
since had repeated some version of "a director-emitted gesture has nowhere to
act." Checked every bar (not just at chunk-build time, so a mid-chunk gesture
isn't missed): Gesture("toggle_singability") flips a mutable, session-local copy
of ensemble/critic.py's DEFAULT_WEIGHTS between singability counting toward
musicality_score's overall and not, letting a director (human or AI, and per
Phase 13's "role determines destination, not capability" principle, a director
sitting at a keyboard using the exact same gesture vocabulary a performer would)
turn the singability metric on/off live — useful, for instance, when a teacher
wants to let a student's fast, exploratory playing not be marked down for being
unsustained. Exposed as generate.critic_weights (a plain attribute on the
returned closure) so tests can assert the toggle happened directly, rather than
re-deriving the effect statistically.

Search-and-evaluate (Phase 14, DESIGN.md §13 — "the chess approach"): sax_generator
optionally generates `n_candidates` candidates per chunk (identical arguments each
time — the model's own RNG state naturally diversifies successive calls, no extra
diversity logic needed) and keeps the highest-scoring one by
`musicality_score(...).overall`. DESIGN.md §13 originally called this "a poor fit
for live performance, which can't pause to search before committing" — checked
directly, not assumed, once there was something to measure: even 20 candidates
over a full 4-bar chunk costs ~164ms against a 7.3-second real-time budget at
blues tempo. Not restricted to machine_speed, but honestly still unverified for
true live call-and-response specifically — the measurement is against `Session`'s
nominal per-bar pacing budget, not human-perceived conversational latency, which
is a different, more subtle question this phase doesn't answer. A side effect
worth noting: `musicality_score` is now always computed once a chunk is built
(previously only `if memory is not None:`), since selection needs it regardless
of whether memory is configured — `memory.store()` reuses that same computation
rather than re-deriving it. At `n_candidates=1` (the default) this is exactly
today's behaviour: one `generate()` call, one score. Exposed for testing as
`generate.last_candidate_scores` (every candidate's `.overall` from the most
recent chunk-build, in generation order) — same convention as
`generate.critic_weights`. Deliberately NOT attempted: varying anything besides
the random draw across candidates (temperature, rhythmic_density — a genuinely
different, larger idea, searching generation *parameters* rather than
re-sampling fixed ones); revision after committing (ImproteK's actual
architecture, DESIGN.md §12); a director-gesture-driven `n_candidates` toggle
(the natural next use of Phase 13's exact critic_weights-mutation pattern, not
built here).

Tension-and-resolution crediting (Phase 22, DESIGN.md §12): sax_generator accepts
credit_resolved_tension: bool = False, threaded straight into every dissonance()
call in the search loop. Prompted directly by a listening-test question -- "I can
hear the difference between conscious use of discordant intervals and use due to
getting lost, panic, or playing randomly. I wonder how we could encode that." --
answered by generalising the existing passing-tone exception (Phase 19) to a
second, distinct melodic device: a clash approached from an in-scale note (a
single isolated reach outward, not mid-excursion) and resolved by step onto an
actual chord tone (ensemble/critic.py's _is_resolved_tension) reads as deliberate
"advanced" playing -- #11/b9/#5-style tensions resolving to consonance -- rather
than noise. Off by default (unlike the always-on passing-tone exception): this
isn't universally uncontroversial the way a passing tone is, it's the "advanced"
behaviour itself, so a "beginner" default leaves it off, matching every existing
call site's behaviour exactly. Explicitly narrower than what it might sound like:
covers only a SINGLE isolated tension-then-resolution note, not a multi-note
excursion (real side-slipping -- shifting a whole pattern outside the changes and
back -- needs actual generation-time mechanics, not a scoring exemption, and isn't
attempted here); doesn't reward tension use, only stops penalising it when
resolved; and isn't wired to a live director gesture this phase (every same-symbol
gesture pattern short enough to be practical is already claimed -- see
gesture/vocabulary.py -- a new pattern needs its own design and empirical
verification, Phase 13/20's discipline, not bundled in here). Register-as-
skill-level and phrasing/"sentences with gaps" are two more adjacent ideas from the
same discussion, also deliberately not attempted here -- a call-site register
choice and a different critic dimension (rest structure) respectively, neither of
which this parameter touches.

The critic no longer fights the generator (Phase 27): `musicality_score`'s call
here passes `extra_tolerated`/`credit_resolved_tension` into `tonal_conformity`
(the SAME context `dissonance()` above is judged against — until this phase,
`tonal_conformity` still checked the plain, unwidened scale, so a candidate that
cleared the dissonance gate via a resolved tension still lost `tonal_conformity`
points for the exact same note) and `modal=song.modal` into `contour_smoothness`
(so a genuine P4/P5 quartal leap, once `modal_strength` is actually asked to
produce more of them below, isn't marked down as "unsmooth"). `Song.modal`
(`song/song.py`, Phase 27) is a chart-authored style choice — David's own framing:
for now it's read off the chart the way he'd read it off a score's artist/date,
with modulating it by narrative-arc position named as a real, separate future
step, not attempted here. `MODAL_STRENGTH_WHEN_ACTIVE` feeds Wolfson's own
`modal_strength` generation parameter (ported, previously always 0.0/unused) —
verified empirically, not assumed, that it has a real effect: 20 real one-shot
generations at `modal_strength=0.0` vs `1.0` show P4/P5 leaps rising from 10.0%
of intervals to 26.0%.

Responding to its own previous phrase (Phase 37, DESIGN.md §12): prompted by
David's observation that Wolfson's own self-play sounded much more alive to
listening musicians than combo's current sax, despite sharing the same
generator core. Checked directly against Wolfson's actual source, not
assumed: `main.py`'s self-play mode literally feeds the sax's own just-played
notes back in as the next call's seed ("the sax continuously responds to
itself"), alternating MIDI channels purely so it *sounds* like two voices
trading. combo's own sax_generator always seeds from `target_voice_id`
("bass" at every real call site) and never its own prior output.
`sax_generator` gains `own_voice_id: Optional[str] = None` — when given, the
seed phrase becomes `target_voice_id`'s recent notes followed by this voice's
OWN recent notes (`_build_combined_seed_phrase` above), added ALONGSIDE the
target voice, not instead of it — a full self-only swap would throw away
real ensemble listening a merge doesn't need to give up. Plain concatenation,
not a time-sorted interleave: interleaving two simultaneous voices by onset
would mix bass-register and sax-register pitches into one monophonic token
stream the model was never trained to expect; concatenation avoids that
entirely (see `_build_combined_seed_phrase`'s own docstring for the full
reasoning). Default `None` reproduces today's bass-only seeding exactly.
Explicitly deferred: Wolfson's separate riff/repeat mechanism (literally
replaying the last phrase verbatim a few times before evolving it) — a
different, repeat-then-develop device, not attempted here; extending this to
`markov_sax_generator` (Phase 35, deliberately kept minimal/comparison-
focused).

A deliberate rest between chunks (Phase 39, DESIGN.md §12): prompted directly by
David's own diagnosis of a listening test — not raw gap count but "speaking in
sentences," distinct phrases a listener can tell apart, which "came for free" with
Wolfson's lick-trading/self-play and is "one of the challenges of moving into the
combo architecture." Checked the actual mechanism, not assumed: Wolfson's self-play
generates one phrase per call, plays it to completion, THEN re-injects it as the
next call's seed after a short gap — silence between phrases is a structural side
effect of that turn-taking loop. combo's plan-chunk architecture has no equivalent:
`_inject_rests` (`ensemble/wolfson/phrase_generator.py`) is the only source of any
rest at all, and its own bell-curve probability (`4·x·(1−x)·REST_MAX_PROBABILITY`)
is mathematically zero at both ends of a `generate()` call — a chunk boundary is
exactly where a rest is least likely to occur on its own. Fixed in combo-authored
code only (the ported model is untouched): right after a new chunk's winning
candidate is selected, a rest is prepended to it — skipped before the very first
chunk of the performance (nothing to separate it from) — reusing `_inject_rests`'
own rest-sentinel shape (`PHRASE_BOUNDARY_REST_BEATS`, duplicating its private
`_REST_DURATIONS` values rather than importing them, same precedent as
`ensemble/critic.py`'s `MIN_BREATH_BEATS`) so `_split_phrase_into_bars` (already
`REST_PITCH`-aware) needs no changes. Costs a small, honest slice of the chunk's
own beat budget, matching how `_inject_rests`' own mid-phrase rests already work —
not a double standard. David separately named a related but deliberately deferred
idea while discussing this: real players often anticipate an imminent chord change
rather than merely continue across it — not attempted here.

Explicitly deferred (see DESIGN.md §12 for the full list): all ~10 of
PhraseGenerator.generate()'s OTHER bias-layer knobs (contour, energy arc, register
contrast, etc. — motif_targets/motif_strength are now wired, via memory) are left
at their defaults — nothing in combo supplies them yet, the same "deliberately
dumb" spirit as chord_tone_generator itself. Hidden-state continuity now exists
WITHIN a planned chunk (which can span several bars, chord-hold permitting) but
still resets BETWEEN chunks — genuinely extended from Phase 8/9, not solved.
`register` is a backstop that drops out-of-range notes, not a real voicing control
like it is in chord_tone_generator/comping_generator — Wolfson's trained pitch
vocabulary is hard-clipped to MIDI 44-93 by the model itself.
"""

import random
from collections import Counter, deque
from typing import List, Optional, Tuple

from song.chord import Chord

from .corpus_motifs import CorpusMotifs
from .critic import DEFAULT_WEIGHTS, corpus_familiarity, dissonance, dissonance_scale, motif_adherence, musicality_score
from .memory import RehearsalMemory
from .timeline import BEATS_PER_BAR, NoteEvent, Timeline
from .voice import Generator
from .wolfson.chords import N_QUALITIES, QUAL_DIM, QUAL_DOM, QUAL_MAJOR, QUAL_MINOR
from .wolfson.phrase_generator import REST_PITCH, PhraseGenerator

DEFAULT_VELOCITY = 75
DEFAULT_PLAN_BARS = 4  # matches Wolfson's own MAX_PHRASE_BEATS=16.0 -- 4 bars at
                        # 4/4, "the model's own planning horizon," not arbitrary.
DEFAULT_MOTIF_STRENGTH = 1.0  # placeholder, same status as INTENSITY_SPREAD
                               # (comping.py) / REFERENCE_MAX_DENSITY (director.py)
                               # -- needs real tuning once there's a way to hear it.
MODAL_STRENGTH_WHEN_ACTIVE = 1.0  # placeholder, same status as above -- full
                                    # quartal/pentatonic bias (Wolfson's own
                                    # modal_strength, phrase_generator.py) when
                                    # song.modal is True (Phase 27), verified
                                    # empirically to have a real effect (20 real
                                    # one-shot generations: P4/P5 leaps go from
                                    # 10.0% of intervals at 0.0 to 26.0% at 1.0).
PHRASE_BOUNDARY_REST_BEATS = (0.5, 1.0)  # same rest-duration vocabulary as
                                            # phrase_generator.py's own (private,
                                            # ported) _REST_DURATIONS -- duplicated,
                                            # not imported, same precedent as
                                            # critic.py's MIN_BREATH_BEATS (Phase 39).

# combo's 17 canonical qualities (song/chord.py's _QUALITY_ALIASES values) mapped
# onto Wolfson's 4 harmonic-function classes. Root translation is the identity
# function: combo's _ROOT_PITCH_CLASS and Wolfson's ROOTS both index 0=C..11=B.
_QUALITY_TO_WOLFSON_CLASS = {
    "maj": QUAL_MAJOR, "maj7": QUAL_MAJOR, "6": QUAL_MAJOR, "maj9": QUAL_MAJOR,
    "7": QUAL_DOM, "sus4": QUAL_DOM, "9": QUAL_DOM, "13": QUAL_DOM,
    "7#11": QUAL_DOM, "7b9": QUAL_DOM, "7alt": QUAL_DOM,
    "m": QUAL_MINOR, "m7": QUAL_MINOR, "m6": QUAL_MINOR, "m9": QUAL_MINOR,
    "m7b5": QUAL_DIM, "dim7": QUAL_DIM,
}


def chord_to_wolfson_index(chord: Chord) -> int:
    """combo's Chord -> Wolfson's chord_idx (root * N_QUALITIES + quality_class).
    Total over combo's closed quality vocabulary — no NC/fallback branch needed."""
    quality_class = _QUALITY_TO_WOLFSON_CLASS[chord.quality]
    return chord.root * N_QUALITIES + quality_class


def _build_seed_phrase(timeline: Timeline, target_voice_id: str, since_beat: float, until_beat: float) -> list:
    """target_voice_id's notes in [since_beat, until_beat) -> Wolfson seed_phrase
    dicts. onset/offset use start_beat directly with beat_dur_sec=1.0, which makes
    phrase_to_tokens's duration formula reproduce event.duration_beats exactly (no
    tempo round-trip). Mirrors comping_generator's lookback-window pattern."""
    return [
        {
            "pitch": event.pitch,
            "onset": event.start_beat,
            "offset": event.start_beat + event.duration_beats,
            "beat_dur_sec": 1.0,
        }
        for event in timeline
        if event.voice_id == target_voice_id and since_beat <= event.start_beat < until_beat
    ]


def _build_combined_seed_phrase(
    timeline: Timeline, target_voice_id: str, own_voice_id: Optional[str], since_beat: float, until_beat: float
) -> list:
    """seed_phrase = target_voice_id's recent notes, followed by this voice's OWN
    recent notes (own_voice_id) when given -- letting generation respond to both
    the other voice AND continue its own previous idea, mirroring Wolfson's own
    self-play ("the sax continuously responds to itself" -- main.py's own module
    docstring). Plain CONCATENATION, not a time-sorted interleave -- checked
    directly that interleaving by onset would mix two simultaneous voices'
    pitches into one monophonic token stream the model was never trained to
    expect; phrase_to_tokens (ensemble/wolfson/encoding.py) computes each note's
    duration from its own onset/offset alone, never a neighbour's, so
    concatenation order doesn't affect any note's duration -- exactly as safe as
    a single-voice seed, just longer. own_voice_id=None (every pre-existing
    caller) reproduces _build_seed_phrase(timeline, target_voice_id, ...)
    exactly -- extend on integration."""
    seed = _build_seed_phrase(timeline, target_voice_id, since_beat, until_beat)
    if own_voice_id is not None:
        seed = seed + _build_seed_phrase(timeline, own_voice_id, since_beat, until_beat)
    return seed


def _bars_until_chord_change(song, start_beat: float, max_bars: int) -> int:
    """How many bars from start_beat keep start_beat's own chord, capped at
    max_bars — 1 means the very next bar already changes chord. Chord equality
    (a frozen dataclass) does the comparison. Song.chord_at cycles through
    Song.changes regardless of Song.form/section state (see module docstring),
    so no section/handover special-casing is needed here."""
    starting_chord = song.chord_at(start_beat)
    for offset in range(1, max_bars):
        if song.chord_at(start_beat + offset * BEATS_PER_BAR) != starting_chord:
            return offset
    return max_bars


def _ii_v_i_target(song, ii_beat: float) -> Optional[int]:
    """Checks whether the chords at ii_beat, ii_beat+BEATS_PER_BAR, and
    ii_beat+2*BEATS_PER_BAR form a textbook major ii-V-I: roots descending by
    fifths (+5 semitones mod 12 at each step) and qualities minor/dominant/
    major -- Wolfson's own 4-class mapping, already the right granularity
    (it already collapses m7/m9/m6 into one minor class, 7/9/13/alt into one
    dominant class, maj7/6/maj9 into one major class). Returns the I chord's
    chord_idx if matched, else None. Bar-granular only (checks downbeat
    chords at bar boundaries) -- a real, honest limit on a chart with faster
    harmonic rhythm (more than one chord change per bar), same category of
    limit _bars_until_chord_change's own docstring already names for
    blues_in_f.chart. song.chord_at is cyclic (Changes.chord_at wraps mod
    total_beats, verified directly -- never raises), so look-ahead/
    look-behind near a chart's boundary is always safe."""
    ii_idx = chord_to_wolfson_index(song.chord_at(ii_beat))
    v_idx = chord_to_wolfson_index(song.chord_at(ii_beat + BEATS_PER_BAR))
    i_idx = chord_to_wolfson_index(song.chord_at(ii_beat + 2 * BEATS_PER_BAR))
    ii_root, ii_qual = ii_idx // N_QUALITIES, ii_idx % N_QUALITIES
    v_root, v_qual = v_idx // N_QUALITIES, v_idx % N_QUALITIES
    i_root, i_qual = i_idx // N_QUALITIES, i_idx % N_QUALITIES
    if ii_qual != QUAL_MINOR or v_qual != QUAL_DOM or i_qual != QUAL_MAJOR:
        return None
    if v_root != (ii_root + 5) % 12 or i_root != (v_root + 5) % 12:
        return None
    return i_idx


def _functional_tonic_scale(song, bar_start: float) -> frozenset:
    """Extra pitch classes to tolerate at bar_start from simplifying a
    recognised ii-V-I to the tonic (Phase 21, Lever E -- "look into how we
    can do the II-V-I simplification... good for teaching"): checks whether
    bar_start could be the ii, V, or I bar of such a pattern (three
    hypotheses, using _ii_v_i_target's cyclic lookback/lookahead), and if
    matched, unions in the target I chord's own dissonance_scale (composing
    with Lever A/D automatically -- e.g. the V chord's own widened scale
    already covers most of the target key, so this mainly adds real
    tolerance at the ii position, checked directly: D-dorian and
    C-major-widened differ by exactly one pitch class). A vi-ii-V-I
    (four-chord) extension and a minor-tonic ii-V-i variant are real,
    reasonable follow-ons -- deliberately not attempted here, named rather
    than silently out of reach."""
    for offset in (0.0, -BEATS_PER_BAR, -2 * BEATS_PER_BAR):
        target = _ii_v_i_target(song, bar_start + offset)
        if target is not None:
            return dissonance_scale(target)
    return frozenset()


def _split_phrase_into_bars(
    notes: list, plan_start: float, n_bars: int, register: Tuple[int, int]
) -> List[List[NoteEvent]]:
    """Split a single multi-bar generate() call's output across the n_bars it
    spans, one continuous beat cursor from plan_start onward. Same discipline as
    Phase 8's _place_phrase_in_bar (which this replaces): skips REST_PITCH
    sentinels as silent gaps, drops notes outside `register` (a backstop, not a
    real voicing control — see module docstring). Different from Phase 8's
    single-bar version in one respect: a note whose remaining duration crosses a
    bar boundary is SPLIT into one fragment per bar it spans (a tied long note),
    rather than truncated with the remainder silently dropped — real musical
    content that a single-bar clip would have lost. Always returns exactly
    n_bars lists, some possibly empty."""
    low, high = register
    bars: List[List[NoteEvent]] = [[] for _ in range(n_bars)]
    plan_end = plan_start + n_bars * BEATS_PER_BAR
    cursor = plan_start
    for note in notes:
        if cursor >= plan_end:
            break
        pitch = note["pitch"]
        keep = pitch != REST_PITCH and low <= pitch <= high
        velocity = max(1, min(127, round(DEFAULT_VELOCITY * note.get("velocity_scale", 1.0))))
        remaining = note["duration_beats"]
        while remaining > 1e-9 and cursor < plan_end:
            bar_idx = int((cursor - plan_start) // BEATS_PER_BAR)
            bar_boundary = plan_start + (bar_idx + 1) * BEATS_PER_BAR
            chunk = min(remaining, bar_boundary - cursor)
            if keep:
                bars[bar_idx].append(
                    NoteEvent(
                        voice_id="",  # stamped by Session once the voice is known
                        pitch=pitch,
                        velocity=velocity,
                        start_beat=cursor,
                        duration_beats=chunk,
                    )
                )
            cursor += chunk
            remaining -= chunk
    return bars


def _pick_achievable_motif(counter: Counter) -> Optional[tuple]:
    """RehearsalMemory.recall_motifs() returns a Counter pooling 2-, 3-, and
    4-interval motifs together, weighted by quality (Phase 12). Picking simply
    the single most-common entry (any length) risks targeting a motif
    _apply_motif_bias (ensemble/wolfson/phrase_generator.py) can essentially
    never fire for: that function only biases the next token once the
    immediately preceding (motif_len - 1) generated intervals already match by
    chance -- a 2-interval motif needs just 1 prior interval to coincide, a
    4-interval motif needs 3 in a row, dramatically rarer. Preferring the
    most-common motif from the SHORTEST non-empty length bucket makes actual
    adherence achievable, not just theoretically fed in -- grounded directly in
    _apply_motif_bias's own prefix-matching logic, not guessed. None if the
    counter is empty."""
    for length in (2, 3, 4):
        candidates = {motif: weight for motif, weight in counter.items() if len(motif) == length}
        if candidates:
            return max(candidates, key=candidates.get)
    return None


def sax_generator(
    register: Tuple[int, int],
    target_voice_id: str,
    lookback_bars: int = 2,
    plan_bars: int = DEFAULT_PLAN_BARS,
    memory: Optional[RehearsalMemory] = None,
    n_candidates: int = 1,
    motif_recall_candidates: Optional[int] = None,
    credit_resolved_tension: bool = False,
    corpus: Optional[CorpusMotifs] = None,
    model_path: Optional[str] = None,
    seed: Optional[int] = None,
    own_voice_id: Optional[str] = None,
) -> Generator:
    """Build a generator that responds to target_voice_id's recent notes with a
    real LSTM-generated phrase (ensemble/wolfson/), planned plan_bars bars ahead
    (chord-hold permitting — see _bars_until_chord_change) and dispensed one bar
    per call from an internal buffer.

    memory, if given, is consulted and updated every time a new plan chunk is
    built: the most-common ACHIEVABLE motif recalled from it (if any —
    _pick_achievable_motif above, preferring the shortest recalled motif, not
    simply the single most-common one of any length) becomes that chunk's
    motif_targets, and the chunk's own generated notes are then stored into it —
    so passing the *same* RehearsalMemory into a later Session/sax_generator call
    lets that later run draw on this one's material (DESIGN.md §12, Phase 11).
    Passing no memory (the default) is exactly Phase 10's behaviour, unchanged.

    n_candidates, if greater than 1, generates that many candidates per chunk
    (identical arguments each call) and keeps the one scoring highest by
    (-dissonance, motif_adherence, musicality_score.overall), lexicographically
    — DESIGN.md §13's "chess" search-and-evaluate idea, Phase 14, extended in
    Phase 17 (motif_adherence) and Phase 18 (dissonance, checked FIRST). David's
    own framing after hearing real output: "what's bad matters a lot" — a
    candidate with one semitone-clash note (critic.py's dissonance — the
    "minor 9th" relationship, judged the worst case in ordinary melodic
    playing) is never preferred over a cleaner one just because it scored
    better on other, unrelated dimensions blended into `overall`; dissonance
    is a gate ahead of the positive-quality tie-breakers, not one more
    positively-weighted ingredient diluted into the blend. When motif_targets
    is empty and no candidate has any clash, this is provably identical to
    Phase 14's overall-only comparison (both leading terms are 0/tied for
    every candidate). Defaults to 1: exactly one generate() call per chunk,
    exactly today's behaviour, unchanged.

    The dissonance gate can be toggled live (Phase 20), the same director-
    gesture pattern as toggle_singability: two separate rests in a row
    (gesture/vocabulary.py's ("R","R") -> Gesture("toggle_dissonance_avoidance"))
    flips it off/on, checked every bar so a mid-chunk gesture isn't missed.
    dissonance() is still computed and logged every chunk-build regardless —
    only whether it counts in the selection key is gated. A real, honest
    reason a director might want it off: plenty of legitimate jazz vocabulary
    (4ths, tritones, deliberately "outside" playing) is technically dissonant
    by this module's own definition; the gate defaults ON (today's Phase 18/19
    behaviour, unchanged) because David asked specifically about ordinary
    melodic/singable soloing, not because avoidance is always correct.

    What counts as tolerated is itself context-aware (Phase 21):
    _functional_tonic_scale checks whether the current bar is the ii, V, or I
    of a recognised major ii-V-I, and if so unions the target I chord's own
    dissonance_scale in via dissonance()'s extra_tolerated parameter, the
    "simplify to the tonic" technique real players use. Computed once per
    chunk (doesn't depend on candidate notes), same timing as motif_targets.
    A vi-ii-V-I extension and minor-tonic ii-V-i variant are real, deliberate
    scope-cuts, not attempted here.

    corpus, if given (Phase 29), is a CorpusMotifs built from wjd_corpus.py's
    WJD motif-frequency cache. It's folded into the selection key ONLY for a
    chunk that's already bias-distorted away from the model's own natural
    distribution — a non-empty motif_targets (Phase 17) or song.modal (Phase
    27's modal_strength) — never for an ordinary chunk. Reasoning (see
    ensemble/critic.py's corpus_familiarity docstring and DESIGN.md's Phase
    29 paragraph for the full version): the LSTM was itself trained on WJD,
    so on an ordinary draw its output is already implicitly corpus-
    consistent — scoring corpus_familiarity there would mostly just
    re-reward what the model already learned, and risks systematically
    favouring "sounds like average WJD" over the deliberate boldness Phases
    17-24 pushed generation TOWARD. Where a bias nudge (motif_targets/
    modal_strength) has already pulled sampling off that natural
    distribution, though, "does this still look like real jazz vocabulary"
    is a genuinely different, useful question. Defaults to None, reproducing
    every existing caller's behaviour exactly — no CorpusMotifs, no change.

    This closure also tracks its own real pitch range explored so far this
    performance (Phase 32) — updated after every chunk's winner is chosen,
    fed into the NEXT chunk's musicality_score as register_usage's
    prior_range (see ensemble/critic.py's own docstring). Fixes a real,
    checked-not-assumed problem: register_usage previously only ever saw one
    chunk's own few notes, and Phase 30's critic baseline found that scored
    WJD's real, wide natural range LOWER than combo's narrower SAX_REGISTER
    — backwards, since a short excerpt can't reflect a whole solo's range.
    Widening what a candidate is judged against (this run's own explored
    range, not just this chunk) rewards a genuine excursion beyond what's
    already been played, not just a wide span within one candidate.

    motif_recall_candidates, if given, overrides n_candidates specifically for
    a chunk that has a non-empty motif_targets — more search shots make it far
    more likely at least one candidate actually lands the motif, justified by
    Phase 14's own measurement (~164ms for 20 candidates over a full 4-bar
    chunk on CPU). Checked empirically, not assumed to be rare: within-run
    persistence (Phase 11) means recall_motifs() returns something as soon as
    ANY earlier chunk in the same run has stored 2+ real notes — true for
    nearly every chunk after the first, not just an occasional cross-loop
    moment — so this applies broadly within a run, not to a rare one-off. The
    absolute cost stays fine for a real performance either way (Phase 14's
    per-chunk number, paid once per chunk during machine_speed generation,
    before playback starts), it's simply not the narrow "rare chunk" case
    first assumed. Defaults to None, reproducing n_candidates for every chunk
    exactly as before this parameter existed.

    The PhraseGenerator (and its ~3.5MB model weights) is constructed once here,
    not per bar or per plan — reused across the whole session for efficiency.
    The plan buffer refills only when exhausted: each refill is one continuous
    generate() call (hidden state and arc_position sweeping across the whole
    planned span, not resetting every bar — see module docstring), re-priming
    with hidden=None only at the start of each new chunk. Director intensity is
    captured once per chunk, at the moment it's built — a deliberate
    simplification, not forced by anything structural (unlike the chord case,
    where a plan literally cannot represent more than one chord per call): an
    intensity swing mid-chunk doesn't trigger replanning.

    seed, if given, seeds TWO separate global RNGs at construction time, not a
    local generator object like drums.py/comping.py use — confirmed necessary
    by testing, not assumed: torch.manual_seed for the model's own sampling
    (torch.multinomial), and Python's stdlib random.seed for rest injection
    (ensemble/wolfson/phrase_generator.py's _inject_rests uses the plain
    `random` module, entirely independent of torch's RNG — seeding only torch
    left rest placement, and therefore note count and timing, nondeterministic).
    Both are real, documented inconsistencies with the rest of this codebase's
    determinism pattern, not a clean port of it. Harmless today since nothing
    else in combo uses torch or reseeds Python's global random module, but
    noted honestly rather than hidden.

    own_voice_id, if given (Phase 37), must be the same string the caller will
    ALSO use for Voice(id=own_voice_id, generator=this_closure) -- the closure
    has no other way to learn its own future id, since Session only stamps
    voice_id onto OUTPUT events, after generation. When given, each chunk's
    seed_phrase becomes target_voice_id's recent notes followed by THIS
    voice's own recent notes (_build_combined_seed_phrase above) -- prompted
    directly by checking why Wolfson's own self-play sounded more alive:
    "the system feeds its own sax output back as input... the sax continuously
    responds to itself" (main.py's own docstring). combo's existing snapshot
    architecture already supports this with zero Session/Timeline changes
    (verified directly: pointing target_voice_id at a voice's own id already
    works today) -- own_voice_id is purely about ADDING self-history
    alongside the target voice, not replacing it, since a full swap would
    lose real ensemble listening a merge doesn't need to give up. Default
    None reproduces exactly today's bass-only seeding for every existing
    caller. A real, honest side effect: call_response_relatedness
    (ensemble/critic.py) reads the whole seed_phrase, so with own_voice_id
    active it partly measures relatedness to THIS voice's own recent playing
    too, not purely to target_voice_id -- not a bug, the natural way to
    observe this mechanism's effect, but worth knowing going in."""
    if seed is not None:
        import torch

        torch.manual_seed(seed)
        random.seed(seed)
    phrase_gen = PhraseGenerator(instrument="sax", model_path=model_path)
    plan: deque = deque()
    critic_weights = dict(DEFAULT_WEIGHTS)
    dissonance_mode = {"enabled": True}
    own_pitch_range = {"low": None, "high": None}  # Phase 32 -- this voice's own real pitch
                                                      # bounds explored so far THIS performance,
                                                      # fed into register_usage as prior_range
    own_pitch_weighted = {"sum": 0.0, "beats": 0.0}  # Phase 36 -- duration-weighted pitch
                                                        # accumulator, fed into register_balance
                                                        # as prior_mean_beats
    chunks_dispensed = {"count": 0}  # Phase 39 -- how many plan chunks this voice has
                                       # built so far THIS performance; gates the
                                       # phrase-boundary rest below (no rest before
                                       # the very first chunk -- nothing to separate
                                       # it from)

    def generate(song, bar_index: int, timeline: Timeline, director_signal) -> List[NoteEvent]:
        if director_signal.gesture is not None and director_signal.gesture.name == "toggle_singability":
            critic_weights["singability"] = 0.0 if critic_weights["singability"] else DEFAULT_WEIGHTS["singability"]
        if director_signal.gesture is not None and director_signal.gesture.name == "toggle_dissonance_avoidance":
            dissonance_mode["enabled"] = not dissonance_mode["enabled"]

        bar_start = bar_index * BEATS_PER_BAR

        if not plan:
            span_bars = _bars_until_chord_change(song, bar_start, plan_bars)
            since_beat = max(0, bar_index - lookback_bars) * BEATS_PER_BAR
            seed_phrase = _build_combined_seed_phrase(timeline, target_voice_id, own_voice_id, since_beat, bar_start)
            chord_idx = chord_to_wolfson_index(song.chord_at(bar_start))
            chord_quality = chord_idx % N_QUALITIES
            functional_scale = _functional_tonic_scale(song, bar_start)

            motif_targets = []
            if memory is not None:
                picked = _pick_achievable_motif(memory.recall_motifs(chord_quality=chord_quality))
                if picked is not None:
                    motif_targets = [picked]

            # Phase 29: corpus_familiarity is only meaningful for a chunk
            # already pushed off the model's own natural distribution by one
            # of our own bias nudges -- see module docstring's `corpus`
            # paragraph for why applying it more broadly would be
            # counterproductive, not just unnecessary.
            is_bias_distorted = bool(motif_targets) or song.modal

            candidates_this_chunk = (motif_recall_candidates or n_candidates) if motif_targets else n_candidates

            # Phase 32: this voice's own real pitch range explored so far this
            # performance, fed into register_usage as prior_range -- None
            # (the pre-Phase-32 behaviour) until the first chunk's winner is
            # chosen below.
            prior_range = (
                (own_pitch_range["low"], own_pitch_range["high"]) if own_pitch_range["low"] is not None else None
            )
            # Phase 36: mirrors prior_range immediately above, for
            # register_balance instead of register_usage -- None until the
            # first chunk's winner is chosen below.
            prior_mean_beats = (
                (own_pitch_weighted["sum"], own_pitch_weighted["beats"]) if own_pitch_weighted["beats"] > 0 else None
            )

            best_notes = None
            best_score = None
            best_key = None
            best_dissonance = None
            candidate_scores = []
            for _ in range(candidates_this_chunk):
                candidate_notes = phrase_gen.generate(
                    seed_phrase,
                    chord_idx=chord_idx,
                    max_phrase_beats=span_bars * BEATS_PER_BAR,
                    rhythmic_density=director_signal.intensity,
                    motif_targets=motif_targets,
                    motif_strength=DEFAULT_MOTIF_STRENGTH if motif_targets else 0.0,
                    modal_strength=MODAL_STRENGTH_WHEN_ACTIVE if song.modal else 0.0,
                )
                candidate_score = musicality_score(
                    candidate_notes, chord_idx, seed_phrase, register, weights=critic_weights,
                    extra_tolerated=functional_scale, credit_resolved_tension=credit_resolved_tension,
                    modal=song.modal, prior_range=prior_range, prior_mean_beats=prior_mean_beats,
                )
                candidate_scores.append(candidate_score.overall)
                # (-dissonance, adherence, corpus_score, overall) lexicographic
                # key: badness checked FIRST (negated so max() prefers the
                # LOWEST dissonance), then adherence to a recalled motif, then
                # corpus familiarity (Phase 29, only non-zero for a
                # bias-distorted chunk -- see is_bias_distorted above), then
                # general quality as the final tie-break -- see module
                # docstring for why dissonance isn't just one more
                # positively-weighted ingredient in `overall`. When
                # motif_targets is empty, no candidate clashes, and the chunk
                # isn't modal, the first three terms are tied at
                # (0.0, 0.0, 0.0) for every candidate, so this is provably
                # identical to comparing candidate_score.overall alone (Phase
                # 14's original behaviour). d is still computed and logged
                # even when dissonance_mode is disabled (Phase 20) -- cheap, and
                # keeps dissonance_log meaningful regardless of what's currently
                # driving selection -- only whether it counts in the key is gated.
                d = dissonance(
                    candidate_notes,
                    chord_idx,
                    extra_tolerated=functional_scale,
                    credit_resolved_tension=credit_resolved_tension,
                )
                corpus_score = (
                    corpus_familiarity(candidate_notes, chord_quality, corpus)
                    if corpus is not None and is_bias_distorted
                    else 0.0
                )
                key = (
                    -d if dissonance_mode["enabled"] else 0.0,
                    motif_adherence(candidate_notes, motif_targets),
                    corpus_score,
                    candidate_score.overall,
                )
                if best_key is None or key > best_key:
                    best_notes, best_score, best_key, best_dissonance = candidate_notes, candidate_score, key, d
            notes = best_notes
            generate.last_candidate_scores = candidate_scores
            generate.dissonance_log.append(best_dissonance)
            generate.motif_adherence_log.append(best_key[1])
            generate.winning_score_log.append(best_score)

            # Phase 39: a deliberate rest between chunks -- "speaking in
            # sentences". Wolfson's own self-play generates one phrase per
            # call, plays it out, THEN feeds it back after a short gap;
            # silence between phrases falls out of that turn-taking loop for
            # free. combo's plan-chunk architecture has no structural
            # equivalent -- _inject_rests (phrase_generator.py) is the only
            # source of any rest, and its own bell-curve probability is
            # mathematically zero at both ends of a generate() call, so a
            # chunk boundary is exactly where a rest is LEAST likely to occur
            # on its own. Reuses _inject_rests' own rest-sentinel shape so
            # _split_phrase_into_bars (already REST_PITCH-aware) needs no
            # changes. Skipped before the very first chunk of the performance
            # -- there's no prior phrase to separate this one from.
            if chunks_dispensed["count"] > 0:
                notes = [
                    {
                        "pitch": REST_PITCH,
                        "duration_beats": random.choice(PHRASE_BOUNDARY_REST_BEATS),
                        "velocity_scale": 1.0,
                    }
                ] + notes
            chunks_dispensed["count"] += 1

            winner_real_in_register = [
                n for n in notes if n["pitch"] != REST_PITCH and register[0] <= n["pitch"] <= register[1]
            ]
            if winner_real_in_register:
                winner_real_pitches = [n["pitch"] for n in winner_real_in_register]
                lo, hi = min(winner_real_pitches), max(winner_real_pitches)
                own_pitch_range["low"] = lo if own_pitch_range["low"] is None else min(own_pitch_range["low"], lo)
                own_pitch_range["high"] = hi if own_pitch_range["high"] is None else max(own_pitch_range["high"], hi)
                own_pitch_weighted["sum"] += sum(n["pitch"] * n["duration_beats"] for n in winner_real_in_register)
                own_pitch_weighted["beats"] += sum(n["duration_beats"] for n in winner_real_in_register)

            if memory is not None:
                memory.store(notes, score=best_score.overall, chord_quality=chord_quality)
            plan.extend(_split_phrase_into_bars(notes, bar_start, span_bars, register))

        return plan.popleft()

    generate.critic_weights = critic_weights  # exposed for testing -- see module docstring
    generate.last_candidate_scores = []  # populated on the first chunk-build; exposed for testing
    generate.motif_adherence_log = []  # one entry per chunk-build, the WINNING candidate's motif_adherence
    generate.dissonance_log = []  # one entry per chunk-build, the WINNING candidate's actual dissonance
                                    # (Phase 18) -- always the real value, even when dissonance_mode is
                                    # disabled and isn't currently driving selection (Phase 20).
    generate.dissonance_mode = dissonance_mode  # exposed for testing -- same "expose the mutable
                                                  # dict directly" convention as critic_weights above
    generate.winning_score_log = []  # one entry per chunk-build (Phase 30), the WINNING candidate's
                                       # full MusicalityScore -- not last_candidate_scores (every
                                       # candidate's .overall, overwritten each chunk) or
                                       # motif_adherence_log/dissonance_log (one sub-score each) --
                                       # this is the complete score of what was actually dispensed,
                                       # for critic_baseline.py's real-vs-WJD comparison.
    generate.own_pitch_range = own_pitch_range  # exposed for testing (Phase 32) -- same "expose the
                                                  # mutable dict directly" convention as dissonance_mode
    generate.own_pitch_weighted = own_pitch_weighted  # exposed for testing (Phase 36) -- same convention
    generate.chunks_dispensed = chunks_dispensed  # exposed for testing (Phase 39) -- same convention
    return generate
