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
honestly-placeholder that is.

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

from collections import Counter, deque
from typing import List, Optional, Tuple

from song.chord import Chord

from .critic import DEFAULT_WEIGHTS, motif_adherence, musicality_score
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
    model_path: Optional[str] = None,
    seed: Optional[int] = None,
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
    (motif_adherence, musicality_score.overall), lexicographically — DESIGN.md
    §13's "chess" search-and-evaluate idea, Phase 14, extended in Phase 17 so a
    chunk with a real motif to aim for actually prefers a candidate that used
    it, not just the generically highest-scoring one. When motif_targets is
    empty, motif_adherence is 0.0 for every candidate, so this is provably
    identical to Phase 14's overall-only comparison. Defaults to 1: exactly one
    generate() call per chunk, exactly today's behaviour, unchanged.

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
    noted honestly rather than hidden."""
    if seed is not None:
        import random

        import torch

        torch.manual_seed(seed)
        random.seed(seed)
    phrase_gen = PhraseGenerator(instrument="sax", model_path=model_path)
    plan: deque = deque()
    critic_weights = dict(DEFAULT_WEIGHTS)

    def generate(song, bar_index: int, timeline: Timeline, director_signal) -> List[NoteEvent]:
        if director_signal.gesture is not None and director_signal.gesture.name == "toggle_singability":
            critic_weights["singability"] = 0.0 if critic_weights["singability"] else DEFAULT_WEIGHTS["singability"]

        bar_start = bar_index * BEATS_PER_BAR

        if not plan:
            span_bars = _bars_until_chord_change(song, bar_start, plan_bars)
            since_beat = max(0, bar_index - lookback_bars) * BEATS_PER_BAR
            seed_phrase = _build_seed_phrase(timeline, target_voice_id, since_beat, bar_start)
            chord_idx = chord_to_wolfson_index(song.chord_at(bar_start))

            motif_targets = []
            if memory is not None:
                picked = _pick_achievable_motif(memory.recall_motifs())
                if picked is not None:
                    motif_targets = [picked]

            candidates_this_chunk = (motif_recall_candidates or n_candidates) if motif_targets else n_candidates

            best_notes = None
            best_score = None
            best_key = None
            candidate_scores = []
            for _ in range(candidates_this_chunk):
                candidate_notes = phrase_gen.generate(
                    seed_phrase,
                    chord_idx=chord_idx,
                    max_phrase_beats=span_bars * BEATS_PER_BAR,
                    rhythmic_density=director_signal.intensity,
                    motif_targets=motif_targets,
                    motif_strength=DEFAULT_MOTIF_STRENGTH if motif_targets else 0.0,
                )
                candidate_score = musicality_score(candidate_notes, chord_idx, seed_phrase, weights=critic_weights)
                candidate_scores.append(candidate_score.overall)
                # (adherence, overall) lexicographic key: when motif_targets is
                # empty, adherence is 0.0 for every candidate, so this is
                # provably identical to comparing candidate_score.overall alone
                # (Phase 14's behaviour) -- see module docstring.
                key = (motif_adherence(candidate_notes, motif_targets), candidate_score.overall)
                if best_key is None or key > best_key:
                    best_notes, best_score, best_key = candidate_notes, candidate_score, key
            notes = best_notes
            generate.last_candidate_scores = candidate_scores
            generate.motif_adherence_log.append(best_key[0])

            if memory is not None:
                memory.store(notes, score=best_score.overall)
            plan.extend(_split_phrase_into_bars(notes, bar_start, span_bars, register))

        return plan.popleft()

    generate.critic_weights = critic_weights  # exposed for testing -- see module docstring
    generate.last_candidate_scores = []  # populated on the first chunk-build; exposed for testing
    generate.motif_adherence_log = []  # one entry per chunk-build, the WINNING candidate's motif_adherence
    return generate
