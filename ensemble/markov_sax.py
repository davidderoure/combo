"""A Markov-chain sax generator sharing the LSTM sax voice's real critic and
search-and-select architecture -- Phase 35, DESIGN.md §13.

Prompted by the WJD-in/WJD-out reflection: ensemble/critic.py was built
(Phase 12) as a pure function over already-generated note dicts, never on
anything LSTM-specific, so the n_candidates search-and-select machinery
(ensemble/sax.py) isn't actually tied to the LSTM. This module proves that
by plugging a genuinely different, much simpler local generator into the
exact same critic and selection key -- and by doing so, gives a real
diagnostic: does combo's elevated `repetition` relative to WJD (Phase 34,
found to be temperature-insensitive) come from something neural-sampling-
specific, or is it a more general property of any short-context local
generator? Markov chains are literally n-gram models, with their own
long-documented repetition tendencies for different reasons than an LSTM's.

Deliberately REUSES, not reimplements, everything generation-mechanism-
agnostic already in ensemble/sax.py -- chord_to_wolfson_index,
_build_seed_phrase, _bars_until_chord_change, _split_phrase_into_bars are
all plain functions over note dicts, none LSTM-specific (checked directly)
-- and ensemble/critic.py's musicality_score/dissonance/motif_adherence for
the SAME selection key sax_generator uses: (-dissonance, motif_adherence,
overall). motif_adherence is always 0.0 here (this MVP has no
RehearsalMemory, see below), so the key degenerates to (-dissonance,
overall) -- still a real, meaningful two-term search, not vacuous.

Explicitly deferred, named rather than silently out of reach: RehearsalMemory/
corpus_familiarity/credit_resolved_tension/modal_strength support -- kept out
so this MVP's comparison against the LSTM stays focused on the generation
mechanism itself, not confounded by feature parity; higher-order chains
(order is a real parameter, not attempted beyond what markov_corpus.py was
built with); joint pitch+duration modeling (two independent chains here);
rest/breath modeling -- this generator never produces a REST_PITCH event, so
phrasing() will likely score its output harshly for the same reason it
initially did for raw WJD note data (Phase 30) -- a known, already-understood
measurement artifact, not a new musical problem; register-aware pitch-walk
biasing -- an unlucky run of intervals can drift the walk outside `register`
entirely, relying on _split_phrase_into_bars' existing backstop clipping to
drop what doesn't fit, the same "backstop, not a real control" pattern the
LSTM path already has for out-of-vocabulary pitches.
"""

import random
from collections import deque
from typing import List, Optional, Tuple

from .critic import dissonance, motif_adherence, musicality_score
from .markov_tables import MarkovTables
from .sax import _bars_until_chord_change, _build_seed_phrase, _split_phrase_into_bars, DEFAULT_PLAN_BARS, chord_to_wolfson_index
from .timeline import BEATS_PER_BAR, NoteEvent, Timeline
from .voice import Generator
from .wolfson.chords import N_QUALITIES
from .wolfson.encoding import dur_to_token, token_to_dur

MAX_GENERATED_NOTES = 200  # safety cap against a pathological all-zero-duration walk;
                             # mirrors PhraseGenerator.generate()'s own MAX_GENERATED_NOTES role


def _generate_markov_phrase(
    tables: MarkovTables, quality: int, seed_phrase: list, register: Tuple[int, int],
    max_phrase_beats: float, order: int, rng: random.Random,
) -> list:
    """Walks the Markov chain to build one candidate phrase. Pitch (via
    signed-interval sampling) and duration (via duration-token sampling) are
    INDEPENDENT chains (see module docstring) -- each maintains its own
    trailing context of the last `order` outcomes. Starts from the seed
    phrase's own trailing pitch/duration context when there's enough of it
    (real continuity with what the target voice just played); a context
    shorter than `order` (including no seed at all) is passed as None to
    MarkovTables, which falls back to the quality's marginal distribution --
    the same fallback used for a genuinely unseen context, not a special
    case. No seed at all starts from the middle of `register` (an explicit,
    named choice, not silently arbitrary). Stops once max_phrase_beats has
    accumulated, or MAX_GENERATED_NOTES is hit."""
    seed_pitches = [n["pitch"] for n in seed_phrase]
    recent_intervals = [seed_pitches[i] - seed_pitches[i - 1] for i in range(1, len(seed_pitches))]
    last_pitch = seed_pitches[-1] if seed_pitches else (register[0] + register[1]) // 2

    recent_dur_tokens = [dur_to_token(n["offset"] - n["onset"]) for n in seed_phrase]

    notes: list = []
    total_beats = 0.0
    for _ in range(MAX_GENERATED_NOTES):
        if total_beats >= max_phrase_beats:
            break
        pitch_context = tuple(recent_intervals[-order:]) if len(recent_intervals) >= order else None
        interval = tables.sample_next_interval(quality, pitch_context, rng)
        next_pitch = last_pitch + interval
        recent_intervals.append(interval)

        dur_context = tuple(recent_dur_tokens[-order:]) if len(recent_dur_tokens) >= order else None
        dur_token = tables.sample_next_duration_token(quality, dur_context, rng)
        duration_beats = token_to_dur(dur_token)
        recent_dur_tokens.append(dur_token)

        notes.append({"pitch": next_pitch, "duration_beats": duration_beats, "velocity_scale": 1.0})
        total_beats += duration_beats
        last_pitch = next_pitch

    return notes


def markov_sax_generator(
    register: Tuple[int, int],
    target_voice_id: str,
    tables: MarkovTables,
    lookback_bars: int = 2,
    plan_bars: int = DEFAULT_PLAN_BARS,
    order: int = 1,
    n_candidates: int = 1,
    seed: Optional[int] = None,
) -> Generator:
    """Build a generator that responds to target_voice_id's recent notes
    with a Markov-chain-generated phrase, planned plan_bars bars ahead
    (chord-hold permitting, via the SAME _bars_until_chord_change as
    sax_generator) and dispensed one bar per call from an internal buffer --
    structurally the same chunk-planning shape as sax_generator, reusing its
    chord-change/bar-splitting machinery directly rather than reimplementing
    it (see module docstring).

    n_candidates, if greater than 1, generates that many candidates per
    chunk and keeps the one scoring highest by (-dissonance,
    motif_adherence, overall) -- the identical selection key sax_generator
    uses, so results are directly comparable. motif_adherence is always 0.0
    here (no RehearsalMemory in this MVP), so ties break on dissonance then
    overall alone -- still real search, just over a smaller signal set.

    Tracks its own real pitch range explored so far this performance
    (Phase 32's register_usage prior_range mechanism) -- cheap, generation-
    mechanism-agnostic, included for a fairer comparison against the LSTM
    side, which has it too."""
    rng = random.Random(seed)
    plan: deque = deque()
    own_pitch_range = {"low": None, "high": None}

    def generate(song, bar_index: int, timeline: Timeline, director_signal) -> List[NoteEvent]:
        bar_start = bar_index * BEATS_PER_BAR

        if not plan:
            span_bars = _bars_until_chord_change(song, bar_start, plan_bars)
            since_beat = max(0, bar_index - lookback_bars) * BEATS_PER_BAR
            seed_phrase = _build_seed_phrase(timeline, target_voice_id, since_beat, bar_start)
            chord_idx = chord_to_wolfson_index(song.chord_at(bar_start))
            chord_quality = chord_idx % N_QUALITIES

            prior_range = (
                (own_pitch_range["low"], own_pitch_range["high"]) if own_pitch_range["low"] is not None else None
            )

            best_notes = None
            best_score = None
            best_key = None
            for _ in range(n_candidates):
                candidate_notes = _generate_markov_phrase(
                    tables, chord_quality, seed_phrase, register,
                    max_phrase_beats=span_bars * BEATS_PER_BAR, order=order, rng=rng,
                )
                candidate_score = musicality_score(
                    candidate_notes, chord_idx, seed_phrase, register, prior_range=prior_range,
                )
                d = dissonance(candidate_notes, chord_idx)
                key = (-d, motif_adherence(candidate_notes, []), candidate_score.overall)
                if best_key is None or key > best_key:
                    best_notes, best_score, best_key = candidate_notes, candidate_score, key
            notes = best_notes
            generate.winning_score_log.append(best_score)

            winner_real_pitches = [n["pitch"] for n in notes if register[0] <= n["pitch"] <= register[1]]
            if winner_real_pitches:
                lo, hi = min(winner_real_pitches), max(winner_real_pitches)
                own_pitch_range["low"] = lo if own_pitch_range["low"] is None else min(own_pitch_range["low"], lo)
                own_pitch_range["high"] = hi if own_pitch_range["high"] is None else max(own_pitch_range["high"], hi)

            plan.extend(_split_phrase_into_bars(notes, bar_start, span_bars, register))

        return plan.popleft()

    generate.own_pitch_range = own_pitch_range  # exposed for testing, same convention as sax_generator
    generate.winning_score_log = []  # one entry per chunk-build, the WINNING candidate's full
                                       # MusicalityScore -- same shape as sax_generator's own,
                                       # for critic_baseline.py's direct comparison
    return generate
