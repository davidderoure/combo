"""Real generation for the sax voice — DESIGN.md §12, Phase 8 of the build plan.

The first voice to move off ensemble/generators.py's chord_tone_generator stub.
Wraps ensemble/wolfson/'s ported PhraseGenerator (an LSTM adapted from David's
earlier system, Wolfson) behind combo's ordinary Generator interface. Everything
here is combo-authored glue, not ported — structured the same context-driven shape
as ensemble/drums.py and ensemble/comping.py, though determinism works differently
here (see sax_generator's docstring: a global torch seed, not a local RNG), with
three real integration seams:

  1. chord_to_wolfson_index — combo's Chord -> Wolfson's chord vocabulary.
  2. _build_seed_phrase — a target voice's recent notes -> Wolfson's seed_phrase,
     mirroring comping_generator's lookback-window pattern exactly.
  3. _place_phrase_in_bar — clips PhraseGenerator.generate()'s output onto the
     current bar. Necessary because max_phrase_beats does NOT bound the returned
     phrase's span (verified empirically: rest injection runs after the beat cap
     and isn't counted against it) — see ensemble/wolfson/phrase_generator.py's
     generate() docstring.

Explicitly deferred (see DESIGN.md §12 for the full list): all ~12 of
PhraseGenerator.generate()'s bias-layer knobs (contour, energy arc, motif,
register contrast, etc.) are left at their defaults — nothing in combo supplies
them yet, the same "deliberately dumb" spirit as chord_tone_generator itself. No
hidden-state continuity across bar-to-bar calls (each bar re-primes from that
bar's own seed). No director-signal modulation of generation (DirectorSignal is
accepted and ignored — standard "extend on integration"; DirectorSignal.intensity
-> rhythmic_density is a natural follow-on, not built now). `register` is a
backstop that drops out-of-range notes, not a real voicing control like it is in
chord_tone_generator/comping_generator — Wolfson's trained pitch vocabulary is
hard-clipped to MIDI 44-93 by the model itself.
"""

from typing import List, Optional, Tuple

from song.chord import Chord

from .timeline import BEATS_PER_BAR, NoteEvent, Timeline
from .voice import Generator
from .wolfson.chords import N_QUALITIES, QUAL_DIM, QUAL_DOM, QUAL_MAJOR, QUAL_MINOR
from .wolfson.phrase_generator import REST_PITCH, PhraseGenerator

DEFAULT_VELOCITY = 75

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


def _place_phrase_in_bar(
    notes: list, bar_start: float, bar_end: float, register: Tuple[int, int]
) -> List[NoteEvent]:
    """Clip PhraseGenerator.generate()'s output onto [bar_start, bar_end). Required
    because max_phrase_beats doesn't bound the actual returned span (see module
    docstring). Walks notes with a beat cursor: stops once the cursor reaches
    bar_end, skips REST_PITCH sentinels as silent gaps (never emitted as events),
    clips any note's duration so it can't cross bar_end, and drops notes outside
    `register` — a backstop, not a real voicing control (see module docstring)."""
    low, high = register
    events: List[NoteEvent] = []
    cursor = bar_start
    for note in notes:
        if cursor >= bar_end:
            break
        duration = min(note["duration_beats"], bar_end - cursor)
        if note["pitch"] != REST_PITCH and low <= note["pitch"] <= high:
            velocity = max(1, min(127, round(DEFAULT_VELOCITY * note.get("velocity_scale", 1.0))))
            events.append(
                NoteEvent(
                    voice_id="",  # stamped by Session once the voice is known
                    pitch=note["pitch"],
                    velocity=velocity,
                    start_beat=cursor,
                    duration_beats=duration,
                )
            )
        cursor += duration
    return events


def sax_generator(
    register: Tuple[int, int],
    target_voice_id: str,
    lookback_bars: int = 2,
    model_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> Generator:
    """Build a generator that responds to target_voice_id's recent notes with a
    real LSTM-generated phrase (ensemble/wolfson/), clipped to the current bar.

    The PhraseGenerator (and its ~3.5MB model weights) is constructed once here,
    not per bar — reused across the whole session for efficiency. Each bar still
    re-primes with hidden=None from that bar's own seed_phrase; there's no
    persistent hidden-state continuity across bars (a real, named limitation vs.
    live Wolfson, not a bug — see module docstring).

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

    def generate(song, bar_index: int, timeline: Timeline, director_signal) -> List[NoteEvent]:
        # director_signal isn't used here yet — see module docstring.
        bar_start = bar_index * BEATS_PER_BAR
        bar_end = bar_start + BEATS_PER_BAR
        since_beat = max(0, bar_index - lookback_bars) * BEATS_PER_BAR
        seed_phrase = _build_seed_phrase(timeline, target_voice_id, since_beat, bar_start)

        chord_idx = chord_to_wolfson_index(song.chord_at(bar_start))
        notes = phrase_gen.generate(seed_phrase, chord_idx=chord_idx, max_phrase_beats=BEATS_PER_BAR)
        return _place_phrase_in_bar(notes, bar_start, bar_end, register)

    return generate
