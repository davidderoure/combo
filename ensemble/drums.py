"""A rule-based drum voice — see DESIGN.md §7.

No trained model here (WJazzD has no drum data), and this is deliberately not what
DESIGN.md §7 literally says either. §7 asks for "brushes": on a real kit that's a
continuous sweeping texture on the snare, not a series of separate note onsets — and
NoteEvent (a pitch, a start time, a duration) simply can't represent that, not even
approximately. What's here instead is brushes-in-spirit: a discrete pattern closer to
what a stick-based comping pattern would actually produce (hi-hat time-keeping, a
walking ride, syncopated snare accents). Genuine continuous brush texture would need a
different kind of event than NoteEvent — a real, structural gap, not a "doesn't sound
good yet" one like ensemble/generators.py's chord_tone_generator.

For anyone who (like David) doesn't already have this vocabulary: the "hi-hat chick"
on beats 2 and 4 is about as fundamental as jazz time-keeping gets — closing the hi-hat
pedal on the off-beats is most of what makes a groove feel like it's swinging at all,
independent of whatever else is happening on top of it. That's why it's the one thing
every density tier below keeps, even the sparsest.
"""

import random
from typing import List, Optional

from .timeline import BEATS_PER_BAR, NoteEvent
from .voice import Generator

# General MIDI percussion map (MIDI channel 10): a fixed note-number-to-drum-sound
# assignment, not a pitch scale — each number is a specific instrument/sound.
ACOUSTIC_SNARE = 38
CLOSED_HI_HAT = 42
RIDE_CYMBAL_1 = 51

HIT_DURATION_BEATS = 0.1  # percussive; duration here is nominal, not a sustained note
TIMING_JITTER_BEATS = 0.02
VELOCITY_JITTER = 10

SPARSE, MEDIUM, BUSY = "sparse", "medium", "busy"


def _density_for_bar(song, bar_index: int) -> str:
    """Sparser under the head/out, busier toward the end of a section — standing in
    for "approaching a peak" since there's no ArcController yet to ask directly."""
    beat = bar_index * BEATS_PER_BAR
    section, chorus_index = song.section_at(beat)
    name = section.name.strip().lower()
    if "head" in name or "out" in name:
        return SPARSE
    if section.repeats is not None and chorus_index >= section.repeats - 1:
        return BUSY
    return MEDIUM


def _hit(pitch: int, bar_index: int, beat_offset: float, base_velocity: int, rng: random.Random) -> NoteEvent:
    start_beat = bar_index * BEATS_PER_BAR + beat_offset + rng.uniform(
        -TIMING_JITTER_BEATS, TIMING_JITTER_BEATS
    )
    velocity = max(1, min(127, base_velocity + rng.randint(-VELOCITY_JITTER, VELOCITY_JITTER)))
    return NoteEvent(
        voice_id="",  # stamped by Session, same convention as chord_tone_generator
        pitch=pitch,
        velocity=velocity,
        start_beat=start_beat,
        duration_beats=HIT_DURATION_BEATS,
    )


def _sparse_bar(bar_index: int, rng: random.Random) -> List[NoteEvent]:
    return [_hit(CLOSED_HI_HAT, bar_index, offset, 70, rng) for offset in (1.0, 3.0)]


def _medium_bar(bar_index: int, rng: random.Random) -> List[NoteEvent]:
    events = _sparse_bar(bar_index, rng)
    # A plain walking quarter-note ride. Real swing ride is a triplet-based
    # "spang-a-lang" rhythm, not straight quarters — deferred to a real swing-timing
    # pass; this is the honest simplification for now.
    events += [_hit(RIDE_CYMBAL_1, bar_index, offset, 75, rng) for offset in (0.0, 1.0, 2.0, 3.0)]
    return events


def _busy_bar(bar_index: int, rng: random.Random) -> List[NoteEvent]:
    events = _medium_bar(bar_index, rng)
    offsets = rng.sample([0.5, 1.5, 2.5, 3.5], rng.choice([1, 2]))
    events += [_hit(ACOUSTIC_SNARE, bar_index, offset, 60, rng) for offset in offsets]
    return events


def drum_generator(seed: Optional[int] = None) -> Generator:
    """Build a rule-based drum generator. Seed for reproducible humanisation
    (tests); omit for natural variation."""
    rng = random.Random(seed)

    def generate(song, bar_index: int, timeline) -> List[NoteEvent]:
        # `timeline` (prior bars, other voices) isn't used here — density is driven
        # purely by section/form, not by listening to another voice.
        density = _density_for_bar(song, bar_index)
        if density == SPARSE:
            return _sparse_bar(bar_index, rng)
        if density == MEDIUM:
            return _medium_bar(bar_index, rng)
        return _busy_bar(bar_index, rng)

    return generate
