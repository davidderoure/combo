"""A rule-based comping voice — the concrete accompanist for DESIGN.md §5.

Structured the same way ensemble/drums.py was: density-driven tiers, humanised via a
seeded RNG. The difference is where the density signal comes from — drums reads
section/form state (ensemble/drums.py's own concern); this reads another voice's
recent output via ensemble/listening.py's `density`.

Only the accompanist-listens-to-soloist, complementary-only case is built here. Two
things DESIGN.md §5 names are explicitly not attempted: "occasional mirrored builds
near arc peaks" (needs a peak/arc signal — no ArcController exists yet) and "the
same-register role-split default applied laterally between two accompanists" (needs
role assignment — §2's role machinery isn't built yet either).
"""

import random
from typing import List, Optional, Tuple

from .generators import place_in_register
from .listening import density
from .timeline import BEATS_PER_BAR, NoteEvent, Timeline
from .voice import Generator

STAB_DURATION_BEATS = 1.5
TIMING_JITTER_BEATS = 0.02
DEFAULT_VELOCITY = 65
VELOCITY_JITTER = 8

# notes/beat thresholds on the *target* voice's recent density
BUSY_THRESHOLD = 1.5  # at/above this, the soloist is busy -> duck (leave space)
SPARSE_THRESHOLD = 0.25  # at/below this, the soloist has left space -> fill


def _stab(chord, register: Tuple[int, int], bar_index: int, beat_offset: float, rng: random.Random) -> List[NoteEvent]:
    # Root + fifth, same interval choice as ensemble/generators.py's
    # chord_tone_generator, reused rather than reinvented: there's no confident way
    # to pick 3rds/7ths correctly across every chord quality string without being
    # able to hear the result, so this stays with the interval that's unambiguous
    # regardless of major/minor/dominant/etc.
    root = place_in_register(chord.root, register)
    fifth = place_in_register((chord.root + 7) % 12, register)
    events = []
    for pitch in (root, fifth):
        start_beat = bar_index * BEATS_PER_BAR + beat_offset + rng.uniform(
            -TIMING_JITTER_BEATS, TIMING_JITTER_BEATS
        )
        velocity = max(1, min(127, DEFAULT_VELOCITY + rng.randint(-VELOCITY_JITTER, VELOCITY_JITTER)))
        events.append(
            NoteEvent(voice_id="", pitch=pitch, velocity=velocity, start_beat=start_beat, duration_beats=STAB_DURATION_BEATS)
        )
    return events


def comping_generator(
    register: Tuple[int, int],
    target_voice_id: str,
    lookback_bars: int = 2,
    seed: Optional[int] = None,
) -> Generator:
    """Build a generator that listens to target_voice_id's density over the previous
    lookback_bars bars and responds complementarily. At bar_index < lookback_bars the
    window is empty -> density 0.0 -> fills by default (a harmless, expected edge
    case, not a bug: nothing's been heard yet, so there's nothing to duck for)."""
    rng = random.Random(seed)

    def generate(song, bar_index: int, timeline: Timeline) -> List[NoteEvent]:
        since_beat = max(0, bar_index - lookback_bars) * BEATS_PER_BAR
        until_beat = bar_index * BEATS_PER_BAR
        target_density = density(timeline, target_voice_id, since_beat, until_beat)
        chord = song.chord_at(bar_index * BEATS_PER_BAR)

        if target_density >= BUSY_THRESHOLD:
            return []  # duck
        if target_density <= SPARSE_THRESHOLD:
            return _stab(chord, register, bar_index, 0.0, rng) + _stab(chord, register, bar_index, 2.0, rng)
        return _stab(chord, register, bar_index, 0.0, rng)  # moderate

    return generate
