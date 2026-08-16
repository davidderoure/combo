"""A voice in the ensemble — see DESIGN.md §2.

This is the thin MVP slice of §2's voice/role concept: an id, an instrument profile,
and a source (human or AI). Role assignment, doubling, and human/AI symmetry of
listening behaviour all build on top of this once there's something to test them
against (Phase 4+ in the build plan) — deliberately not included here.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

from .director import DirectorSignal
from .timeline import NoteEvent, Timeline

if TYPE_CHECKING:
    from song import Song

# (song, bar_index, timeline-of-prior-bars, aggregated-director-signal) -> notes for
# that bar. The timeline lets a voice listen to what other voices have already played
# (DESIGN.md §5) — see ensemble/session.py for exactly what "prior" means and why. The
# director signal is the same for every voice in a given bar (DESIGN.md §11); with no
# Director configured it's the neutral default, so existing generators are unaffected.
Generator = Callable[["Song", int, Timeline, DirectorSignal], List[NoteEvent]]


@dataclass
class Voice:
    id: str
    instrument: str
    register: Tuple[int, int]  # (low, high) MIDI note range
    source: str  # "human" or "ai"
    generator: Optional[Generator] = None

    def __post_init__(self) -> None:
        if self.source not in ("human", "ai"):
            raise ValueError(f"source must be 'human' or 'ai', got {self.source!r}")
        if self.source == "ai" and self.generator is None:
            raise ValueError(f"AI-sourced voice {self.id!r} needs a generator")
