"""A song: one chord progression (cycled repeatedly) plus a form (ordered
named sections, each a number of cycles through the changes) and
tempo/feel. Combo's persistent, nameable "play this song" object — see
DESIGN.md §3.

Deliberately close to IRCAM ImproteK's "scenario" concept: a chord chart as
an explicit, reusable structural object, distinct from any particular
performance of it (see DESIGN.md §12).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .changes import Changes
from .form import Section


@dataclass
class Song:
    title: str
    changes: Changes
    form: List[Section]
    tempo_bpm: float = 120.0
    feel: str = "swing"
    key: Optional[int] = None  # pitch class, for reference/transposition only
    modal: bool = False  # Phase 27: chart-authored style choice (mirroring how a
                          # player reads the artist/date to decide triadic vs
                          # quartal/modal vocabulary -- see ensemble/critic.py's
                          # MODAL_LEAP_SEMITONES and ensemble/sax.py's
                          # MODAL_STRENGTH_WHEN_ACTIVE)

    @property
    def total_beats(self) -> Optional[float]:
        """None if the form contains an open-ended section (repeats=None) —
        the song's length isn't fixed in advance; bar counts are a nominal
        scaffold, not a hard endpoint (DESIGN.md §8)."""
        total = 0.0
        for section in self.form:
            if section.repeats is None:
                return None
            total += section.repeats * self.changes.total_beats
        return total

    def chord_at(self, beat: float):
        return self.changes.chord_at(beat)

    def section_at(self, beat: float) -> Tuple[Section, int]:
        """Returns (section, chorus_index): chorus_index counts full cycles
        through `changes` since the start of that section (0-based). Once
        past the last section's nominal length (or inside an open-ended
        section), chorus_index keeps counting rather than raising — the
        actual end is a live decision (the musical director), not fixed
        here."""
        if not self.form:
            raise ValueError("Song has no form")
        chorus_beats = self.changes.total_beats
        elapsed = 0.0
        for section in self.form:
            if section.repeats is None:
                return section, max(int((beat - elapsed) // chorus_beats), 0)
            section_beats = section.repeats * chorus_beats
            if beat < elapsed + section_beats:
                return section, int((beat - elapsed) // chorus_beats)
            elapsed += section_beats
        last = self.form[-1]
        chorus_index = int((beat - elapsed) // chorus_beats) + (last.repeats or 0)
        return last, chorus_index

    def transpose(self, semitones: int) -> "Song":
        new_key = None if self.key is None else (self.key + semitones) % 12
        return Song(
            title=self.title,
            changes=self.changes.transpose(semitones),
            form=list(self.form),
            tempo_bpm=self.tempo_bpm,
            feel=self.feel,
            key=new_key,
            modal=self.modal,
        )
