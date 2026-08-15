"""A song's form: an ordered list of named sections, each a number of
cycles through the song's changes (see DESIGN.md §3)."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Section:
    name: str
    repeats: Optional[int]  # number of chorus cycles; None = open-ended
