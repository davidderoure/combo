"""Plain-text chart format for authoring a Song by hand, the way a jazz
fake-book chart gets written down.

    title: Blues in F
    tempo: 132
    feel: swing
    key: F

    changes:
    F7   | Bb7 | F7  | F7
    Bb7  | Bb7 | F7  | F7
    C7   | Bb7 | F7  | C7

    form:
    Head x1
    Solos x3
    Out x1

`changes:` is one cycle of the harmony, written as bar lines separated by
`|`. Multiple chords in a bar split that bar's beats evenly (4/4 assumed
throughout — no time-signature support yet). `%` repeats the previous
chord. `form:` lists sections in order as "Name xN" (N repeats through the
changes) or "Name x?" for an open-ended section (repeats=None).

An optional `modal: true` header line (Phase 27, DESIGN.md §12) marks the
whole chart as triadic vs. quartal/modal — a David-authored style choice
(mirroring how a player would read the artist/date on a score), not
inferred. Absent, or any other value, means False — every existing chart
keeps working unchanged.
"""

import re

from .chord import Chord
from .changes import Changes, ChangesEvent
from .form import Section
from .song import Song

_BEATS_PER_BAR = 4.0


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def parse_chart(text: str) -> Song:
    lines = [_strip_comment(line) for line in text.splitlines()]

    header: dict[str, str] = {}
    changes_lines: list[str] = []
    form_lines: list[str] = []
    section = None  # None | "changes" | "form"

    for line in lines:
        if not line:
            continue
        if line == "changes:":
            section = "changes"
            continue
        if line == "form:":
            section = "form"
            continue
        if section == "changes":
            changes_lines.append(line)
        elif section == "form":
            form_lines.append(line)
        else:
            key, _, value = line.partition(":")
            header[key.strip().lower()] = value.strip()

    changes = _parse_changes(changes_lines)
    form = _parse_form(form_lines)

    key = header.get("key")
    return Song(
        title=header.get("title", "untitled"),
        changes=changes,
        form=form,
        tempo_bpm=float(header["tempo"]) if "tempo" in header else 120.0,
        feel=header.get("feel", "swing"),
        key=Chord.parse(key).root if key else None,
        modal=header.get("modal", "").lower() in ("true", "yes"),
    )


def _parse_changes(lines: list[str]) -> Changes:
    events: list[ChangesEvent] = []
    last_chord: Chord | None = None
    for line in lines:
        for bar in line.split("|"):
            tokens = bar.split()
            if not tokens:
                continue
            beats_each = _BEATS_PER_BAR / len(tokens)
            for token in tokens:
                if token == "%":
                    if last_chord is None:
                        raise ValueError("'%' with no preceding chord to repeat")
                    chord = last_chord
                else:
                    chord = Chord.parse(token)
                events.append(ChangesEvent(chord, beats_each))
                last_chord = chord
    if not events:
        raise ValueError("chart has no changes")
    return Changes(events)


_FORM_LINE_RE = re.compile(r"^(?P<name>.+?)\s+x(?P<repeats>\d+|\?)$")


def _parse_form(lines: list[str]) -> list[Section]:
    sections = []
    for line in lines:
        match = _FORM_LINE_RE.match(line)
        if not match:
            raise ValueError(f"invalid form line {line!r}, expected 'Name xN' or 'Name x?'")
        repeats_text = match.group("repeats")
        repeats = None if repeats_text == "?" else int(repeats_text)
        sections.append(Section(name=match.group("name"), repeats=repeats))
    if not sections:
        raise ValueError("chart has no form")
    return sections


def format_chart(song: Song) -> str:
    lines = [f"title: {song.title}", f"tempo: {song.tempo_bpm:g}", f"feel: {song.feel}"]
    if song.key is not None:
        lines.append(f"key: {Chord(song.key, 'maj')}")
    if song.modal:
        lines.append("modal: true")
    lines.append("")
    lines.append("changes:")

    bar = []
    beats_in_bar = 0.0
    bar_lines = []
    for event in song.changes.events:
        bar.append(str(event.chord))
        beats_in_bar += event.duration_beats
        if beats_in_bar >= _BEATS_PER_BAR:
            bar_lines.append(" ".join(bar))
            bar, beats_in_bar = [], 0.0
    if bar:
        bar_lines.append(" ".join(bar))
    lines.append(" | ".join(bar_lines))

    lines.append("")
    lines.append("form:")
    for s in song.form:
        lines.append(f"{s.name} x{'?' if s.repeats is None else s.repeats}")

    return "\n".join(lines) + "\n"
