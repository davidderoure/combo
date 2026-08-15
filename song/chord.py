"""Chord symbols: a pitch-class root plus jazz quality shorthand."""

from dataclasses import dataclass

_ROOT_PITCH_CLASS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# maps written shorthand -> canonical quality name
_QUALITY_ALIASES = {
    "": "maj", "maj": "maj", "M": "maj",
    "7": "7",
    "maj7": "maj7", "M7": "maj7", "delta7": "maj7",
    "m": "m", "-": "m", "min": "m",
    "m7": "m7", "-7": "m7", "min7": "m7",
    "m7b5": "m7b5", "-7b5": "m7b5", "half-dim": "m7b5",
    "o7": "dim7", "dim7": "dim7", "dim": "dim7",
    "6": "6",
    "m6": "m6", "-6": "m6",
    "sus": "sus4", "sus4": "sus4",
    "9": "9", "maj9": "maj9", "m9": "m9", "-9": "m9",
    "13": "13", "7#11": "7#11", "7b9": "7b9", "7alt": "7alt",
}

_FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


@dataclass(frozen=True)
class Chord:
    root: int  # pitch class 0-11
    quality: str  # canonical quality, e.g. "maj7", "m7", "7", "maj"

    @classmethod
    def parse(cls, text: str) -> "Chord":
        text = text.strip()
        if not text:
            raise ValueError("empty chord symbol")
        letter = text[0].upper()
        if letter not in _ROOT_PITCH_CLASS:
            raise ValueError(f"unrecognised chord root in {text!r}")
        pitch = _ROOT_PITCH_CLASS[letter]
        i = 1
        while i < len(text) and text[i] in "#b":
            pitch += 1 if text[i] == "#" else -1
            i += 1
        quality_text = text[i:]
        if quality_text not in _QUALITY_ALIASES:
            raise ValueError(f"unrecognised chord quality {quality_text!r} in {text!r}")
        return cls(root=pitch % 12, quality=_QUALITY_ALIASES[quality_text])

    def transpose(self, semitones: int) -> "Chord":
        return Chord(root=(self.root + semitones) % 12, quality=self.quality)

    def __str__(self) -> str:
        suffix = "" if self.quality == "maj" else self.quality
        return f"{_FLAT_NAMES[self.root]}{suffix}"
