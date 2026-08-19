"""Validates our own WJD chord extraction (wjd_corpus.py's _wjd_chord_idx/
_wjd_chord_quality) against something recorded INDEPENDENTLY of the chord
string and of anything built in Phases 29-30 -- beats.bass_pitch, a real
per-beat MIDI note number already in wjazzd.db. Phase 31, DESIGN.md §13.

Prompted directly by Phase 30's critic baseline: tonal_conformity scored
real WJD solos LOWER than combo, the one result that couldn't be explained
away as tautological (unlike singability/phrasing, whose targets were
calibrated from combo's own output) or as genuinely informative (unlike
repetition). The obvious check -- do the melody notes fit the chord -- was
flagged as circular: a poor fit could mean a wrong chord label, or it could
mean legitimate outside playing, and that signal alone can't distinguish
them. beats.bass_pitch sidesteps this entirely: bass players overwhelmingly
land on the root (or an unambiguous chord tone) at a chord's downbeat, so
comparing our extracted root against it validates chord extraction without
assuming anything about how the soloist plays over it. Also a real,
musicologist-endorsed technique, not invented for this project -- David used
implied bassline as part of an earlier musicality measure for LSTM-generated
piano pieces.

    python bass_chord_check.py

For a slash chord ("Ab79/C"), the expected bass pitch class is the part
after '/' (wjd_corpus._wjd_expected_bass_pc), not the chord's own root --
checked directly that this matters, slash chords are real and not rare in
this corpus.

Real result (first run, whole corpus): 42.3% exact root match, 53.7% match
against any chord tone (root/3rd/5th/7th) over 28,887 comparable rows (chord-
annotated beats with a real, non-zero bass_pitch) -- both well above chance
(~8% for a random root of 12, ~25-33% for a random chord tone) but far from
a clean confirmation. Two breakdowns checked to see whether this points at a
real, localized classification bug (like Phase 29's parse_chord "j" bug) or
something more general: bar-downbeat rows (beat==1) score only modestly
higher than mid-bar chord changes (43.6% vs 38.1% -- not a dramatic gap);
broken down by our own derived quality class, all four classes cluster in a
narrow 40-46% root-match band with no outlier. Honest reading: this doesn't
look like a bug concentrated in one place -- more likely real walking-bass
practice (chromatic approaches, passing tones, syncopation around the
labelled beat) and/or WJD's own beat-grid alignment granularity, but this is
the most likely explanation given the breakdowns, not a settled conclusion.
Not yet acted on -- chord classification is left as-is pending further
investigation (e.g. auditing individual real solos by ear/lead-sheet, a
separate, human-driven check not attempted here).
"""

from collections import defaultdict

from ensemble.wolfson.chords import QUAL_DIM, QUAL_DOM, QUAL_MAJOR, QUAL_MINOR
from ensemble.wolfson.scales import chord_tones
from wjd_corpus import DB_PATH, _wjd_chord_idx, _wjd_chord_quality, _wjd_expected_bass_pc

QUALITY_NAMES = {QUAL_MAJOR: "major", QUAL_DOM: "dominant", QUAL_MINOR: "minor", QUAL_DIM: "diminished"}
MIN_SUFFIX_SAMPLE = 20  # a raw chord string needs at least this many comparable
                          # rows before its own match rate is reported -- avoids
                          # noise from a suffix seen only once or twice


def _comparable_rows(db_path):
    """Yields (chord, bass_pc, beat) for every beats row with a real chord
    annotation and a real (non-null, non-zero) bass_pitch."""
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute("SELECT chord, bass_pitch, beat FROM beats WHERE chord != ''")
        for chord, bass_pitch, beat in cur:
            if bass_pitch:
                yield chord, bass_pitch % 12, beat
    finally:
        conn.close()


def _report(rows, label: str) -> None:
    total = 0
    root_match = 0
    chordtone_match = 0
    for chord, bass_pc, _beat in rows:
        expected = _wjd_expected_bass_pc(chord)
        idx = _wjd_chord_idx(chord)
        if expected is None or idx is None:
            continue
        total += 1
        if bass_pc == expected:
            root_match += 1
        if bass_pc in chord_tones(idx):
            chordtone_match += 1
    if total == 0:
        print(f"{label}: no comparable rows")
        return
    print(f"{label}: n={total}  root_match={100 * root_match / total:.1f}%  chord_tone_match={100 * chordtone_match / total:.1f}%")


def main() -> None:
    if not DB_PATH.exists():
        print(f"{DB_PATH} not found -- see wjd_corpus.py's module docstring for how to obtain it.")
        return

    rows = list(_comparable_rows(DB_PATH))

    _report(rows, "overall")
    _report([r for r in rows if r[2] == 1], "bar downbeat (beat==1)")
    _report([r for r in rows if r[2] != 1], "mid-bar chord change (beat!=1)")

    print()
    by_quality = defaultdict(list)
    for chord, bass_pc, _beat in rows:
        q = _wjd_chord_quality(chord)
        if q is not None:
            by_quality[q].append((chord, bass_pc, _beat))
    for q in sorted(by_quality):
        _report(by_quality[q], f"quality={QUALITY_NAMES[q]}")

    print(f"\nworst-matching raw chord strings (n>={MIN_SUFFIX_SAMPLE}, by root-match rate):")
    by_chord = defaultdict(lambda: [0, 0])  # total, root_match
    for chord, bass_pc, _beat in rows:
        expected = _wjd_expected_bass_pc(chord)
        if expected is None:
            continue
        by_chord[chord][0] += 1
        if bass_pc == expected:
            by_chord[chord][1] += 1
    ranked = sorted(
        ((chord, total, rm) for chord, (total, rm) in by_chord.items() if total >= MIN_SUFFIX_SAMPLE),
        key=lambda t: t[2] / t[1] if t[1] else 0,
    )
    for chord, total, rm in ranked[:15]:
        print(f"  {chord:12s} n={total:4d}  root_match={100 * rm / total:.1f}%")


if __name__ == "__main__":
    main()
