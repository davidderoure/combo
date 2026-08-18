from pathlib import Path

from song import Chord, Changes, ChangesEvent, Section, Song, parse_chart, format_chart

CHARTS_DIR = Path(__file__).resolve().parent.parent / "songs"


def test_chord_parse_and_str():
    assert str(Chord.parse("F7")) == "F7"
    assert str(Chord.parse("Bbmaj7")) == "Bbmaj7"
    assert str(Chord.parse("C")) == "C"  # bare triad
    assert str(Chord.parse("D-7")) == "Dm7"  # alias normalises to canonical spelling


def test_chord_transpose_wraps_pitch_class():
    assert Chord.parse("F7").transpose(12) == Chord.parse("F7")
    assert str(Chord.parse("Bb7").transpose(2)) == "C7"


def test_changes_chord_at_cycles():
    changes = Changes([
        ChangesEvent(Chord.parse("F7"), 2.0),
        ChangesEvent(Chord.parse("Bb7"), 2.0),
    ])
    assert changes.chord_at(0.0) == Chord.parse("F7")
    assert changes.chord_at(1.9) == Chord.parse("F7")
    assert changes.chord_at(2.0) == Chord.parse("Bb7")
    # cycles past total_beats
    assert changes.chord_at(4.0) == Chord.parse("F7")
    assert changes.chord_at(6.5) == Chord.parse("Bb7")


def test_blues_chart_round_trip():
    text = (CHARTS_DIR / "blues_in_f.chart").read_text()
    song = parse_chart(text)

    assert song.title == "Blues in F"
    assert song.tempo_bpm == 132
    assert song.changes.total_beats == 48.0  # 12 bars x 4 beats
    assert [s.name for s in song.form] == ["Head", "Solos", "Out"]
    assert [s.repeats for s in song.form] == [1, 3, 1]

    reparsed = parse_chart(format_chart(song))
    assert reparsed.changes.events == song.changes.events
    assert reparsed.form == song.form


def test_song_total_beats_fully_specified():
    song = parse_chart((CHARTS_DIR / "blues_in_f.chart").read_text())
    # (1 + 3 + 1) choruses x 48 beats/chorus
    assert song.total_beats == 5 * 48.0


def test_song_total_beats_none_when_open_ended():
    song = parse_chart((CHARTS_DIR / "blues_in_f.chart").read_text())
    song.form[-1] = Section(name="Solos", repeats=None)
    assert song.total_beats is None


def test_song_section_at_walks_form_and_counts_choruses():
    song = parse_chart((CHARTS_DIR / "blues_in_f.chart").read_text())
    chorus = song.changes.total_beats  # 48 beats

    section, chorus_index = song.section_at(0.0)
    assert (section.name, chorus_index) == ("Head", 0)

    section, chorus_index = song.section_at(chorus)  # start of 2nd chorus overall
    assert (section.name, chorus_index) == ("Solos", 0)

    section, chorus_index = song.section_at(chorus * 3.5)  # partway through Solos
    assert (section.name, chorus_index) == ("Solos", 2)

    section, chorus_index = song.section_at(chorus * 4.2)  # into Out
    assert (section.name, chorus_index) == ("Out", 0)

    # past the nominal end of a fully-specified form (5 choruses total: 1+3+1):
    # keeps cycling the last section rather than raising
    section, chorus_index = song.section_at(chorus * 5.0)
    assert (section.name, chorus_index) == ("Out", 1)


def test_song_section_at_open_ended_keeps_counting():
    song = parse_chart((CHARTS_DIR / "blues_in_f.chart").read_text())
    song.form[1] = Section(name="Solos", repeats=None)
    chorus = song.changes.total_beats

    section, chorus_index = song.section_at(chorus * 10)
    assert section.name == "Solos"
    assert chorus_index == 9  # 10th chorus of the open-ended Solos section


def test_song_transpose():
    song = parse_chart((CHARTS_DIR / "blues_in_f.chart").read_text())
    up_a_step = song.transpose(2)
    assert up_a_step.chord_at(0.0) == Chord.parse("G7")
    assert up_a_step.key == Chord.parse("G").root
    # original is untouched
    assert song.chord_at(0.0) == Chord.parse("F7")


# ---------------------------------------------------------------------------
# Song.modal (Phase 27) -- a chart-authored style choice
# ---------------------------------------------------------------------------


def test_chart_without_modal_header_defaults_false():
    # Every existing chart, including blues_in_f.chart -- must keep working
    # unchanged.
    song = parse_chart((CHARTS_DIR / "blues_in_f.chart").read_text())
    assert song.modal is False


def test_chart_with_modal_true_header_parses_to_modal_song():
    text = "title: Modal Tune\ntempo: 120\nmodal: true\n\nchanges:\nDm7\n\nform:\nSolos x1\n"
    song = parse_chart(text)
    assert song.modal is True


def test_chart_with_modal_header_case_insensitive_and_yes_variant():
    text = "title: Modal Tune\ntempo: 120\nmodal: YES\n\nchanges:\nDm7\n\nform:\nSolos x1\n"
    assert parse_chart(text).modal is True


def test_modal_chart_round_trips_through_format_chart():
    text = "title: Modal Tune\ntempo: 120\nmodal: true\n\nchanges:\nDm7\n\nform:\nSolos x1\n"
    song = parse_chart(text)
    reparsed = parse_chart(format_chart(song))
    assert reparsed.modal is True


def test_non_modal_chart_format_chart_omits_the_header():
    song = parse_chart((CHARTS_DIR / "blues_in_f.chart").read_text())
    assert "modal:" not in format_chart(song)


def test_song_transpose_preserves_modal():
    song = parse_chart((CHARTS_DIR / "blues_in_f.chart").read_text())
    modal_song = Song(
        title=song.title, changes=song.changes, form=song.form,
        tempo_bpm=song.tempo_bpm, feel=song.feel, key=song.key, modal=True,
    )
    assert modal_song.transpose(2).modal is True
