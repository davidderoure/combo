"""Tests for gesture/vocabulary.py — two layers, deliberately.

Most tests feed synthetic SubGesture objects directly via feed_subgesture, so the
composition logic (rules, record/alias/pending) is tested precisely without being
coupled to SubGestureRecognizer's exact note-timing internals. One end-to-end test at
the bottom proves the real note-to-gesture wiring works too.
"""

from gesture.recognizer import SubGesture
from gesture.vocabulary import GestureRecognizer


def make_sg(label: str) -> SubGesture:
    return SubGesture(label=label, start_time=0.0, duration=0.1, n=1)


def make_vocab():
    fired = []
    vocab = GestureRecognizer(on_gesture=fired.append)
    return vocab, fired


def feed(vocab: GestureRecognizer, labels):
    for label in labels:
        vocab.feed_subgesture(make_sg(label))


def test_long_note_fires_handover():
    vocab, fired = make_vocab()
    feed(vocab, ["L"])
    assert [g.name for g in fired] == ["handover"]


def test_trill_fires_reset_tempo():
    vocab, fired = make_vocab()
    feed(vocab, ["T"])
    assert [g.name for g in fired] == ["reset_tempo"]


def test_two_same_runs_fire_toggle_singability():
    vocab, fired = make_vocab()
    feed(vocab, ["S", "S"])
    # Exactly one fire, on the second S -- no 1-length "S" rule exists to fire
    # prematurely on the first (see DEFAULT_RULES' own comment on why "T","T"
    # couldn't have been used here instead).
    assert [g.name for g in fired] == ["toggle_singability"]


def test_two_separate_rests_fire_toggle_dissonance_avoidance():
    vocab, fired = make_vocab()
    feed(vocab, ["R", "R"])
    # Exactly one fire, on the second R -- see DEFAULT_RULES' own comment on
    # why this was verified empirically before being chosen.
    assert [g.name for g in fired] == ["toggle_dissonance_avoidance"]


def test_a_rest_interrupted_by_a_note_does_not_fire_toggle_dissonance_avoidance():
    vocab, fired = make_vocab()
    feed(vocab, ["R", "U", "R"])
    assert fired == []


def test_unrelated_labels_fire_nothing():
    vocab, fired = make_vocab()
    feed(vocab, ["U", "D", "S", "R"])
    assert fired == []


def test_record_and_alias_teaches_a_new_pattern():
    vocab, fired = make_vocab()
    rule_count_before = len(vocab.rules)

    # begin-record (U,U), teach content "D,S,L" (its tail "L" matches the existing
    # L -> handover rule), end-record (D,D) — should alias the full taught pattern.
    # Deliberately not "S,S,..." for the filler content: "S","S" is itself now a
    # seed rule (toggle_singability, DEFAULT_RULES) — using it here would make the
    # REPLAY below fire that 2-length rule as a prefix match before ever reaching
    # the taught 3-length pattern, since rules are checked in list order and a
    # shorter match always wins first (same reasoning DEFAULT_RULES' own comment
    # gives for why toggle_singability couldn't be built on "T","T" either).
    feed(vocab, ["U", "U", "D", "S", "L", "D", "D"])
    assert fired == []  # recording suppresses ordinary rule-firing while it's active
    assert vocab.pending == []
    assert len(vocab.rules) == rule_count_before + 1
    assert vocab.rules[-1].pattern == ("D", "S", "L")
    assert vocab.rules[-1].gesture.name == "handover"

    # replaying the exact taught sequence should now fire the aliased gesture
    fired.clear()
    feed(vocab, ["D", "S", "L"])
    assert [g.name for g in fired] == ["handover"]


def test_record_without_alias_is_stored_as_pending_not_dropped_or_guessed():
    vocab, fired = make_vocab()
    feed(vocab, ["U", "U", "S", "R", "D", "D"])
    assert fired == []
    assert vocab.pending == [("S", "R")]


def test_recording_marker_labels_themselves_are_not_included_in_recorded_content():
    vocab, fired = make_vocab()
    feed(vocab, ["U", "U", "S", "D", "D"])
    assert vocab.pending == [("S",)]


def test_end_to_end_real_long_note_fires_handover():
    fired = []
    vocab = GestureRecognizer(on_gesture=fired.append)

    vocab.note_on(60, 80, 0.0)
    # no note_off — held well past min_long_duration (2.0s default), same as
    # test_recognizer.py's own long-note test
    vocab.tick(2.5)

    assert [g.name for g in fired] == ["handover"]
