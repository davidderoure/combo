"""Tests for ensemble/roles.py — pure functions, no MIDI/audio/weights needed."""

from ensemble.roles import default_accompanist_roles


def test_non_overlapping_registers_both_stay_full():
    roles = default_accompanist_roles([("keys", (48, 60)), ("guitar", (61, 72))])
    assert roles == {"keys": True, "guitar": True}


def test_overlapping_registers_first_full_second_lays_out():
    roles = default_accompanist_roles([("keys", (48, 72)), ("guitar", (60, 84))])
    assert roles == {"keys": True, "guitar": False}


def test_overlap_order_is_list_order_not_alphabetical():
    roles = default_accompanist_roles([("guitar", (60, 84)), ("keys", (48, 72))])
    assert roles == {"guitar": True, "keys": False}


def test_touching_registers_count_as_overlapping():
    # (48, 60) and (60, 72) share pitch 60 — an interval overlap, not exact-match.
    roles = default_accompanist_roles([("keys", (48, 60)), ("guitar", (60, 72))])
    assert roles == {"keys": True, "guitar": False}


def test_third_separate_voice_stays_full_independently():
    roles = default_accompanist_roles(
        [("keys", (48, 72)), ("guitar", (60, 84)), ("bass", (28, 47))]
    )
    assert roles == {"keys": True, "guitar": False, "bass": True}


def test_empty_input_returns_empty():
    assert default_accompanist_roles([]) == {}


def test_single_voice_stays_full():
    assert default_accompanist_roles([("keys", (48, 72))]) == {"keys": True}
