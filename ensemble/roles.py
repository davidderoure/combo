"""Role assignment — the accompanist-doubling slice of DESIGN.md §2.

§2 names two distinct things under "role assignment": (a) any voice taking any role
(solo/accompany/lay out/trade) in any section, assigned by a tune-level form
controller — genuinely ArcController-scale work, not attempted here; (b)
same-instrument doubling — when two same-register voices are both accompanying at
once, the default is to split the role, one full, one laying out. This module builds
(b) only, the smallest real slice, same discipline as every earlier phase's partial
slice of a bigger deferred concept.

Deliberately construction-time, not a live per-bar signal: role splitting is
inherently per-voice (voice A gets "full," voice B gets "lay out"), and deciding it
live inside Session.generate would mean computing it from what other voices are ABOUT
to play this same bar — which conflicts with the tested "every voice sees only a
snapshot of prior bars, order-independent" guarantee. So this is decided once, at
ensemble-construction time, the same pattern as sax_generator's n_candidates/memory/
plan_bars — no changes to Generator, Session, or any other generator's signature.
"""

from typing import Dict, List, Tuple


def _overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    a_low, a_high = a
    b_low, b_high = b
    return not (a_high < b_low or b_high < a_low)


def default_accompanist_roles(voices: List[Tuple[str, Tuple[int, int]]]) -> Dict[str, bool]:
    """voices: (voice_id, register) pairs for accompanist CANDIDATES only — callers
    decide which voices are accompanists (no formal 'role' field exists on Voice, and
    this phase deliberately doesn't add one — see module docstring). Returns
    {voice_id: True} for the full accompanist, {voice_id: False} for lay-out, per a
    simple greedy rule: process in list order, keep a voice "full" unless its register
    overlaps an already-claimed full accompanist's, in which case it lays out.
    Deterministic, order-dependent by design (documented, not hidden) — general
    interval-graph clustering for 3+ mutually-overlapping voices is real complexity
    not needed to prove this mechanism."""
    roles: Dict[str, bool] = {}
    claimed_registers: List[Tuple[int, int]] = []
    for voice_id, register in voices:
        if any(_overlaps(register, claimed) for claimed in claimed_registers):
            roles[voice_id] = False
        else:
            roles[voice_id] = True
            claimed_registers.append(register)
    return roles
