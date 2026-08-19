# combo — design notes

Status: design consolidated 2026-08-06. This document is the single source of truth
for the design; individual decisions below supersede any earlier scattered notes.

**Built so far**: the gesture sub-gesture layer (§9, ported from AGRP), the
song/scenario data model (§3), an ensemble skeleton MVP (§2/§4), a gesture vocabulary
composition layer MVP (§10.1/§10.2), a rule-based drums voice (§7),
accompaniment-listening (§5), the musical director's dial channel (§11),
performer/director MIDI input (§6), the handover half of transition triggers
(§8), the same-instrument-doubling slice of role assignment (§2), and the
MIDI playback stage §4 named as designed-but-not-built (`output/midi_output.py`,
`self_test.py` — see [README.md](README.md)). The ensemble MVP is a thin
`Voice`, a `Session` that steps a `Song` bar-by-bar into a symbolic `Timeline`, and
machine-speed-vs-real-time pacing behind a single generation loop — proven multi-voice,
not just single-voice, via `ensemble/demo.py` running the `chord_tone_generator` sax
stub alongside the drums voice together over `blues_in_f.chart`. The gesture
vocabulary MVP (`gesture/vocabulary.py`, `GestureRecognizer`) composes
`SubGestureRecognizer`'s output into named `Gesture`s: two argument-less seed gestures
(`handover()`, `reset_tempo()`), and the record/alias/pending teaching mechanism from
§10.2 (a reserved begin/end marker pair, chosen for a checkable — not arbitrary —
reason: see the module docstring). Drums (`ensemble/drums.py`, `drum_generator`) are
section-aware via `Song.section_at` (sparse under head/out, medium by default, busy on
a section's last chorus) with honestly-flagged discrete-hits-not-continuous-brushes and
seeded humanisation. Accompaniment-listening (`ensemble/listening.py`,
`ensemble/comping.py`) required extending every `Generator` to receive a defensive-copy
snapshot of prior bars, not just `(song, bar_index)` — voices can now actually listen
to each other; the concrete accompanist (`comping_generator`) ducks/fills/plays
moderately based on a target voice's recent density, demonstrated against a synthetic
varying-density fixture since the sax stub's density never varies. The director
(`ensemble/director.py`) mirrors `Voice`/`Generator`'s shape (`Director`,
`DirectorSource`); `comping_generator` shifts its thresholds with the aggregated
intensity, and `ensemble_intensity_critic` derives that intensity by genuinely
listening to the ensemble's own combined density — the first real "AI critic," not
just a manually-supplied constant. MIDI input (`input/sources.py`) now dispatches by
role from `config.MIDI_SOURCES`: performer sources each get their own
`MidiListener`/`GestureRecognizer` pair (§2's multi-human principle — this is no
longer just an unwired data-model gap), director sources get the same
`MidiListener` (unified in Phase 13 — see below) reading a live Control Change into
the intensity dial. Verified with a real (if virtual, via macOS's IAC Driver) MIDI
port end-to-end — an actual note-on produced `Gesture("handover")`, an actual CC
message updated a live intensity value — though not against physical hardware,
which this environment doesn't have. Handover
transitions (`ensemble/transitions.py`, `TransitionController`) let a recognised
`handover()` shorten the current section to end after its current chorus — every
existing generator picks this up with zero code changes, since `Session` now passes
an *effective* (possibly form-truncated) `Song` through the same parameter slot they
already read from; proven against a real consumer (`drum_generator`'s density shifts
sections early), not just checked in isolation. This is the first real slice of the
long-referenced `ArcController`. And the sax voice's stub generator has been replaced
with a real generative model (`ensemble/sax.py`, `ensemble/wolfson/`) — an LSTM
adapted from Wolfson, David's earlier one-voice system — the first of the ensemble's
voices to move off `chord_tone_generator` (§12). The director's aggregated intensity
now reaches that generation too: `rhythmic_density` is the one bias parameter the
ported model itself already frames as a general busyness dial, so `sax_generator`
passes `director_signal.intensity` straight through with no translation needed —
verified with a real empirical probe (0.723 vs. 0.419 beats average note duration at
`rhythmic_density` 0.0 vs. 1.0) before the test threshold was chosen. Sax now also
plans several bars ahead — David asked directly whether generating bar-by-bar loses
a soloist's "conscious planning," and the answer built into `sax_generator` is a
buffer: it generates a chord-consistent multi-bar span in ONE continuous
`generate()` call (the model's own arc-position-driven bias layers now sweep across
that real span instead of resetting every bar) and dispenses one bar per `Session`
call from the buffer, refilling only when it's exhausted. No revision-on-mismatch
mechanism was needed — checked directly in `ensemble/transitions.py` and
`song/song.py`, not assumed: a handover only ever changes `Song.form`, never
`Song.changes`, and `chord_at` only ever reads `changes`, so a plan's chord
assumptions can never go stale between when it's built and when it's dispensed.
Sax also now has rehearsal memory (`ensemble/memory.py`, `RehearsalMemory`) — the
first thing in combo that persists across separate `Session.generate()` calls,
prompted directly by David distinguishing search-in-flight ("the chess approach,"
still deferred) from *rehearsal*: play a piece more than once, carry what worked
into the next run, and the actual performance happens on the fly but is informed
by that practice. Real prior art was found and partly reused: wolfson's
`memory/phrase_memory.py`/`input/phrase_analyzer.py` already had this shape
(`extract_interval_motifs`, ported near-verbatim into `ensemble/wolfson/motifs.py`;
`PhraseMemory`'s store/recall pattern re-authored, not ported, since its reset
policy — between `ArcController`'s arc loops within one performance — doesn't fit
persisting *across* performances). Wiring was verified with a spy on
`PhraseGenerator.generate`'s actual call arguments, not the musical output —
a real empirical probe found the model only follows a fed-in motif rarely (2/40
trials), which is why the test asserts the *plumbing* deterministically rather
than the stochastic musical effect. What's remembered is now quality-weighted, not
pure frequency (`ensemble/critic.py`, a musicality critic) — closing the gap
`ensemble/memory.py`'s own docstring named. Grounded in two real sources, not
first-principles guessing: David's unrelated prior work measuring "musicality" for
LSTM-generated piano sight-reading pieces (a Colab notebook analysing real Grade 1
specimens via corpus-similarity — interval histograms and melodic contour reduced
to a string of U/D/S compared by edit distance; a striking, unplanned overlap with
`gesture/recognizer.py`'s own sub-gesture alphabet, which already contains that
exact U/D/S vocabulary for an unrelated purpose), and Wolfson's own ported bias
layers, three of which are repurposed from generation-time sampling biases into
retrospective scoring functions (singability's bell curve, voice-leading's
chord-tone resolution). All seven metrics (`tonal_conformity`, `contour_smoothness`,
`repetition`, `call_response_relatedness`, `singability`, `phrasing` — Phase 23,
`register_usage` — Phase 24, below) are pure, deterministic functions needing no
model inference — the first sax-adjacent test file since Phase 8 that's fully
testable without `sax_best.pt`. `phrasing` is the only one of the seven that looks
at raw notes (including `REST_PITCH` sentinels) rather than the
`_real_notes()`-filtered view every other metric uses — prompted directly by a
listening-test observation ("the solos are not speaking in 'sentences' with gaps
in between") distinct from the dissonance thread: it measures the fraction of a
chunk's total duration spent in a genuine rest (`phrase_generator.py`'s own
`_inject_rests` already splices these in, bell-curve-weighted toward the phrase
midpoint but capped at `REST_MAX_PROBABILITY=0.15`), scored via the same
bell-curve treatment as `singability`, peaking at an empirically-grounded target
(40 real 4-bar chunks had mean breath fraction 0.129, median 0.124 — the target
sits close to that natural centre, nudged slightly higher to push selection
toward a bit more breath than the unweighted default). Needed **no changes to
`ensemble/sax.py` at all** — `sax_generator`'s selection key already reads
`musicality_score(...).overall` generically, so the new sub-score flows into
real selection through the existing weighted-sum mechanism the moment it's added
to `DEFAULT_WEIGHTS`. Deliberately not a separate "sentence count" measure on top
of the breath fraction — a possible future refinement, not attempted.

`register_usage` (Phase 24) closes another listening-test gap: "not much range
of the instrument is being used. But on reflection, this is correct for
beginners." Measures the fraction of the active `register`'s width spanned by a
chunk's IN-REGISTER real notes only — checked directly, not assumed:
`musicality_score` scores the model's RAW candidate output, before
`_split_phrase_into_bars` clips out-of-register pitches for playback, and 13 of
40 real chunks had at least one out-of-register note (Wolfson's own trained
pitch vocabulary is MIDI 44-93, wider than any `register` passed in) — a
raw-span measurement would sometimes reward spread that never actually sounds.
Deliberately **not** a bell curve around a target the way `phrasing`/`singability`
are: there's no clear "too much range" complaint the way there is for breath —
`register` itself already caps what's appropriate for the chosen skill level (a
call-site choice), and `contour_smoothness` already penalises erratic wide leaps
separately — so this is a monotonic "use more of what you're given" reward
instead. No separate beginner/advanced mode either: the skill-level distinction
lives entirely in which `register` the caller passes to `sax_generator` (already
real), not in the metric's own shape. Unlike `phrasing`, this one DID need a
small `ensemble/sax.py` change — `musicality_score()` gained a new required
`register` parameter (no sensible default, same as `chord_idx`/`seed_phrase`),
rippling through every existing call site (14 total, across `ensemble/sax.py`
and both critic/integration test files) — expected, mechanical fallout, not a
regression. `DEFAULT_WEIGHTS` rebalanced to seven keys.

**`register_usage` now rewards excursions across a whole performance, not
just span within one chunk (Phase 32)** — the first concrete fix following
Phase 30's critic baseline. Root cause was structural: the metric only ever
saw one candidate's own 5-8 notes, and a short excerpt can't reflect a whole
solo's real range no matter how far the player goes elsewhere — exactly why
WJD's own wide natural range scored *lower* than combo's narrower
`SAX_REGISTER` in Phase 30. Fix: `register_usage` gains an optional
`prior_range: Optional[Tuple[int, int]]` — the (low, high) bounds this same
voice has already explored earlier in the performance — and judges the
candidate against `prior_range ∪ candidate's own notes`, not the candidate
alone. This has the right incentive shape for free: replaying
already-covered territory can't raise the score above `prior_range`'s own
span (it's already counted); only a genuine excursion beyond what's already
been played does — a direct, mechanical model of "occasional excursions are
good," not just "reward wide spans." `ensemble/sax.py`'s `sax_generator`
tracks `own_pitch_range` as closure state across chunk-builds (same
convention as `critic_weights`/`dissonance_mode`), updated after each
chunk's winner is chosen. `prior_range=None` (every pre-existing caller)
reproduces the exact old per-chunk-only formula. Real, measured effect via
`critic_baseline.py --self-test-only`: combo's own `register_usage` average
rose from 0.348 to **0.903**, and `overall` from 0.632 to 0.744 — a large,
real shift, not a marginal tweak. A named, accepted limitation: `prior_range`
only ever grows within one performance, so once the full register has
genuinely been explored, later chunks can't score higher on this axis no
matter what they do — a decaying/windowed "recent range" instead of the
whole performance so far is a possible future refinement, not attempted here.

**`repetition`'s weight is negative now, not just retuned (Phase 33).**
David's own proposal was to reweight `repetition` using the real WJD number
(29.4% of chunks show repetition) as a calibration target, mirroring how
`TARGET_BREATH_FRACTION` was set. Checked directly before implementing that
literally: `repetition()` returns `1.0` when a chunk shows a repeated
pattern, and `DEFAULT_WEIGHTS["repetition"]` blended it *positively* into
`overall` — Phase 12's original "motivic restatement as coherence" framing,
a real technique, but combo already shows it in 64.8% of chunks vs. WJD's
29.4%, so simply *increasing* the weight (the literal reading of
"reweighting") would have pushed selection toward even more repetition, the
wrong direction. A bell-curve target doesn't transfer either — repetition is
binary per-chunk, and 29%/65% are aggregate rates, not a per-chunk quantity
a candidate could aim for. The actual fix: `repetition()` itself is
untouched (still a clear, individually meaningful "shows a pattern" signal);
only `DEFAULT_WEIGHTS["repetition"]` changed, `0.1 → -0.1` — literally
negated, the smallest change that corrects the direction, not yet retuned to
a new magnitude. Doesn't touch `motif_adherence` (Phase 17, a separate,
standalone signal — echoing a specifically *recalled* motif is unaffected).
Weights no longer sum to 1.0 (0.8 now) — never a strictly enforced
invariant. Real, measured via `critic_baseline.py --self-test-only`:
combo's own `repetition` rate dropped from 62.3% (Phase 32's own baseline,
immediately before this change) to **41.0%** — a real move toward WJD's
29.4%, though not all the way there; `overall`'s average dropped too (0.744
→ 0.648), expected and not a concern, since `overall` was never a bounded
probability, just a comparison key. Whether `-0.1` needs a larger magnitude
to close the remaining gap is an open, empirically-answerable question for
a future rerun, not decided here.

Recall is also **chord-quality-aware, not just pooled globally** (Phase 25):
`RehearsalMemory.store`/`recall_motifs` take an optional `chord_quality`
(Wolfson's 4-class major/dominant/minor/diminished system), computed once per
chunk in `ensemble/sax.py` as `chord_idx % N_QUALITIES` and threaded through
both calls — recall becomes "what worked over a dominant chord", not "what
worked anywhere". Deliberately tagged by quality, not root or full `chord_idx`
— checked directly before choosing this: `extract_interval_motifs` is already
transposition-invariant, so a shape that worked over one root of a quality is
exactly as valid, transposed, over any other root of the same quality; tagging
by root too would only fragment the buffer for no musical reason. Strict
filtering, no cross-quality fallback (a quality with no history yet simply
recalls nothing, same as no memory at all) — a blended fallback is a real,
separate future refinement, not attempted. **A real finding worth stating
plainly**: `songs/blues_in_f.chart` — the project's own reference chart — is
every chord a dominant 7th (different roots, same quality), so chord-tagged
recall shows *zero* observable difference there; the real effect is
demonstrated instead over `tests/test_sax_wolfson_integration.py`'s existing
`build_ii_v_i_song()` fixture (Dm7-G7-Cmaj7, three distinct qualities),
verified by replaying the actual stored history to confirm a later chorus's
motif target came only from same-quality phrases, never a different chord's.

**Rehearsals now cross process boundaries too, not just `Session.generate()`
calls within one program** (Phase 26), prompted directly by David's own
playing experience: he develops ideas within a song across choruses, and
across separate rehearsals he experiments and carries the best ideas into the
gig — "more rehearsals gives more ideas to remember." `RehearsalMemory` gains
an optional `persist_path`: loaded from at construction if it already exists,
written back to (atomically — write-then-replace, so a crash or Ctrl-C
mid-write can never corrupt the file) after every `store()` call, not deferred
to some explicit save step. Folded into the constructor/`store()` themselves —
"nothing resets automatically... not a `reset()` method to remember to call"
extended to "nothing needs manual saving either." `self_test.py --persist`
keys the file per chart (`rehearsal_memory/<chart>.json`, gitignored — personal
practice data, not source), matching "rehearsing a specific piece" rather than
a global cross-tune vocabulary; off by default, so a plain run still touches
nothing on disk. Verified with the genuine article: two independent
`RehearsalMemory` objects (not the same one reused) sharing one file, the
second recalling real motifs the first wrote — the only way that's possible is
if save and load both actually work end-to-end, not just in isolation.

**The critic no longer fights the generator** (Phase 27) — two concrete
mismatches found from the "why does it still sound beginner-noodly" discussion.
First: `tonal_conformity` never learned about the dissonance work (Phases 19-22)
— it checked the plain, unwidened scale, so a candidate that cleared the
dissonance gate via a resolved tension (`credit_resolved_tension`) still lost
`tonal_conformity` points for the same note, quietly penalising exactly the
"advanced" playing those phases were built to allow. Two real consequences this
fixes: in `n_candidates` search, a bold candidate tied with a safe one on
dissonance no longer automatically loses the `overall` tie-break; in
`RehearsalMemory`, quality-weighted recall no longer systematically favours the
safe phrasing over the bold one. `tonal_conformity` now uses `dissonance_scale`
(the widened scale) plus the same `_is_passing_tone`/`_is_resolved_tension`
exemptions `dissonance()` already had — checked directly before making the
change: every existing `tonal_conformity` test's example notes land the same
result under the new scale reference, so nothing needed updating, only new
tests added.

Second: Wolfson's own quartal/modal generation parameter, `modal_strength`
(`PhraseGenerator.generate()`, biasing toward P4/P5 leaps — *"quartal/pentatonic
character of modal jazz stages"*), was ported but never wired up, and would
have fought `contour_smoothness` (`SMOOTH_INTERVAL_MAX_SEMITONES=4`) if it had
been. Verified empirically before wiring it up: 20 real one-shot generations at
`modal_strength=0.0` vs `1.0` show P4/P5 leaps rising from 10.0% of intervals to
26.0% — a real, substantial effect. `Song` gains a chart-authored `modal: bool`
field (`song/chart.py`'s `modal: true` header line; every existing chart
defaults to `False`, unchanged) — David's own framing: for now read directly off
the chart, the way he'd personally read the artist/date on a score to judge
triadic vs. modal vocabulary (real stylistic knowledge that can't be
replicated); modulating it by narrative-arc position instead is named as a real,
separate future step, not attempted here. `contour_smoothness` gains a `modal`
parameter tolerating P4/P5 leaps (not wider ones) when true, and
`sax_generator` threads `song.modal` into both `modal_strength` (generation)
and `musicality_score` (scoring) from the same place, so the two can never
disagree.

**A corpus-similarity critic is feasible, measured for real, not estimated**
(Phase 28) — the efficiency worry raised discussing it (naive edit-distance-
against-every-corpus-window-per-candidate would be far too slow) turns out to
be a non-issue for the precomputed-frequency-table approach actually proposed.
`wjd_corpus.py` (new, gitignored `wjd_data/` holds the downloaded Weimar Jazz
Database `wjazzd.db` and a derived JSON motif-frequency cache — external
research data, not source) builds a corpus-wide pitch- and duration-motif
`Counter` from all 456 WJD solos (200,809 notes) — pitch via the existing
`extract_interval_motifs`, duration via a new sibling, `ensemble/rhythm_motifs.py`'s
`extract_duration_motifs` (duration-TOKEN n-grams via `dur_to_token`, not
interval deltas — a rhythmic figure's identity is the actual sequence of note
values, not a relative delta, unlike pitch shape). Real numbers: building the
whole table from scratch takes 3.1s (a one-time cost); once built, 20
candidate lookups (matching `motif_recall_candidates`) against a real 8-note
sample chunk's own 33 motifs cost 0.05ms total — effectively free, the same
`Counter`-lookup cost order as every other critic function. This phase is
feasibility/benchmarking only: nothing is wired into `sax_generator`'s real
selection, and the relationship between this corpus-based signal and the
existing rule-based critic — David's own open question, "we'll see what
combination we need" — is deliberately still unresolved.

**The corpus-similarity critic gets a real, narrowly-scoped first use**
(Phase 29) — prompted by a sharp follow-up question: the LSTM was *trained*
on WJD and the corpus table is *built from* WJD, so isn't that the same
information twice? Not quite: the LSTM encodes WJD *implicitly and
generatively* (a diffuse, chord-conditioned sampling distribution); the
corpus table encodes it *explicitly and non-generatively* (exact shape
counts). The two only meaningfully diverge where `sax_generator` already
pushes sampling *off* the model's natural distribution — a recalled
`motif_targets` bias (Phase 17) or `modal_strength` (Phase 27) — since only
there does "did the model produce this" and "does this still look like real
jazz vocabulary" become genuinely different questions. Outside those
interventions, scoring corpus-familiarity would mostly just re-reward what
the LSTM already learned, and risks systematically favouring "sounds like
average WJD" over the deliberate boldness Phases 17-24 pushed generation
*toward*. So: `corpus_familiarity` (`ensemble/critic.py`, standalone, not
part of `MusicalityScore`/`DEFAULT_WEIGHTS`, same precedent as
`motif_adherence`/`dissonance`) only enters `sax_generator`'s selection key
for a chunk where `motif_targets` is non-empty or `song.modal` is true —
verified against real selection on both paths (`tests/test_sax_wolfson_
integration.py`).

The corpus table is now **chord-quality-tagged** (major/dominant/minor/
diminished, matching what the LSTM itself conditions on) — closing the
asymmetry the feasibility phase left open. A real bug was found empirically
before it could cause a silent error: the obvious way to classify a chord
was `ensemble/wolfson/chords.py`'s already-ported `parse_chord`, but checked
directly, `parse_chord("Abj7")` returns dominant, not major — its quality
check looks for the substring `"maj"`, but Jazzomat's own notation marks a
major-seventh with a bare `"j"`. Not a rare case: 10.7% of all 30,548 real
chord annotations use this exact prefix. Per this project's standing rule
(never edit `ensemble/wolfson/*.py`), the fix is a new, combo-authored
classifier in `wjd_corpus.py`, reusing only the `QUAL_*` integer constants
from the ported file, not its logic — a documented limitation worked around,
not fixed in place, same posture as Phase 27's `tonal_conformity` fix.

**A real WJD critic baseline (Phase 30) came back surprising, worth recording
honestly rather than tidied up.** After another listening test ("still very
beginner-noodly... stuck in a narrow range"), the natural check: does
`musicality_score` reward real playing more than combo's own output? Built
`critic_baseline.py` — full chord_idx (root+quality, not Phase 29's
quality-only tagging) via `wjd_corpus.iter_solos_with_chord_idx`/
`split_into_chord_runs` (26,357 real chunks, mean 7.6 notes), a rest-gap
synthesis step so `phrasing()` can score real transcriptions fairly (WJD has
no explicit rest events the way Wolfson's `_inject_rests` produces — a
format gap, not a musical one), scored against real 20-take combo output via
the same critic. Real result: **WJD scored LOWER than combo on 6 of 7
sub-metrics** (`overall` 0.447 vs 0.632) — the opposite of "it should score
well." Not read as "combo plays better than real jazz" — two of the metrics
that favour combo most, `singability`/`phrasing`, have their target
constants (`TARGET_BREATH_FRACTION`, `SINGABLE_DUR_CENTER`/`WIDTH`)
literally calibrated from *combo's own generated output* (Phase 23), so
scoring closer to that target is closer to tautological, not evidence of
quality. `repetition` (WJD 0.29 vs combo 0.65) and `call_response_
relatedness` (0.36 vs 0.51) read as more genuinely informative — combo
repeats internal patterns far more than real solos do, a concrete number
behind the "beginner-noodly" complaint. `tonal_conformity` (WJD 0.71 vs
combo 0.87) is the most concerning, unresolved one: real playing scoring
*lower* on the metric most directly about "in the changes" suggests the WJD
chord classification itself may have real accuracy limits beyond the
already-fixed `parse_chord` bug — checked for coverage and a sane
distribution (Phase 29), never validated against ground truth. `register_
usage` is confounded by chunk granularity (a short chord-idx run can't span
much of any register bound regardless of a whole solo's real range) — WJD
under its own natural pitch bound (44-93) scores lower still (0.16) than
under combo's narrower `SAX_REGISTER` (0.29), the opposite of what "real
players use more range" would predict, a sign the per-chunk measurement
itself needs rethinking before trusting it. Explicitly not yet acted on —
this is a diagnostic result to interpret together, not a finished conclusion
or a set of constants to silently retune.

**Validating chord extraction against an independent signal (Phase 31)
narrowed the `tonal_conformity` question, without fully resolving it.**
Checking the *notes played* against the chord would just re-measure
`tonal_conformity` itself — circular, since a poor fit could mean a wrong
chord label or legitimate outside playing, indistinguishable from that
signal alone (David's own catch). `beats.bass_pitch` sidesteps this: a real
per-beat MIDI note recorded independently of the chord string and of
anything built here, and (per a musicologist David worked with on an
earlier piano-musicality measure) implied bassline is a real signal to
listen for, not a proxy invented for this project. `bass_chord_check.py`
compares our extracted root against it (handling slash chords' explicit bass
override, `_wjd_expected_bass_pc`): 42.3% exact match, 53.7% against any
chord tone, over 28,887 comparable rows — well above chance, far from clean
confirmation. Aggregated by quality class, no outlier (40-46% band across
all four) — argues against a bug concentrated the way Phase 29's
`parse_chord` "j" bug was. But the finer, per-raw-chord-string breakdown
found what the aggregate hid: **`sus` chords and augmented (`+7`) chords
score dramatically worse** (`"Bbsus"` 3.4%, `"Csus"` 4.6%, `"F+7"` 15.2%),
and bare diminished *triads* without a written 7th similarly low (`"Bbo"`
5.6%, `"Abo"` 9.1%). A real, musically-grounded explanation rather than a
parsing bug: augmented and diminished-seventh harmony are intervallically
*symmetric* (an augmented triad's three notes are equally valid "roots" a
major third apart; a diminished seventh's four notes likewise a minor third
apart), so a transcriber's single written root and what the bass actually
sounds can both be musically correct while disagreeing letter-for-letter;
`sus` chords are commonly voiced/used as a substitute for a different
underlying function in practice. This strengthens confidence in the
classifier for ordinary chord types while flagging these specific harmonic
categories as inherently hard to validate this way — and worth remembering
as a genuine limit on how confidently `tonal_conformity` can be trusted for
sus/augmented/symmetric-diminished chords specifically, not just a
data-validation footnote. Still not acted on: no code changed as a result,
pending further investigation (e.g. auditing individual real solos by
ear/lead-sheet).

Register/phrasing/tension-and-resolution (Phases 22-24) are three pieces of a
larger "beginner vs advanced" idea from the same listening-test discussion; side-
slipping and an actual call-site register-narrowing switch (vs. today's single
`register` bound) remain explicitly open. A much bigger, explicitly deferred idea
from the same discussion is a whole-solo *narrative-arc* critic (intro/tension/
resolution across many chunks, not one) — genuinely different in kind, since
every critic function today scores a single chunk in isolation, not a
performance. MIDI input is now unified
(Phase 13): checking a specific request (a director should be able to use the
same interface a performer does, "dual control car") surfaced a general
principle — **role determines destination, not recognition capability**. Checked
directly: `DirectorMidiListener` really did only handle Control Change, ignoring
notes entirely, but `GestureRecognizer`/`MidiListener` never cared who was
playing. `DirectorMidiListener` is retired; `MidiListener` (`input/midi_listener.py`)
is now the one listener type for every source, gaining an optional `cc_number`
alongside its existing recognizer — recognition is uniform, `input/sources.py`
now only varies *where* a source's output is routed by role. This gave
`DirectorSignal.gesture` its first real consumer since the dial channel was built
in Phase 5 (every phase since had repeated some version of "a director-emitted
gesture has nowhere to act"): a new seed gesture, two same-note-repeat runs in a
row (`("S","S")`, not `("T","T")` — checked and confirmed the obvious choice
would have collided with the existing single-`"T"` `reset_tempo()` rule and
never actually fired), toggles whether `ensemble/critic.py`'s `singability`
metric counts toward `RehearsalMemory`'s quality weighting — letting a director
or teacher turn it off live for a student playing fast, exploratory lines that
shouldn't be marked down for being unsustained. `Voice` and `Director` remain
deliberately distinct types (different `Session` contracts — merged timeline
content vs. an aggregated-away signal); only the input-recognition layer
unified. Sax can now search, not just generate once (Phase 14, DESIGN.md §13's
long-deferred "chess" idea) — `n_candidates` generates several candidates per
chunk and keeps the highest-scoring one by `musicality_score`, the evaluator §13
always said this needed. §13's own "poor fit for live performance" reasoning was
never actually checked against real numbers; done directly once there was
something to measure: even 20 candidates over a 4-bar chunk cost ~164ms against
a 7.3-second real-time budget at blues tempo, correcting that assumption rather
than inheriting it — while being honest that this measures computation against
`Session`'s pacing budget, not human-perceived latency in true live
call-and-response, which is a different question this phase doesn't answer.
Role assignment's smallest slice is also built now (Phase 15, §2): when two
same-register voices are both accompanying, `ensemble/roles.py`'s
`default_accompanist_roles` splits them — one full, one laying out — via a
simple greedy register-overlap rule, decided once at ensemble-construction time
(not a live per-bar signal, deliberately, to avoid conflicting with the tested
voice-order-independence guarantee); `comping_generator` gained a `lay_out`
parameter that replaces its usual duck/fill/moderate response with a rare,
low-probability accent. The full tune-level solo/accompany/lay-out/trade
assignment (needing the rest of `ArcController`) is not attempted — this is only
the same-instrument-doubling piece §2 names as a separate, smaller thing.
Phase 16 closes a real gap that had been true since Phase 1: nothing in combo
could actually be heard, only printed as text. §4 already named the fix as
designed-but-not-built ("generation should produce a symbolic timeline...
with playback/scheduling as a separate stage") — `output/midi_output.py`'s
`build_schedule` (pure: `Timeline` + tempo + a voice_id-to-channel map -> a
time-sorted MIDI schedule) and `play_timeline` (real-time playback, reusing
`ensemble/session.py`'s `Clock`/`FakeClock` rather than a new pacing
abstraction) are it. Deliberately much simpler than wolfson's own
`output/midi_output.py`: wolfson continuously interleaves generation and
playback one phrase at a time (needing a dedicated output thread and a
"latest wins" pending-queue); combo's `Session` already generates a whole
multi-voice `Timeline` up front, so the entire schedule is known before
playback starts and no thread coordination is needed — `KeyboardInterrupt`
during `time.sleep()` fires immediately, so a `try/finally` is enough for
all-notes-off cleanup too. `self_test.py` (new, top-level) generalises
wolfson's AI-vs-AI self-play to combo's whole ensemble — bass stand-in, sax,
two role-split comping voices, drums — generated once and played through a
real MIDI output port; `--loop N` shares one `RehearsalMemory` across N
playthroughs, making Phase 11's rehearsal idea audible, not just tested. This
is the first of a three-step testing plan (see README): self-test today (no
MIDI input needed), `listen.py` tomorrow (MIDI input device sanity-check,
already built, no new code), and — named honestly as still missing, not
glossed over — true live rehearsal (a human's live input actually driving the
ensemble's real-time response) needs a "live performance driver" wiring
`input/sources.py` into a running `Session.generate(mode=REAL_TIME)`, which
doesn't exist yet. Once the self-test could actually be heard, David's own
listening turned up two real gaps, both fixed by measuring rather than
re-asserting: `self_test.py`'s bass stand-in (Phase 8's `chord_tone_generator`,
a simultaneous root+fifth double-stop only on beats 1 and 3) sounded thuddy
and staccato through a real bass sample — replaced with `self_test.py`'s own
`walking_bass_stub` (quarter notes, alternating root/fifth, near-full-beat
sustain), local to that script, not the shared/tested `chord_tone_generator`.
And a controlled A/B test (`rehearsal_ab_test.py`, kept as a reusable tool,
not a throwaway script) of whether `RehearsalMemory`'s cross-loop persistence
actually changes anything found a genuine null result at first — traced to
two real causes, not re-measured harder but fixed: `ensemble/critic.py`'s
`repetition()` measures whether a chunk repeats a pattern *within itself*,
with no reference to what memory actually recalled, so a chunk that used the
recalled motif exactly once (without also repeating it again in that same
short chunk) scored `0.0` regardless; and `_apply_motif_bias`
(`phrase_generator.py`) only ever nudges the *next* token once the model has
already spontaneously started matching the target's prefix by chance — rare
for a long (3-4 interval) motif. Phase 17 fixed both, entirely in
combo-authored code, the ported model itself untouched: a new
`motif_adherence` metric (does a candidate's own output actually contain the
recalled target, not just repeat itself), `_pick_achievable_motif` (prefer
the shortest, most achievable recalled motif — grounded directly in
`_apply_motif_bias`'s own prefix-matching logic), a `(motif_adherence,
overall)` selection key (provably identical to Phase 14's overall-only
comparison when nothing's recalled), and `motif_recall_candidates` spending
extra search on chunks that have a real target — checked empirically to fire
on most chunks after the first once memory has anything stored, not the rare
one-off first assumed, though the absolute cost stays fine paid once during
machine_speed generation. Rerunning `rehearsal_ab_test.py` afterward
(20 loops, `motif_recall_candidates=20`) found a clean, construction-clear
effect exactly where it should live: the first plan-chunk of every loop after
loop 0 — the one place cross-loop persistence specifically acts — scores
motif_adherence 1.0 for the persistent condition, 0.0 for a fresh-memory
control, every single time. Whole-run averages are close between conditions
(~1.00 vs ~0.95-0.98), a real, honest side-effect of the same fix rather than
a confound: the mechanism is now reliably audible throughout a run, not only
at rehearsal boundaries. Real listening also turned up a stuck note and
dissonance David flagged directly ("even non-expert audiences can hear when
something is dissonant... the one semitone delta is as bad as it gets in
melodic playing, the dreaded minor 9th") — both fixed (Phase 18). The stuck
note: `output/midi_output.py`'s cleanup only sent CC 123/120 (All Notes
Off/All Sound Off); wolfson's own `output/midi_output.py` already documented
that Logic's software instruments ignore both and need an explicit note_off
per pitch — the same fix, reused, not rediscovered. The dissonance: a new
`out_of_key_check.py` (kept as a reusable tool, same lifecycle as
`rehearsal_ab_test.py`) found ~16-22% of sax notes out of key, and — checked,
not assumed — every single one landed exactly 1 semitone from the scale, the
"minor 9th" clash relationship David named as the worst case, never further
away. `ensemble/critic.py`'s new `dissonance` metric targets exactly that
1-semitone case specifically (a note further from the scale isn't counted —
David's judgment that being "more outside" reads as deliberate, not a clash);
`ensemble/sax.py`'s candidate selection now checks it FIRST, ahead of
motif_adherence and general quality — "what's bad matters a lot," a gate
rather than one more positively-weighted ingredient in `overall`'s blend.
`self_test.py`'s `n_candidates` raised 3→8 to give that gate real candidates
to choose among. Result, reproduced across two separate 5-loop runs: 2.1% out
of key (31/1505, then 32/1525), down from ~16-22% — not zero (the model can
still generate an all-clashing batch, just increasingly rarely as
`n_candidates` grows), but a real, repeatable, ~8-10x reduction. Discussing
that result, David raised a real nuance: plenty of legitimate jazz vocabulary
— 4ths, tritones, maj7 as a bebop passing tone over a dominant chord — is
technically "dissonant" by a plain scale-membership check, and flatly
penalising all of it risks blander soloing, not just cleaner. Of three levers
that came out of that discussion (widen the per-quality scale reference;
an anti-dissonance mode/strength toggle, reusing Phase 13's
`toggle_singability` pattern; tolerate genuine passing tones), David asked to
scope the passing-tone case first, since — unlike the other two — it needs no
new architecture: a direct refinement of `dissonance` itself (Phase 19). New
`_is_passing_tone(real_notes, i)`: a flagged note approached AND left by step
(`PASSING_TONE_MAX_STEP`, a placeholder set to a major 2nd), continuing in the
SAME direction, is excused — the classical tonal-theory treatment of a
dissonance (David's own example: "a chromatically descending bass line is
strong in itself and justifies the one-semitone deltas"). A NEIGHBOUR tone
(approached/left in OPPOSITE directions, e.g. C-D-C) is a related, distinct
device, deliberately not covered — named as a scope-cut, not an oversight, same
as the wider-scales and toggle-mode levers still sitting open. `out_of_key_check.py`
extended to break its report down by passing-tone-vs-clash (reusing
`_is_passing_tone` directly): the out-of-key rate that survives into final
output rose from 2.1% to 3.2-4.2% across two runs — expected, not a
regression, since passing tones are now less penalised in selection — of
which 51.5% and 62.5% respectively were genuine passing tones, a real,
repeatable majority, not unexplained clashes. The other two levers were built
next, together (Phase 20): widening the scale `dissonance` itself judges
against (`_dissonance_scale`, `ensemble/critic.py`) unions the plain
per-quality mode with a named jazz-standard "richer" variant already sitting
in `ensemble/wolfson/scales.py`'s `MODES` table but never wired to anything —
`mixolydian | bebop_dom` for dominant chords (the literal "E natural over F7"
case), `ionian | bebop_major` for major chords (the b6). Checked directly, not
assumed, before building it: `ensemble/sax.py` never actually passes
`scale_pitch_classes` into `PhraseGenerator.generate()` at all (it defaults to
`None`, and the ported sampling loop only applies the bias `if
scale_pitch_classes:`), so this is purely a critic-side accuracy fix — it
doesn't touch generation-time bias or `scales.py` (the ported file) at all.
Minor and diminished have no comparably-named richer variant and are left
unwidened — a named scope-cut, not an oversight (`altered`/`lydian_dom` are
real, available options for a bigger, separate reharm idea, not reached for
here). And an anti-dissonance toggle, reusing Phase 13's `toggle_singability`
director-gesture pattern exactly: two separate rests in a row
(`gesture/vocabulary.py`'s new `("R","R")` rule) flips `ensemble/sax.py`'s new
`dissonance_mode["enabled"]` off/on, checked every bar. Picking the pattern
needed the same care Phase 13's postmortem already established — `"T"`/`"L"`
are unusable anywhere in a new pattern (a 1-length rule's tail check matches
the instant its own symbol arrives, in any position), `("U","U")`/`("D","D")`
are reserved record-marker prefixes, `("S","S")` is taken — `("R","R")` was
verified empirically before committing to it (the same discipline that caught
the `("T","T")` collision originally): fires only on two genuinely separate
rests with nothing between them, not on a rest interrupted by a note or on
ordinary varied playing. `dissonance()` is still computed and logged every
chunk-build regardless of the toggle — only whether it drives selection is
gated. `out_of_key_check.py` updated to use `dissonance_scale` (not the
plain scale) so its own report matches what selection actually judges against
— two more runs after Lever A landed: 4.4% (65/1485) then 1.2% (16/1374),
down from Phase 19's 3.2-4.2%, and the original "E natural over F7" example
no longer appears at all — simply in-scale now, not merely excused. Two more
levers followed (Phase 21), closing the architectural gap named above.
**Tritone/b5 substitution** (Lever D): checked directly before building it,
not assumed — the first instinct, unioning the WHOLE scale of the tritone-
substitute dominant (mirroring Lever A's pattern), saturates the metric
almost completely: F7's own widened scale is 8 notes, its substitute B7's
widened scale is another 8, and their union is all 12 pitch classes, since a
tritone is the most harmonically distant interval and two mixolydian-family
scales that far apart share almost nothing. So `dissonance_scale` (renamed
public — `ensemble/sax.py` now calls it directly, the first time a
`critic.py` helper is needed by production code, not just tests/tooling)
instead tolerates a SINGLE extra pitch class for dominant chords — the
tritone from the root — matching exactly what David named ("a b5
substitution," a specific color tone, not "the whole substitute chord is
valid"). **ii-V-I simplification** (Lever E): new `ensemble/sax.py` functions
`_ii_v_i_target`/`_functional_tonic_scale` check whether the current bar
could be the ii, V, or I of a textbook major ii-V-I (root motion by
descending fifths, qualities minor/dominant/major — Wolfson's own 4-class
mapping), using `Song.chord_at`'s cyclic lookup (verified directly to never
raise, even near a chart's boundary) to look 1-2 bars ahead/behind. If
matched, the target I chord's own `dissonance_scale` is unioned in via a new
`extra_tolerated` parameter on `dissonance()` (default empty, reproducing
Phase 20 exactly for every existing call site). Verified numerically before
trusting it composes safely: D-dorian (the ii of a C ii-V-I) and
C-major-widened differ by exactly one pitch class (the b6) — diatonically
related scales overlap almost entirely, unlike the tritone case, which is
why Lever E can safely union a WHOLE scale while Lever D cannot. A real
selection-behaviour test (spy-and-recompute, not just checking the pure
functions in isolation) confirms the extra tolerance actually reaches
`sax_generator`'s selection over a genuine Dm7-G7-Cmaj7 chart. Explicitly
deferred, named rather than lost: a vi-ii-V-I (four-chord) extension, a
minor-tonic ii-V-i variant, and sub-bar-granular chord sequences (more than
one chord change per bar). All with passing tests.
**Tension-and-resolution crediting** (Phase 22): prompted directly by a
listening-test question — "I can hear the difference between conscious use
of discordant intervals and use due to getting lost, panic, or playing
randomly. I wonder how we could encode that." — generalises the existing
passing-tone exception (Phase 19, which moves THROUGH a dissonance between
two flanking pitches) to a second, distinct device: a clash approached from
an in-scale note (a single isolated reach outward, not mid-excursion) and
resolved by step onto an actual chord tone (`ensemble/critic.py`'s new
`_is_resolved_tension`) — a b9 resolving down a half-step to the root, say.
Unlike every earlier lever, this one is **opt-in**: `dissonance()` gains
`credit_resolved_tension: bool = False`, threaded through as
`sax_generator`'s own parameter of the same name, default off — a
"beginner" default, since unlike a passing tone this isn't universally
uncontroversial, it's the "advanced" behaviour itself. Verified on real
generated output, not assumed: without the flag (5 loops, `blues_in_f.chart`),
only 4.1% (2/49) of what survives selection happens to already look like a
resolved tension by accident; with it, two separate 5-loop runs, 28.3%
(15/53) then 52.4% (22/42) of what's flagged are genuine resolved tensions —
a real, repeatable, substantial share, not a one-off (`out_of_key_check.py`'s
own docstring has the full numbers). Explicitly narrower than it might
sound: covers only a SINGLE isolated tension-then-resolution note, not a
multi-note excursion (genuine side-slipping — a whole pattern shifted a
semitone and back — needs actual generation-time mechanics, not a scoring
exemption); doesn't reward tension use, only stops penalising it once
resolved; not wired to a live director gesture this phase (every practical
same-symbol gesture pattern is already claimed — see `gesture/vocabulary.py`
— a new pattern needs its own design and empirical verification). Two more
adjacent ideas from the same listening-test discussion, deliberately not
attempted here: register range as a skill-level control (beginner stays in
the middle of the instrument's range, advanced uses the full range — a
call-site choice, not a `critic.py`/`sax.py` change, since `register` is
already a real, existing parameter) and phrasing/"speaking in sentences"
with gaps between phrases (a different critic dimension — rest structure
across a chunk — unrelated to dissonance). All with passing tests.
**Not yet built even within these MVPs**: the tune-level solo/accompany/lay-out/
trade role assignment (needs the rest of `ArcController`), same-instrument-
doubling role splitting applied to a voice changing role *over the course of* a
piece rather than fixed for a whole `Session`, and tempo elasticity (§4.1/§4.2)
within the ensemble skeleton; recognising *parameterised* gestures (`handover(target=…)`, `trade(unit=…)`
— the data model can carry params, nothing populates them from raw playing yet) and
§10.3's automatic inference of a genuinely new (non-aliased) taught meaning, within the
gesture vocabulary layer; real swing-timing (triplet-based ride) and generative/soloing
behaviour within drums; "mirrored" builds near arc peaks within accompaniment-listening
(needs a peak/arc signal — the *rest* of `ArcController` — that doesn't exist yet; the
same-register role-split default is now built, see above); within the director, batch-mode scoring (the gesture channel now has its
first consumer, §11/§12, though only one gesture and one voice so far); within MIDI
input, a source feeding more than one destination at once (a performer's gesture
*also* reaching a `DirectorSignal` — representable now, not built), live human
note-capture into the `Timeline` for a `source="human"` `Voice` (a separate, much
bigger, entirely unbuilt capability), `listen.py` becoming a full live-performance
driver that runs an actual `Session.generate(mode=REAL_TIME)` concurrently with
input (found while closing the director-gesture gap: even a *performer's* live
gestures don't reach a running `Session` today, `listen.py` only prints them), the
audience/room-mic path, and any verification against real (non-virtual) hardware;
within transitions, "pulling late" (no gesture for it yet), genuine total-length
shortening, and wiring the director's gesture channel to `TransitionController`
specifically (it now reaches `sax_generator`, not yet transitions); within sax's
real generation, all ~10 of the ported model's OTHER rule-based bias-layer knobs
(contour, energy arc, register contrast, etc. — left at their defaults;
`rhythmic_density` and, via memory, `motif_targets`/`motif_strength` are wired),
hidden-state continuity *between* planned chunks (it now exists *within* one chunk,
which can span several bars, chord-hold permitting — a real extension from Phases
8/9, not a full solve), and any voice besides sax; within the critic, real tuning
of every weight/threshold it uses (`DEFAULT_WEIGHTS`, the contour-smoothness and
near-repeat placeholders — all explicitly unvalidated, same status as every other
hand-picked constant in this codebase); within search-and-evaluate (§13, Phase
14), varying anything besides the random draw across candidates, revision after
committing, and a director-gesture-driven `n_candidates` toggle; within MIDI
playback (Phase 16), any audience/musician display (the terminal dashboard and
web display wolfson had — explicitly deferred, not forgotten), and — the
biggest remaining gap — a live performance driver connecting a human's live
MIDI input to a running, responding `Session`; `self_test.py` only ever plays
back a `Timeline` generated with no human in any role.
See `/Users/davidderoure/.claude/plans/modular-dazzling-emerson.md`
for the full build-order plan across all remaining subsystems.
**Open research questions, not yet answered**: can sub-gesture sequences compose into
a genuine gesture grammar rather than a hand-authored one (§10.3); can the system
develop and recall recurring "tunes" of its own (§1, §4).

## 1. Vision

A self-playing multi-instrument improvising ensemble (bass, sax, keys, drums, roles
TBD) that can perform entirely on its own against a pulse — "like a radio station" —
or interactively with a live human for rehearsal. Successor/sibling to **wolfson**,
David's earlier system (not yet public), which proved trading licks, accompaniment
(emergent, beyond its original spec), and AI-vs-AI self-test all work for one human +
one AI voice trained on WJazzD. combo generalises that to N voices, adds a notion of
structure so the ensemble doesn't just noodle, and asks a genuine research question:
can the system develop — and later recall — recurring "tunes" and even a communicative
"grammar" of its own, rather than only reproducing what it was trained on?

## 2. Voices, not fixed chairs

The architecture is organised around a flexible set of **voices**, not instrument-named
chairs. Each voice has an instrument/register profile and a **source**: human-live
(monophonic pitch-tracked MIDI — see §6) or AI-generated. Any voice can take any role
(solo / accompany / lay out / trade) in any section, assigned by a tune-level form
controller (successor to Wolfson's `ArcController`, promoted from phrase-level to
tune-level).

**Cross-cutting principle (applies to voices here and to directors in §11)**: every
role in combo — performer/voice, director/listener, and any future role — is a set of
N independently-sourced slots, each either human or AI, with no assumption that only
one slot in the whole system is ever human. Multiple simultaneous humans in the same or
different roles is a real prior-tested case, not a hypothetical one: the AGRP concert
setup (§9) already ran five human performers at once, each on their own Sonuus tracker
and MIDI channel. Only the human-facing input plumbing needs to grow to match this —
see the note on `MidiListener` below.

- **Any instrument can accompany, not just solo** — a human can sit in on any chair,
  including an accompanying one (e.g. play bass while an AI voice solos), and the
  system's listening/response logic must be symmetric to that.
- **Same-instrument doubling is supported**: two saxes, two basses — any combination of
  sources (human+AI, human+human, or AI+AI, the last a generalisation of Wolfson's
  self-test mode). When two same-register voices are
  both in an accompanying role at once, the **default is to split the role** — one
  plays full accompaniment, the other lays out or plays sparse punctuation — rather
  than both playing independently at full density and colliding. **Built** (Phase
  15): `ensemble/roles.py`'s `default_accompanist_roles` — a greedy, order-dependent
  register-overlap rule, decided once at ensemble-construction time, not live —
  and `comping_generator`'s `lay_out` parameter. The larger tune-level form
  controller (any voice, any role, any section) named at the top of this section is
  still entirely unbuilt.

## 3. Song as a persistent object

A **song** is changes + form: chord changes, and a form template (chorus length,
section order — e.g. "12-bar blues" or "AABA-32"), plus tempo/feel. MVP is changes+form
only, no composed head melody (a possible later layer). The system can be asked to
"play a song" — recall a stored one, or generate-and-store a new one under a name. This
is deliberately close to IRCAM ImproteK's "scenario" concept (see §11) — a chord chart
as an explicit, reusable, structural object — and is the piece of combo's design with
the most direct published prior art.

## 4. Generation modes, paced by whether a human occupies any role

The real-time constraint on generation is a direct consequence of §2/§11's principle,
not a separately-chosen "mode": it follows from whether a human occupies *any* role at
all — performer or director/listener — not from a fixed enumeration. Three cases fall
out of that rule:

1. **Machine speed — no human in any role.** No real-time pacing at all; the generator
   runs as fast as inference allows. This covers both the **one-shot song generator**
   (generate a single song right now, as fast as possible, then return it — nobody is
   listening live) and generating a large research corpus (the emergent-tunes work,
   §9) or quickly iterating the arc/harmony/role logic — same mechanism, different call
   pattern, not different modes.
2. **Live radio-station self-play** — no human performer, but a human occupies the
   director/listener role (even just "an audience is listening," per Mood Conductor,
   §11) — must pace to real time, because a human ear is receiving it live even though
   nothing is playing back *to* the ensemble.
3. **Interactive rehearsal** — a human performer is present — real-time, reactive.

Architecturally this means generation should produce a symbolic timeline (notes tagged
with beat position/duration, not wall-clock time), with playback/scheduling as a
separate stage — only that later stage needs to know whether a human occupies any role
right now (return the result immediately, or pace to real time).

**The playback stage is built** (Phase 16): `output/midi_output.py`'s `play_timeline`
takes a symbolic `Timeline` and a tempo and paces real MIDI output to real time —
exactly the "separate stage" described above, currently used only for case 1 (machine
speed) via `self_test.py`: generate with no pacing, then play back the whole result
in real time to a synth. Cases 2 and 3 (a human occupying the director or performer
role while the ensemble is live) still need the not-yet-built "live performance
driver" that interleaves this playback stage with concurrent generation and live
input — see the status blurb at the top of this document and README's three-step
testing plan.

### 4.1 Tempo is a runtime value, not a fixed constant

The current MVP (`ensemble/session.py`'s `Session.generate`) paces real-time mode off a
single `song.tempo_bpm` fixed for the whole song — deliberately rigid, since Phase 1
was only about proving the pacing mechanism worked at all, not about musical
elasticity. The intended direction, prompted by David asking how rigid the pulse
should be (combo has a deliberate pulse; Wolfson never did — its timing was always
derived from phrases, never clock-driven): tempo, and tempo nudges (a push, a rit., a
snap back to nominal), should be one of the director's dial parameters (§11), read live
each bar rather than computed once at the start of generation. A tempo gesture —
`reset_tempo()`, alongside §10.1's `handover`/`trade` frames — is then just another
producer writing to that same dial; no new mechanism needed beyond what's already
planned for the director and the seeded vocabulary.

**Three sources can feed the tempo dial**, none needing new architecture beyond what's
already planned: the song's nominal tempo (the default); an explicit gesture
nudge/reset (above); and — prompted by David's rehearsal/teaching question below —
**tempo tracked live from a performer's actual note-onset timing**, extending §5's
accompaniment-listening feature extraction (which already reads live density/register/
dynamics) to also read timing. This is really Wolfson's original beat-sync mechanism
(aligning a proactive phrase to the bassist's actual beat) generalised from a one-off
alignment into a continuously tracked tempo. It's smaller and nearer-term than §4.2's
full free-time stretch goal, and covers a lot of the same practical ground for this
specific case: a learner deliberately playing slowly doesn't need the ensemble to
abandon the clock entirely, just to have the clock's rate genuinely follow them instead
of sitting fixed at the song's nominal tempo.

### 4.2 Free time (stretch goal, captured now so it shapes nearer-term decisions)

A sharper version of the same question: can a stretch of playing come off the pulse
*entirely*, not just have its rate nudged — genuinely phrase-derived timing for a
passage, the way Wolfson always worked, rejoining the pulse later? Not being built now,
but worth capturing because it's a different axis from §4.1 above (that's "is the
clock's rate steady"; this is "is there a clock running at all"), and conflating them
risks shaping the tempo-dial mechanism in a way that can't later accommodate a
genuinely clock-free stretch.

- **Real precedent this isn't unusually ambitious**: ImproteK's real-time architecture
  (§12) already syncs its audio rendering "with a non-metronomic beat" — a live,
  elastic pulse, not a fixed click. This is a solved problem elsewhere, not
  speculative.
- **Likely shape**: a per-section property in the form (§3) — "free time" vs. "in
  time," the same way real charts already notate a rubato intro. A free section
  advances on phrase/gesture completion (closer to how Wolfson actually worked) rather
  than fixed beat-interval pacing; a resolving gesture or structural landmark is what
  snaps the ensemble back onto the pulse — the same handover-style gesture mechanism
  as §8, just triggering a return to clock time instead of a role change.

## 5. Accompaniment-listening

Accompanists extract live features (density, register, dynamics, space/rests) from
whichever voice(s) they're accompanying, and modulate around the planned arc
trajectory — mostly **complementary** (duck density when the soloist is busy, fill in
when they leave space), with occasional **mirrored** builds near arc peaks. Applies
symmetrically regardless of whether the accompanied voice is human or AI. The
same-register role-split default (§2) is this same complementary logic applied
laterally, between two accompanists, not just between accompanist and soloist.

- **Status**: the complementary, accompanist-listens-to-soloist case is built
  (`ensemble/listening.py`, `ensemble/comping.py`), tests passing. Required a real
  architectural extension, not just a new file: every `Generator` now receives a
  snapshot of prior bars (a defensive copy, not the live timeline — see
  `ensemble/session.py`) alongside `(song, bar_index)`, so a voice can actually listen
  to what others have already played. `chord_tone_generator` and `drum_generator` were
  updated to accept (and ignore) the new argument — the "extend on integration" step
  the MVP-per-subsystem plan anticipated.
- **Feature extraction**: all four of density, register (as pitch range), dynamics (as
  average velocity), and space/rests (as beats of silence) are implemented in
  `ensemble/listening.py` — but only density is actually consumed by the comping
  accompanist built here. The other three are extracted and tested for future
  consumers, not yet used by anything. Said plainly rather than left to be discovered.
- **Not built**: "occasional mirrored builds near arc peaks" (needs a peak/arc signal
  — no `ArcController` exists yet).
- **The same-register role-split default is built** (Phase 15, `ensemble/roles.py`,
  `comping_generator`'s `lay_out` parameter — see §2). Deliberately construction-time,
  not a live per-bar signal computed inside `Session.generate`: role-splitting is
  per-voice (unlike `Timeline`/`DirectorSignal`, universal/aggregate across voices),
  and deciding it live would need knowing what other voices are *about* to play this
  same bar — conflicting with the tested voice-order-independence guarantee. So it's
  decided once, at ensemble-construction time, the same pattern as `sax_generator`'s
  `n_candidates`/`memory`/`plan_bars` — no changes to `Generator` or `Session`.
- `ensemble/demo.py` demonstrates the accompanist-listens-to-soloist case against a
  synthetic varying-density fixture (`synthetic_varying_density_generator`), not the
  sax stub — `chord_tone_generator` plays a constant 4 notes every bar with no
  density variation at all, so there'd be nothing for an accompanist to visibly react
  to — and separately demonstrates the role-split case with two overlapping-register
  comping voices.

## 6. Human input: everything through MIDI, no control panels or apps

One principle covers every role: all human interaction with combo — performer,
director, audience — goes through MIDI (a pitch-tracked voice/instrument, or a MIDI
keyboard/controller), never a bespoke control panel or app.

- **Performer**: a monophonic pitch-to-MIDI tracker (e.g. a Sonuus i2M) on bass, sax,
  voice, or any monophonic instrument. Polyphonic/keyboard **chord** input is
  explicitly out of scope here — extracting a musical line from chords is a much
  harder, separate problem (§13). Any human-occupied voice uses this same input path
  regardless of which instrument they're playing.
- **Director**: a MIDI keyboard or controller, read as discrete note/CC values mapped
  to the shared dial parameters (§11). This is a trivial use of keyboard MIDI — reading
  control values, not extracting a melodic line — so it doesn't reopen the chord-input
  exclusion above; the two are different problems that happen to share hardware. No
  dedicated fader/knob box needed.
- **Audience**: no phone app (supersedes the earlier Mood-Conductor-style deferral,
  §13) — a room/ambient microphone through the same pitch-tracker-to-gesture pipeline
  already built for a single performer (§9), treated as one aggregate voice feeding the
  director's aggregation stage (§11). Reuses the existing architecture entirely; the
  audience isn't a new subsystem, just another source for machinery that already
  exists. Still not built — see status below.

- **Status**: performer and director roles are both built (`input/sources.py`) — one
  `MidiListener`/`GestureRecognizer` pair per performer source, one
  `DirectorMidiListener` (reads a Control Change into a live intensity value, `§11`'s
  dial) per director source, dispatched by role from `config.MIDI_SOURCES`. Verified
  further than "logic only, no hardware" originally allowed for: this machine has
  virtual MIDI ports (macOS's IAC Driver), so a real message was actually sent through
  a real (if virtual) MIDI port and received — a genuine note-on produced
  `Gesture("handover")` end-to-end, and a genuine CC message updated a
  `DirectorMidiListener`'s live intensity correctly. Not verified: an actual physical
  tracker (Sonuus or otherwise) or MIDI controller, which this environment doesn't
  have. The automated test suite (`tests/test_midi_sources.py`) stays hardware-
  independent regardless, since the IAC-based check depends on macOS-specific virtual
  MIDI infrastructure not guaranteed present elsewhere (e.g. in CI).
  The **audience** (room-mic) path is not built — same shape as the performer path in
  principle, but nothing wires a room microphone's pitch-tracked output into a
  `MidiSourceConfig`-style source yet.

## 7. Drums

Start as a rule-based pattern engine (brushes, humanised timing/velocity,
section-aware density changes) rather than a trained model, since WJazzD has no drum
data. Generative/soloing drums are an explicit later phase, reusing the trading-licks
engine but substituting kit-voice + density contour for pitch contour as the
"melodic" analogue — a drum solo needs its own stand-in for phrase shape since drums
don't have pitch contour in the same sense a horn line does.

- **Status**: built (`ensemble/drums.py`, `drum_generator`), tests passing, and wired
  into `ensemble/demo.py` as a second voice alongside the sax stub — the first genuine
  multi-voice run of the ensemble.
- **"Brushes," honestly**: a real brush pattern is continuous sweeping texture on the
  snare, not discrete note onsets — `NoteEvent` (pitch + start time + duration) can't
  represent that at all, not even approximately. What's built is brushes-in-spirit,
  using discrete hi-hat/ride/snare hits closer to what a stick-based comping pattern
  would produce. Genuine continuous texture would need a different kind of event than
  `NoteEvent` — a real, structural gap, not a "doesn't sound good yet" one like
  `chord_tone_generator`'s.
- **Section-aware density**, reusing `Song.section_at` (§3) directly, no new machinery
  needed: sparse (hi-hat on 2 and 4 only) under any section named "head" or "out";
  medium (adds a plain-quarters walking ride — real swing ride's triplet-based
  "spang-a-lang" feel is deferred to a real swing-timing pass) as the default; busy
  (adds syncopated snare accents) on the *last* chorus of a multi-chorus section,
  standing in for "approaching a peak" since there's no `ArcController` yet to ask
  directly.
- Humanised via a seeded `random.Random` — deterministic under test, naturally varied
  live.

## 8. Handover / transition triggers

Fixed bar counts alone are good for rehearsal legibility, but listening-driven
transitions were more musically effective in Wolfson's own live-tested experience. The
resolution: bar counts are a **nominal scaffold** (a target section length), and a
**trained gesture vocabulary** is the concrete, learnable trigger that can pull a
transition early or late — closer to how a real band uses a nominal form as a shared
map while letting a cue move the actual boundary.

- **Status**: the "pull early" half is built (`ensemble/transitions.py`,
  `TransitionController`), tests passing — the first real slice of the
  long-referenced `ArcController` (still missing everywhere else it's cited: §5, §7,
  §11), specifically the transition-timing piece, not the full tension/peak-modelling
  concept. Only `handover()` triggers anything — it's the sole seeded gesture whose
  meaning maps onto a transition; `reset_tempo()` is §4.1's tempo dial, untouched
  here. A recognised handover shortens the current section to end after its current
  chorus. Every existing generator (`chord_tone_generator`, `drum_generator`,
  `comping_generator`) picks this up with **zero code changes** — `Session` now
  passes an *effective* (possibly form-truncated) `Song` through the same parameter
  slot they already read `section_at`/`chord_at` from.
- **Not built**: "pulling late" (extending a section) — no seeded gesture means what
  it yet; genuine shortening of the *total* performance length — a handover
  reallocates which section plays when within the same nominal total duration, it
  doesn't end the performance early (the bars a truncated section gives up are
  absorbed into whichever section governs later bars); and the director's gesture
  channel isn't wired to this — `TransitionController` consumes gestures from a
  `Session.gesture_source`, not from `DirectorSignal.gesture` (§11).
- **First thread-safety primitive in the codebase**: `LiveGestureQueue`, needed now
  that a live MIDI callback thread (§6) and `Session.generate`'s loop (real-time
  mode) genuinely run concurrently — a new category of concern, not glossed over.

## 9. Gesture recognition and vocabulary

Built on [AGRP](https://github.com/davidderoure/AGRP), David's 2022 concert-tested,
rule-based gesture recogniser for George Lewis's Voyager/Forager work at PRiSM — proof
that a real, usable gesture vocabulary (runs, rips, staccato calls, trills, flutter
tongue, cresc-dim) can be recognised in real time purely from a monophonic pitch
tracker's event stream (pitch onsets + amplitude), no raw audio/timbral analysis
needed. That directly matches combo's input scope (§6).

- **Status**: the sub-gesture layer (rest / up / down / same / trill / long-note
  detection) is ported to Python — `gesture/recognizer.py`, `SubGestureRecognizer`,
  one instance per voice, tests passing. See [README.md](README.md) for what changed
  in porting (two confirmed bugs fixed, two more found and fixed, one accuracy
  improvement, one preserved-but-flagged behavioural quirk).
- **Composition layer**: `gesture/vocabulary.py`'s `GestureRecognizer` composes
  sequences of sub-gestures into named higher-level gestures (§10.1/§10.2 built —
  the seed vocabulary and the record/alias/pending teaching mechanism; see §10 for
  what's still not built within that). Never finished in AGRP; developed here
  empirically-in-spirit rather than by hand-designing a full rule set up front — the
  seed patterns are explicitly placeholders, same as `chord_tone_generator` was for
  the ensemble skeleton.
- These gestures were general call-and-response dialogue gestures in Voyager, not
  handover-specific — combo still needs to decide which gesture(s), or gesture+rest
  patterns, are handover triggers specifically vs. general responsive cues.
- **Symmetric across human and AI voices, matching §2**: gesture recognition isn't a
  human-only input feature. `SubGestureRecognizer`'s core interface is source-agnostic
  (quartertone note + amplitude events in, sub-gesture events out) — an AI voice's own
  generated output can be fed through the same recogniser as a live human's, so any
  voice can cue any other, exactly as Voyager's mutual gesture recognition intended.
- **PRiSM's alternative ML-based recogniser**
  ([PRiSM-MusicGestureRecognition](https://github.com/rncm-prism/PRiSM-MusicGestureRecognition))
  took a different approach — raw audio, spectrogram features, trained classifier per
  gesture — adopted for later performances after AGRP. It's a poorer fit for combo's
  monophonic-MIDI-only scope than AGRP's approach, which needs no raw audio pipeline.

## 10. Establishing the gesture vocabulary

Three complementary mechanisms, not competing ones — build roughly in this order,
since each rests on the previous one existing rather than starting from nothing.

**Status**: 10.1 and 10.2 are built (`gesture/vocabulary.py`, tests passing) — but only
their argument-less cases; recognising a parameterised gesture's *value*
(`handover(target=bass)`, not just `handover()`) isn't built, since nothing in a raw
sub-gesture stream supplies one yet, and §10.2's "genuinely new meaning" case is built
only as far as storing the recorded-but-unresolved pattern (`pending`) — inferring what
it means automatically is 10.3's territory. 10.3 itself remains fully open, as below.

### 10.1 Seeded

A small starter set of gestures with designer-assigned meanings, present from day one
(as Lewis's Voyager score, or AGRP's hard-coded eight, were). Worth having from the
start, and worth noticing these span two different shapes:

- Plain signals, no argument — e.g. "I'm finishing now, handing over": the form
  controller's existing role-assignment logic picks who receives it.
- **Parameterised frames**, not flat symbols — e.g. "...handing over to a bass
  solo..." is really `handover(target=bass)`; "let's play fours" is
  `trade(unit=fours)` (a proposal to change the *form*, §3/§8, not a handover at all,
  with a parameter on the trading unit — fours, vs. twos, vs. eights). Seeding with a
  handful of small two-slot frames (action + target/parameter) rather than a longer
  flat list gives the emergent-grammar work (§10.3) real compositional structure to
  extend later — directly analogous to the "groups" with roles and an order constraint
  that Steels' agents built up from repeated two-word combinations.
- A **tempo gesture** — `reset_tempo()` at minimum, possibly a push/pull pair — belongs
  in this same seed set (§4.1): it's a plain signal like the handover example above,
  just writing to the director's tempo dial instead of triggering a role change.

### 10.2 Taught

The thing AGRP could have done but never built: a human teaches a new gesture, ahead of
a session or live, entirely through playing — no separate control panel, per §6.

- **Mechanism**: a reserved "begin-record" gesture, followed by the new material,
  followed by a reserved "end-record" gesture — implemented entirely on top of the
  existing `SubGestureRecognizer` pipeline, no new subsystem. Works identically whether
  it happens in a pre-concert teaching session or mid-performance.
- **Teaching by alias**: following end-record immediately with an *existing* known
  gesture means "this new thing means the same as that" — no naming or labelling step
  needed at all, entirely musical.
- **Teaching a genuinely new meaning** (not an alias): recording without a following
  known gesture gives a tightly scoped window ("a new definition is happening right
  now") rather than a label — the meaning still has to be inferred from what happens
  structurally right after, across a few repetitions. Much more tractable than the
  fully unscoped version of this problem (§10.3), because the record markers narrow
  down *when* a new definition is being attempted, not just *that* one might be.
- **Risk to design around**: the begin/end markers need to be gestures unlikely to
  occur by accident during normal playing — the same "wake word" problem voice
  assistants have. Choose something deliberately distinctive for those two, not
  anything already in the working vocabulary.
- The director can teach and cue this way too, not just through the dial channel —
  see §11's note on the two-channel director.

### 10.3 Emergent

Prompted by Luc Steels' language-game work (Steels 1998, 2000 — robot populations
bootstrapping a shared lexicon and then rudimentary grammar through repeated
interaction, no central design, no explicit teaching). The harder, longest-run
mechanism of the three — rests on 10.1 and 10.2 already existing:

- **AGRP's fixed sub-gesture thresholds are already, literally, a hand-built
  discrimination game** (Steels' term for carving a continuous feature space into
  discrete categories, refined only when existing distinctions fail to disambiguate).
  A contained place to introduce real adaptivity later: let thresholds refine based on
  what's proving distinctive in use, instead of staying fixed constants.
- **The unsolved composition-rule layer (§9) maps onto naming games + chunking**:
  start with atomic sub-gestures as meaningful units; log
  (context, sub-gesture(s) played, ensemble response, outcome); let combinations that
  reliably succeed get **chunked** into new reusable named gestures (Steels' term —
  "the backbone of the grammar"). Steels' sharpest transferable idea: grammar/
  composition should only emerge once a flat vocabulary becomes genuinely ambiguous
  relative to what needs expressing — not designed in up front.
- **The hard open problem**: Steels' games work because there's a cheap, checkable
  ground truth (two robot heads watching the same scene). Music has no free equivalent
  for "did this gesture communicate what was intended." This is where the musical
  director (§11) comes in directly — see below.
- **Theoretical framing, useful if this is ever written up**: Steels frames this as a
  "Complex Adaptive Systems" account of language (meaning emerges from repeated
  interaction, never in a steady state), explicitly against Chomskyan Universal Grammar
  (fixed, innate). combo's gesture vocabulary — and its emergent-tunes idea (§1/§4,
  effectively "the same principle applied to whole songs") — both sit in this
  camp. It's a third lineage distinct from the Voyager/Lewis tradition (fixed,
  hand-scored vocabulary) and the IRCAM/DYCI2 tradition (memory-navigation, §12).
  George Lewis himself is independently interested in how gestures might form — worth
  keeping in mind alongside the Steels connection.

## 11. The musical director

A continuously-running critic/director that listens to the **overall ensemble** (not
one voice) and shares the same evaluation logic across two very different consumption
modes: in real-time modes it **nudges** generation as it happens (like a bandleader);
in batch mode the same signal, accumulated over a whole song, becomes a **score** for
curation (generate many candidate songs, pick the best — with automatic filters for
clear rule violations plus human curation on top, rather than trying to fully automate
aesthetic judgment). This generalises three previously-separate ideas into one
component: the tune-level form controller (a planned trajectory), listening-driven
transitions (nudging that plan from live cues), and batch song evaluation.

- **Status**: the dial channel is built end-to-end (`ensemble/director.py`), tests
  passing, with a real consumer — `comping_generator` (§5) actually shifts its
  duck/fill thresholds in response to it, not just accepts and ignores it. The gesture
  channel's *data model and aggregation* are built (`DirectorSignal` can carry a
  `Gesture`, `aggregate_director_signals` handles it) but nothing consumes a
  director-emitted gesture yet — it has nowhere to act until §4.1's runtime tempo and
  §8's handover triggers are code, not just design. Batch-mode scoring and live
  human/MIDI director input aren't attempted at all. `Director` deliberately mirrors
  `Voice`'s shape (`id`, `source`, a per-bar callable) rather than inventing a new
  pattern; `Session` gains a `directors` list alongside `voices`, aggregating one
  signal per bar that every voice's generator receives — this is `ensemble`'s first
  cross-package dependency (`ensemble.director` imports `Gesture` from
  `gesture.vocabulary`), a real integration point, not a smell.
- **AI critic sources are built, human ones aren't — same gap as human `Voice`s**:
  `ensemble_intensity_critic` genuinely listens to the ensemble (averages
  `listening.density()` across named voices) to produce an intensity signal, fulfilling
  §11's "AI critic" language concretely. `constant_director_source` stands in for a
  human at a fixed control position, for tests/demo — live human input follows once §6
  exists, the same relationship human `Voice`s already have to live MIDI input (§2).
- **Teacher is a purpose this role can serve, not a new role**: in rehearsal (§4, case
  3), a director slot can be occupied by, or configured for, teaching. Live tempo-
  tracking (§4.1) is the first concrete mechanism this motivates — the ensemble
  adapting to a learner playing slowly, rather than the learner having to keep up with
  a fixed nominal tempo — and it needs no new architecture beyond what §11 already has.
  Other teaching-specific behaviour (patience, simplified accompaniment, practice
  loops) is a later question, deliberately not specified now.
- **Two channels, not one**: the **dial** (continuous, ambient — intensity, closeness
  to arc resolution) sits alongside the **same gesture vocabulary** every performer
  shares (§9/§10) for discrete, structural decisions — an explicit "let's trade fours"
  or handover cue, not just a gradual nudge. Mirrors how a real bandleader mostly
  shapes energy continuously but occasionally gives an explicit cue. The director is
  just another participant that can emit and recognise gesture events, same channel as
  everyone else (§6) — it isn't a separate mechanism bolted on.
- **Dial nudge mechanism, decided**: dial-based, not directive, for the continuous
  channel specifically. The director adjusts a small set of shared parameters (e.g. an
  intensity/density target, or **tempo** — §4.1) that voice-generators read and
  interpret individually — composes with the per-voice accompaniment-listening model
  (§5) rather than overriding it, and avoids the director becoming a single point of
  micromanagement. (The gesture channel above is deliberately more directive — that's
  fine, it's playing the same role an explicit human cue does.)
- **Also the answer to §10.3's hard problem**: a live director's real-time input is a
  genuine ground-truth preference signal for reinforcing or abandoning gesture-meaning
  pairings — stronger than any proxy we'd otherwise have to invent.
- **Director is itself multi-instance and source-agnostic**, exactly like voices: N
  director "slots," each producing the same signal type, whether the source is a human
  (via §6's MIDI keyboard/controller or room mic) or an AI critic computing it from
  listening to the ensemble. An **aggregation stage** combines however many active
  director signals are present into the single effective dial-state voice-generators
  read — start simple (e.g. a weighted average), designed as a swappable piece so it
  can grow into something richer (e.g. clustering) if this scales toward many
  simultaneous listeners.
- **Real prior art David performed with live**: Mood Conductor (Fazekas, Barthet,
  Sandler, QMUL, ACII 2013) — an audience, via a phone web app, marks a target mood on
  a 2D arousal-valence plane; responses are clustered in real time into "emotion blobs"
  shown to performers as live guidance. "Conductor" is explicitly metaphorical there
  too. Open Symphony (same group) is a related/successor project — audiences vote for
  discrete musical "modes" instead of a continuous position. Worth a citation nod later
  — combo isn't reusing their tech (§6's room-mic mechanism replaces the app entirely)
  but the shape (a listening entity → continuous signal → live nudge) is the same
  lineage.

## 12. Positioning against prior art — IRCAM's OMax/ImproteK/Somax2/Dicy2 family

Researched in depth (papers read in full, not just abstracts) because a colleague
flagged them. **They don't supersede combo.** All four are memory/corpus-navigation
generators (factor oracle, concatenative synthesis, n-gram matching) driven by
continuous machine listening of audio/MIDI features — mechanically and philosophically
different from a trained generative model. Per the Somax2 paper's own framing, that
whole family "does not generate novel material ex nihilo, but recombines and
transforms what it has learned" (cliché re-injection, in ImproteK's own words) — combo
aims at *origination* (its own tunes, its own gesture grammar), not recombination.
None of the four use a discrete trained gesture vocabulary as a communication channel,
and none have combo's self-generation/persistent-tune-memory or musical-director ideas.

That said, real overlap is worth taking seriously, not dismissing:

- **ImproteK's "scenario"** is, in real jazz use, often literally a chord chart (one
  documented session used "Blue in Green"'s chart as the scenario while the memory was
  a recording of a *different* tune, "Autumn Leaves," transposed to fit) — the closest
  published prior art to combo's song-as-changes+form idea (§3), including running
  multiple simultaneous voices (accompaniment/solo) against one scenario. Cite
  directly if §3 is ever written up.
- **ImproteK's real-time architecture** — an offline scenario-guided generation module
  wrapped in a reactive layer that revises its own precomputed anticipations as live
  control events arrive — is a real working precedent for the musical director's
  plan-then-revise shape (§11), just scoped to one voice's harmonic continuity rather
  than an ensemble-level critic.
- **Somax2** is confirmed to run genuine multi-agent networks (documented live use with
  multiple simultaneous interacting instances) — the best existing precedent for
  combo's multi-voice coordination, independent of the differing generative mechanism.

**Decided**: happy to build on IRCAM/DYCI2 work where useful rather than reimplementing
everything from scratch — in particular, `DYCI2/Dicy2-python`'s generative core is a
candidate reusable substrate for some voices, with combo's actual novel contribution
being the layers on top (gesture vocabulary and its possible emergence, ensemble
form/song identity, self-play + emergent tunes, the musical director).

**Resolved for the sax voice** (Phase 8 of the build plan): adapted Wolfson's LSTM,
not `Dicy2-python`. Wolfson's separable generative core (the trained model + its token
vocabulary, `ensemble/wolfson/`) is genuinely generative — origination, matching
combo's own aim, not recombination — and proven live. `Dicy2-python` was ruled out for
this phase on practical grounds, not philosophical ones: it isn't pip-installable
(git-clone-with-submodules, macOS+Python-3.9-only), and it's **GPLv3-licensed**, a real
concern for combo specifically since this is a public repo with no LICENSE file —
pulling in a GPLv3 dependency would effectively force combo's own licensing. It remains
a documented, deferred candidate for a **future comping voice** specifically
(recombination/corpus-navigation fits accompaniment's idiomatic-vocabulary-reuse
better than a soloed line) — not attempted now. Which voices beyond sax would use
either substrate, if any, is still a sketching-phase question.

## 13. Explicitly out of scope (for now)

- ~~**Lookahead/search generation**~~ **Built, Phase 14**: `sax_generator`'s
  `n_candidates` parameter generates several candidates per chunk (identical
  arguments — the model's own RNG diversifies successive calls) and keeps the
  highest-scoring one by `ensemble/critic.py`'s `musicality_score`, the evaluator
  this item always said it needed. The "poor fit for live performance" reasoning
  below was never actually checked against real numbers — done directly once
  there was something to measure: even 20 candidates over a full 4-bar chunk
  costs ~164ms against a 7.3-second real-time budget at blues tempo, so this
  isn't restricted to `machine_speed` after all. What's still genuinely
  unverified: that's computation time against `Session`'s nominal per-bar
  pacing budget, not human-*perceived* latency in true live call-and-response —
  a different, more subtle question this phase doesn't answer. Also not
  attempted: varying anything besides the random draw across candidates
  (temperature, `rhythmic_density` — searching generation *parameters*, a
  larger idea than resampling fixed ones); revision after committing (ImproteK's
  actual architecture, §12); a director-gesture-driven `n_candidates` toggle
  (the natural next use of Phase 13's `critic_weights`-mutation pattern, not
  built).
- **Polyphonic/keyboard chord input**, for the performer role specifically (§6) — the
  director's use of a MIDI keyboard is a different, much simpler problem and is in
  scope.
- **Composed head melodies** as part of a song object (§3) — changes+form only for now.

## 14. Nods owed, not yet written

Small acknowledgments to fold in once there's something being written up, not
structural influences on the design itself:

- SHRDLU (Winograd) — David visited MIT in the 80s; a nod, not a design input.
- George Lewis's Voyager/Forager — direct lineage for §9, not just a nod.
- Steels' language games — direct design input for §10, not just a nod.
- Mood Conductor / Open Symphony — direct design input for §11 (§12 above).
