"""Synthetic-event tests for SubGestureRecognizer — no MIDI hardware needed."""

from gesture.recognizer import SubGesture, SubGestureRecognizer


def make_recognizer():
    detected: list[SubGesture] = []
    recognizer = SubGestureRecognizer(on_subgesture=detected.append)
    return recognizer, detected


def play_run(recognizer, pitches, start=0.0, step=0.08, hold=0.05):
    """Play a monophonic sequence of quartertone pitches, each held `hold`
    seconds and spaced `step` seconds apart, starting at `start`."""
    t = start
    for pitch in pitches:
        recognizer.note_on(pitch, 80, t)
        recognizer.note_off(pitch, t + hold)
        t += step
    return t


def test_ascending_run_detected_as_up():
    recognizer, detected = make_recognizer()
    # 6 consecutive rising quartertone steps, well past min_up_duration (0.2s)
    play_run(recognizer, [60, 61, 62, 63, 64, 65, 66])

    up_events = [sg for sg in detected if sg.label == "U"]
    assert up_events, f"expected a 'U' sub-gesture, got: {detected}"


def test_descending_run_detected_as_down():
    recognizer, detected = make_recognizer()
    play_run(recognizer, [80, 79, 78, 77, 76, 75, 74])

    down_events = [sg for sg in detected if sg.label == "D"]
    assert down_events, f"expected a 'D' sub-gesture, got: {detected}"


def test_long_held_note_detected_as_long():
    recognizer, detected = make_recognizer()
    recognizer.note_on(60, 80, 0.0)
    # no note_off — held well past min_long_duration (2.0s default)
    recognizer.tick(2.5)

    long_events = [sg for sg in detected if sg.label == "L"]
    assert long_events, f"expected an 'L' sub-gesture, got: {detected}"


def test_rest_detected_after_silence():
    recognizer, detected = make_recognizer()
    recognizer.note_on(60, 80, 0.0)
    recognizer.note_off(60, 0.1)

    # advance past rest_confirm_duration (0.4s) + min_rest_duration (0.1s)
    for t in [0.15, 0.3, 0.45, 0.6, 0.7]:
        recognizer.tick(t)

    rest_events = [sg for sg in detected if sg.label == "R"]
    assert rest_events, f"expected an 'R' sub-gesture, got: {detected}"


def test_trill_duration_is_bounded_not_since_epoch():
    """Regression test for the fixed trillStartTime bug: a trill's reported
    duration should be close to the trill's actual elapsed time, not the
    time since the recognizer was created."""
    recognizer, detected = make_recognizer()
    # oscillate within trill_tolerance for >10 notes, each spaced 0.03s apart
    t = 100.0  # start recognizer "session" partway through, like a long-running process
    pitches = [60, 62] * 8
    for pitch in pitches:
        recognizer.note_on(pitch, 80, t)
        recognizer.note_off(pitch, t + 0.02)
        t += 0.03

    trill_events = [sg for sg in detected if sg.label == "T"]
    assert trill_events, f"expected a 'T' sub-gesture, got: {detected}"
    for sg in trill_events:
        assert sg.duration < 5.0, (
            f"trill duration {sg.duration} looks like elapsed-since-start rather "
            "than elapsed-since-trill-onset"
        )
