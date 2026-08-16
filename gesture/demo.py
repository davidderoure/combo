"""Manual sanity check for the gesture vocabulary layer (DESIGN.md §10) — no MIDI or
hardware needed, continuing AGRP's own replayed-test-gesture convention.

    python -m gesture.demo
"""

from gesture.recognizer import SubGesture
from gesture.vocabulary import GestureRecognizer


def on_gesture(gesture) -> None:
    print(f"  -> gesture recognised: {gesture}")


def demo_seed_gestures() -> None:
    print("Seed gestures (§10.1) — real note-level playing:")
    vocab = GestureRecognizer(on_gesture=on_gesture)

    print("  playing a long held note...")
    vocab.note_on(60, 80, 0.0)
    vocab.tick(2.5)  # past min_long_duration (2.0s) with no note_off -> "L"

    print("  playing a trill...")
    vocab2 = GestureRecognizer(on_gesture=on_gesture)
    t = 100.0
    for pitch in [60, 62] * 8:
        vocab2.note_on(pitch, 80, t)
        vocab2.note_off(pitch, t + 0.02)
        t += 0.03


def demo_record_and_teach() -> None:
    print("\nRecord + alias teaching (§10.2) — fed at the sub-gesture level for a")
    print("reliable, reproducible demo (note-level choreography for exact multi-run")
    print("sub-gesture sequences is timing-sensitive; see tests/test_recognizer.py):")

    vocab = GestureRecognizer(on_gesture=on_gesture)

    def feed(labels):
        for i, label in enumerate(labels):
            vocab.feed_subgesture(SubGesture(label=label, start_time=float(i), duration=0.1, n=1))

    print("  begin-record (U,U), play 'S,S,L', end-record (D,D)...")
    feed(["U", "U", "S", "S", "L", "D", "D"])
    print(f"  taught: {vocab.rules[-1].pattern} -> {vocab.rules[-1].gesture}")

    print("  replaying the taught pattern 'S,S,L'...")
    feed(["S", "S", "L"])

    print("\n  begin-record (U,U), play 'S,R' (no known ending), end-record (D,D)...")
    feed(["U", "U", "S", "R", "D", "D"])
    print(f"  pending (unresolved, not guessed): {vocab.pending}")


if __name__ == "__main__":
    demo_seed_gestures()
    demo_record_and_teach()
