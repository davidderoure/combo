"""Live MIDI in from every role-tagged source in config.MIDI_SOURCES: recognised
gestures from performers, live intensity from directors — DESIGN.md §6.

    python listen.py --list        # show available MIDI input ports
    python listen.py               # start every source in config.MIDI_SOURCES

No MIDI hardware was available to verify this end-to-end while building it — the
config/dispatch logic is unit tested (tests/test_midi_sources.py), but actually
running this script against real devices hasn't been.
"""

import argparse
import threading
import time

from config import MIDI_SOURCES, PITCH_BEND_RANGE
from gesture.vocabulary import Gesture
from input.midi_listener import MidiListener
from input.sources import start_midi_sources


def on_gesture(source_id: str, gesture: Gesture) -> None:
    print(f"[{source_id}] gesture: {gesture}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for i, name in enumerate(MidiListener.list_ports()):
            print(f"{i}: {name}")
        return

    sources = start_midi_sources(MIDI_SOURCES, on_gesture=on_gesture, pitch_bend_range=PITCH_BEND_RANGE)

    stop = threading.Event()

    def tick_loop() -> None:
        while not stop.is_set():
            now = time.time()
            for listener in sources.performers.values():
                listener.recognizer.tick(now)
            time.sleep(0.02)

    def status_loop() -> None:
        while not stop.is_set():
            for source_id, listener in sources.directors.items():
                print(f"[{source_id}] intensity: {listener.intensity:.2f}")
            time.sleep(1.0)

    threading.Thread(target=tick_loop, daemon=True).start()
    if sources.directors:
        threading.Thread(target=status_loop, daemon=True).start()

    print("Listening... (Ctrl-C to stop)")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        sources.stop_all()


if __name__ == "__main__":
    main()
