"""Live MIDI in, prints detected sub-gestures as they arrive.

    python listen.py --list        # show available MIDI input ports
    python listen.py --port 0      # listen on port 0 (Ctrl-C to stop)
"""

import argparse
import threading
import time

from config import MIDI_INPUT_PORT, PITCH_BEND_RANGE
from gesture.recognizer import SubGesture, SubGestureRecognizer
from input.midi_listener import MidiListener


def on_subgesture(sg: SubGesture) -> None:
    print(f"{sg.label:>2}  start={sg.start_time:8.2f}  dur={sg.duration:5.2f}  n={sg.n}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=MIDI_INPUT_PORT)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for i, name in enumerate(MidiListener.list_ports()):
            print(f"{i}: {name}")
        return

    recognizer = SubGestureRecognizer(on_subgesture=on_subgesture, pitch_bend_range=PITCH_BEND_RANGE)
    listener = MidiListener(recognizer)
    listener.start(args.port)

    stop = threading.Event()

    def tick_loop() -> None:
        while not stop.is_set():
            recognizer.tick(time.time())
            time.sleep(0.02)

    threading.Thread(target=tick_loop, daemon=True).start()

    print("Listening for sub-gestures... (Ctrl-C to stop)")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        listener.stop()


if __name__ == "__main__":
    main()
