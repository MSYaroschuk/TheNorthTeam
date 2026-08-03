#!/usr/bin/env python3
# motor_check.py - which wheel spins which way?
#
# Drives ONE channel at a time so you can watch each wheel on its own. The
# autonomous code drives all four together, where a backwards motor just looks
# like "it curves" and you cannot tell which one is at fault.
#
# WHEELS OFF THE GROUND.

from time import sleep
from fusion_hat.servo import Servo

MOTOR_NEUTRAL = 5
MOTOR_MAX_THROTTLE = 20
TEST_SPEED = 0.3          # same as the intake creep
HOLD_S = 1.5

# (label, channel, invert-flag-name) - mirrors auto_chassis.py
MOTORS = [
    ("LEFT  front", 0, "INVERT_LEFT_1"),
    ("LEFT  rear ", 8, "INVERT_LEFT_2"),
    ("RIGHT front", 1, "INVERT_RIGHT_1"),
    ("RIGHT rear ", 9, "INVERT_RIGHT_2"),
]


def angle_for(speed):
    return MOTOR_NEUTRAL + (speed * MOTOR_MAX_THROTTLE)


def main():
    print(__doc__)
    if input("Wheels off the ground? type yes: ").strip().lower() != "yes":
        print("aborted")
        return

    servos = {ch: Servo(ch) for _, ch, _ in MOTORS}
    for s in servos.values():
        s.angle(MOTOR_NEUTRAL)
    sleep(0.5)

    print(f"\nDriving each motor at +{TEST_SPEED} for {HOLD_S}s.")
    print("Note which ones spin BACKWARDS (against robot-forward).\n")

    results = []
    try:
        for label, ch, flag in MOTORS:
            input(f"  [enter] to run {label} (channel {ch})")
            servos[ch].angle(angle_for(TEST_SPEED))
            sleep(HOLD_S)
            servos[ch].angle(MOTOR_NEUTRAL)
            ans = input("      forward or backward? [f/b]: ").strip().lower()
            results.append((label, ch, flag, ans.startswith("b")))
            print()
    finally:
        for s in servos.values():
            s.angle(MOTOR_NEUTRAL)
        print("all channels neutral")

    print("\n--- set these in auto_chassis.py ---")
    for label, ch, flag, backwards in results:
        # A motor that ran BACKWARDS needs its flag flipped from whatever
        # auto_chassis.py currently has. Right-side flags default to True.
        current = flag.startswith("INVERT_RIGHT")
        needed = (not current) if backwards else current
        mark = "   <-- CHANGED" if needed != current else ""
        print(f"{flag} = {needed}{mark}")


if __name__ == "__main__":
    main()
