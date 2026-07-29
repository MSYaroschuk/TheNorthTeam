#!/usr/bin/env python3

# encoder_and_pid_msy.py
# Closed-loop flywheel speed control for the shooter.
#
# (C) Team Northeast
#
# Commands, typed while running:
#   2000      spin up to 2000 RPM        0    stop
#   kp 0.003  set a gain live            ?    show current settings
#   kd 0.02                              q    quit
#   ramp 0.8  throttle change limit
#   db 20     ignore errors below this
#
# Gain changes are not written back to this file.

import math
import sys
import threading
import time
from collections import deque
from queue import Queue

from fusion_hat.servo import Servo
from fusion_hat.pin import Pin

# ==========================================================================
# HARDWARE
# ==========================================================================

MOTOR_CHANNEL = 2
CHANNEL_A = 17
CHANNEL_B = 4

# The ESC stops at NEUTRAL_ANGLE, not at 0, and DIRECTION selects which side of
# neutral drives the shooting direction. DIRECTION must never multiply the
# neutral value itself or "stop" becomes a throttle command the other way:
#
#     angle = NEUTRAL_ANGLE + DIRECTION * throttle       throttle 0 = stopped
#
# Consistent with the chassis code, where 5 + (-1 * 25) = -20 is full speed.
NEUTRAL_ANGLE = 5.0
DIRECTION = -1
MAX_THROTTLE = 25.0

# ==========================================================================
# SPEED MEASUREMENT
# ==========================================================================

# Counts registered per revolution. encoder_thread() counts twice per
# quadrature cycle, so 16 implies 8 cycles per revolution.
#
# STILL UNCONFIRMED: measured over a single hand-turned revolution. Verify over
# ten (expect 160). If the true figure is lower than this, RPM under-reports and
# the shooter can exceed 12.0 m/s while the display still looks legal.
PULSES_PER_REV = 16

# Speed comes from the time spanned by this many pulses, not from counting
# pulses in a fixed window. Counting quantises to (60/PULSES_PER_REV)/window,
# which was 18.75 RPM steps - coarser than the accuracy the shot needs.
#
# The limit on the timing method is scheduler jitter in the encoder thread,
# about 1 ms, not the poll clock. Since that error is roughly fixed in time, a
# longer window dilutes it: 64 pulses spans ~120 ms at 2000 RPM and holds the
# noise near 16 RPM.
RPM_WINDOW_PULSES = 64
RPM_STALE_SECONDS = 0.3      # no pulses this long means the wheel has stopped

# ==========================================================================
# CONTROL
# ==========================================================================

# This controller is INCREMENTAL - throttle accumulates the output - so the
# terms do not do what their names suggest:
#
#     KP term  adds to throttle each tick     -> acts as INTEGRAL action
#     KD term  responds to change in error    -> acts as PROPORTIONAL action
#
# There is deliberately no I term. Integrating an already-integrating output is
# a double integral, and it drives the flywheel far past target.
KP = 0.0020
KD = 0.0200

# Limit on throttle change per control update. This sits after the PID, so it
# caps how fast the wheel can spin up regardless of the gains.
MAX_ANGLE_CHANGE = 0.8

# Ignore errors below this. Sits just above the measurement noise floor so the
# controller does not chase noise.
ERROR_DEADBAND = 20.0

CONTROL_INTERVAL_SECONDS = 0.1

# Measured plant: rpm = 340 * throttle - 1544. The ESC produces nothing below
# about throttle 4.5; everything under that is dead travel. Feedforward uses
# this so the controller starts near the right throttle instead of walking up
# from zero and leaving the integral to hold the whole operating point.
#
# RE-MEASURE after any change to the battery, ESC, wheel or belt: command a few
# speeds, note the settled throttle, fit a line.
RPM_PER_THROTTLE = 340.0
THROTTLE_DEADBAND = 4.54

MAX_TRIM = 4.0               # how far the PID may pull off the feedforward

# ==========================================================================
# SAFETY
# ==========================================================================

# Rule 5.5: tennis balls may not be launched above 12.0 m/s.
FLYWHEEL_DIAMETER_IN = 3.0
FLYWHEEL_RADIUS_M = FLYWHEEL_DIAMETER_IN * 0.0254 / 2.0
MAX_LEGAL_RPM = (12.0 / FLYWHEEL_RADIUS_M) * 60.0 / (2.0 * math.pi)

STALL_SECONDS = 3.0          # commanded but not turning -> cut power

# An under-reporting encoder makes the controller see a slow wheel and keep
# adding throttle; the wheel accelerates while the display stays calm, and the
# stall guard cannot catch it because the wheel is turning. Bounding the
# throttle by what the target could plausibly need is the backstop. Headroom
# applies only above the ESC's dead travel.
THROTTLE_HEADROOM = 1.6

# ==========================================================================

motor = Servo(MOTOR_CHANNEL)
pin_a = Pin(CHANNEL_A, mode=Pin.IN)
pin_b = Pin(CHANNEL_B, mode=Pin.IN)

encoder_lock = threading.Lock()
pulse_times = deque(maxlen=RPM_WINDOW_PULSES)
pulse_count = 0
poll_count = 0

rpm = 0.0
target_rpm = 0.0             # starts stopped; nothing spins until asked
throttle = 0.0
trim = 0.0
previous_error = 0.0
input_queue = Queue()


def clamp(value, low, high):
    return max(low, min(high, value))


def write_throttle(value):
    motor.angle(NEUTRAL_ANGLE + DIRECTION * clamp(value, 0.0, MAX_THROTTLE))


def stop_motor():
    motor.angle(NEUTRAL_ANGLE)


def feedforward_throttle(target):
    if target <= 0:
        return 0.0
    return THROTTLE_DEADBAND + target / RPM_PER_THROTTLE


def muzzle_mps(speed_rpm):
    return (speed_rpm * 2.0 * math.pi / 60.0) * FLYWHEEL_RADIUS_M


stop_motor()


# ==========================================================================
# ENCODER
# ==========================================================================

def encoder_thread():
    """Poll the encoder on a dedicated thread.

    This must not share a thread with printing or sleeping. Catching every edge
    needs a poll faster than channel A's high time - 1067 Hz at 2000 RPM, 1600
    at the legal ceiling - and a loop doing anything else falls below that.
    Dropped pulses read as a slow wheel, so the controller adds throttle, which
    drops more pulses. It runs away, and no gain change can fix it.
    """
    global pulse_count, poll_count

    last_a = pin_a.value()
    last_b = pin_b.value()

    while True:
        a = pin_a.value()
        b = pin_b.value()
        poll_count += 1

        if (a != last_a or b != last_b) and a == 1:
            with encoder_lock:
                pulse_count += 1
                pulse_times.append(time.monotonic())

        last_a, last_b = a, b


def read_rpm():
    with encoder_lock:
        if len(pulse_times) < 2:
            return 0.0
        stamps = list(pulse_times)

    if time.monotonic() - stamps[-1] > RPM_STALE_SECONDS:
        return 0.0
    span = stamps[-1] - stamps[0]
    if span <= 0:
        return 0.0
    return ((len(stamps) - 1) / PULSES_PER_REV) / span * 60.0


# ==========================================================================
# COMMANDS
# ==========================================================================

TUNABLES = ("kp", "kd", "ramp", "db")


def read_commands():
    while True:
        try:
            line = input().strip()
        except EOFError:
            break
        if line:
            input_queue.put(line)


def show_settings():
    print(f"\n  kp {KP:g}   kd {KD:g}   ramp {MAX_ANGLE_CHANGE:g}   "
          f"db {ERROR_DEADBAND:g}   target {target_rpm:.0f} RPM")


def handle_commands():
    global target_rpm, KP, KD, MAX_ANGLE_CHANGE, ERROR_DEADBAND

    while not input_queue.empty():
        line = input_queue.get().strip().lower()
        if not line:
            continue

        if line in ("?", "h", "help"):
            show_settings()
            continue
        if line == "q":
            raise KeyboardInterrupt

        parts = line.split()
        if len(parts) == 1:
            for key in TUNABLES:                    # accept "kd0.02" too
                if parts[0].startswith(key) and len(parts[0]) > len(key):
                    parts = [key, parts[0][len(key):]]
                    break

        if len(parts) == 2 and parts[0] in TUNABLES:
            try:
                value = float(parts[1])
            except ValueError:
                print(f"\n  '{parts[1]}' is not a number")
                continue
            if parts[0] == "kp":
                KP = value
            elif parts[0] == "kd":
                KD = value
            elif parts[0] == "ramp":
                MAX_ANGLE_CHANGE = max(0.0, value)
            elif parts[0] == "db":
                ERROR_DEADBAND = max(0.0, value)
            show_settings()
            continue

        try:
            requested = float(line)
        except ValueError:
            print(f"\n  '{line}' not understood - type a number, 'kd 0.02', or '?'")
            continue

        if requested > MAX_LEGAL_RPM:
            print(f"\n  capping {requested:.0f} -> {MAX_LEGAL_RPM:.0f} RPM "
                  f"(12.0 m/s, rule 5.5)")
            requested = MAX_LEGAL_RPM
        target_rpm = max(0.0, requested)


# ==========================================================================
# MAIN
# ==========================================================================

def main_loop():
    global rpm, throttle, trim, previous_error, target_rpm

    print(f"Motor STOPPED until you type a target. Legal max "
          f"{MAX_LEGAL_RPM:.0f} RPM.")
    print("Commands: <rpm> | 0 | kp 0.003 | kd 0.02 | ramp 0.8 | db 20 | ? | q")

    threading.Thread(target=encoder_thread, daemon=True).start()
    threading.Thread(target=read_commands, daemon=True).start()

    last_control = time.monotonic()
    last_motion = time.monotonic()
    last_print = time.monotonic()
    last_poll_count, last_poll_time, poll_hz = 0, time.monotonic(), 0.0

    while True:
        now = time.monotonic()
        rpm = read_rpm()
        if rpm > 5.0:
            last_motion = now

        if now - last_control >= CONTROL_INTERVAL_SECONDS:
            if target_rpm <= 0:
                throttle = trim = previous_error = 0.0
                last_motion = now
                stop_motor()
            else:
                error = target_rpm - rpm
                if abs(error) < ERROR_DEADBAND:
                    error = 0.0
                step = clamp(KP * error + KD * (error - previous_error),
                             -MAX_ANGLE_CHANGE, MAX_ANGLE_CHANGE)
                previous_error = error
                trim = clamp(trim + step, -MAX_TRIM, MAX_TRIM)

                ceiling = min(
                    MAX_THROTTLE,
                    THROTTLE_DEADBAND
                    + (target_rpm / RPM_PER_THROTTLE) * THROTTLE_HEADROOM,
                    THROTTLE_DEADBAND
                    + (MAX_LEGAL_RPM / RPM_PER_THROTTLE) * 1.25)
                throttle = clamp(feedforward_throttle(target_rpm) + trim,
                                 0.0, ceiling)

                if now - last_motion > STALL_SECONDS:
                    print(f"\n  !! {target_rpm:.0f} RPM commanded but nothing "
                          f"turning for {STALL_SECONDS:.0f}s - stopping. Motor "
                          f"not spinning, or encoder not reading.")
                    target_rpm = throttle = trim = previous_error = 0.0
                    stop_motor()
                else:
                    write_throttle(throttle)

            last_control = now

        handle_commands()

        # 5 Hz. Printing every pass starves the encoder thread.
        if now - last_print >= 0.2:
            elapsed = now - last_poll_time
            if elapsed > 0:
                poll_hz = (poll_count - last_poll_count) / elapsed
            last_poll_count, last_poll_time, last_print = poll_count, now, now
            sys.stdout.write(
                f"\rpulses {pulse_count:7d} | RPM {rpm:6.0f} / {target_rpm:<6.0f}"
                f" | {muzzle_mps(rpm):5.2f} m/s | throttle {throttle:5.2f}"
                f" | trim {trim:+5.2f} | poll {poll_hz / 1000:4.1f}kHz   ")
            sys.stdout.flush()

        time.sleep(0.002)


try:
    main_loop()
except KeyboardInterrupt:
    print("\nExiting...")
finally:
    # finally, not just the KeyboardInterrupt handler: any crash must stop the
    # motor too.
    stop_motor()
    print("Motor commanded to neutral.")
