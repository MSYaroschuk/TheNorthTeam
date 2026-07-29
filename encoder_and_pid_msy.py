#!/usr/bin/env python3

# encoder_and_pid_msy.py
# Closed-loop flywheel speed control for the shooter.
# (C) Team Northeast
#
# Type while running:
#   2000  target RPM      kp 0.003 / kd 0.02 / ramp 0.8 / db 20  set a gain
#   0     stop            ?  show settings        q  quit
# Gain changes are not saved back to this file.

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

# angle = NEUTRAL_ANGLE + DIRECTION * throttle, so throttle 0 is a real stop.
# DIRECTION must never multiply NEUTRAL_ANGLE or "stop" becomes throttle the
# other way. Matches the chassis code: 5 + (-1 * 25) = -20 is full speed.
NEUTRAL_ANGLE = 5.0
DIRECTION = -1
MAX_THROTTLE = 25.0

# ==========================================================================
# SPEED MEASUREMENT
# ==========================================================================

# UNCONFIRMED - measured over one hand-turned revolution. Verify over ten
# (expect 160). If the true value is lower, RPM under-reports and the shooter
# can pass 12.0 m/s while the display still looks legal.
PULSES_PER_REV = 16

# Speed is the time spanned by this many pulses. Counting pulses in a fixed
# window instead would quantise to 18.75 RPM steps. The limit here is ~1 ms of
# scheduler jitter, not the poll clock, so a longer window dilutes it: 64
# pulses spans ~120 ms at 2000 RPM and holds noise near 16 RPM.
RPM_WINDOW_PULSES = 64
RPM_STALE_SECONDS = 0.3

# ==========================================================================
# CONTROL
# ==========================================================================

# Throttle accumulates the output, so the terms are not what their names say:
# KP acts as integral action, KD as proportional. There is no I term on
# purpose - integrating an integrating output drives the wheel past target.
KP = 0.0020
KD = 0.0200

MAX_ANGLE_CHANGE = 0.8       # throttle change per update; caps spin-up rate
ERROR_DEADBAND = 20.0        # sits just above the measurement noise floor
CONTROL_INTERVAL_SECONDS = 0.1

# Measured: rpm = 340 * throttle - 1544. Nothing turns below throttle ~4.5.
# Feedforward uses this so the controller starts near the right throttle.
# Re-measure after any battery, ESC, wheel or belt change.
RPM_PER_THROTTLE = 340.0
THROTTLE_DEADBAND = 4.54
MAX_TRIM = 4.0               # how far the PID may pull off the feedforward

# ==========================================================================
# SAFETY
# ==========================================================================

FLYWHEEL_DIAMETER_IN = 3.0
FLYWHEEL_RADIUS_M = FLYWHEEL_DIAMETER_IN * 0.0254 / 2.0
MAX_LEGAL_RPM = (12.0 / FLYWHEEL_RADIUS_M) * 60.0 / (2.0 * math.pi)  # rule 5.5

# BENCH TESTING - targets are capped here, not at MAX_LEGAL_RPM. Rule 5.5 caps
# competition launches at 12.0 m/s; anything above MAX_LEGAL_RPM is bench only
# and the status line marks it. SET THIS BACK TO MAX_LEGAL_RPM BEFORE THE MATCH.
#
# Still bounded rather than open: this value is also what stops a failing
# encoder from driving the wheel to free speed.
MAX_TARGET_RPM = 6000.0

STALL_SECONDS = 3.0          # commanded but not turning -> cut power

# A failing encoder reads slow, so the controller adds throttle and the wheel
# runs away while the display stays calm. Capping throttle at what the target
# could plausibly need is the backstop; headroom applies above the dead travel.
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
target_rpm = 0.0             # starts stopped
throttle = 0.0
trim = 0.0
previous_error = 0.0
input_queue = Queue()
pending_event = ""           # written into the next log row, then cleared

LOG_RATE_HZ = 50             # fine enough to resolve the dip as a ball passes

# A ball passing through the flywheel steals energy and the speed drops
# sharply. That dip is the true instant of the shot, so it is detected here
# rather than relying on how fast you can type - your label afterwards just
# attaches to the most recent detected shot.
# Compared as the mean of the older half of the window against the newer half.
# Comparing a peak against a single sample instead triggers constantly on
# +-16 RPM noise; averaging cuts that scatter to about 6 RPM, so 25 sits well
# clear of it. Tested against synthetic traces: every ball found, no false
# positives, and none in two minutes of noise with no shots at all.
SHOT_DIP_RPM = 25.0          # at 2000 RPM; scaled up with speed, see below
SHOT_DEBOUNCE_S = 1.5        # a real dip can outlast 0.8 s and re-trigger

recent_rpm = deque(maxlen=30)    # ~0.6 s of history at LOG_RATE_HZ
last_shot_time = 0.0
last_shot_rpm = 0.0


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
    """Poll the encoder. Must not share a thread with printing or sleeping:
    catching every edge needs 1600 Hz at the legal ceiling, and dropped pulses
    read as a slow wheel, which makes the controller add throttle and run away.
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


def label_shot(text):
    """Attach an outcome to the most recently detected shot.

    Type it whenever you like after the shot - the shot's own timestamp and
    speed came from the RPM dip, not from when you pressed the key.

        short / in / long        where it went relative to the bucket
        long 0.5                 add how far out, if you can judge it
    """
    global pending_event

    if not text:
        print("\n  say short / in / long")
        return
    pending_event = f"label:{text}"

    if last_shot_time <= 0:
        print(f"\n  '{text}' recorded, but no shot was detected to attach it "
              f"to - lower SHOT_DIP_RPM if the dip is being missed")
        return
    print(f"\n  '{text}' -> shot {time.monotonic() - last_shot_time:.1f}s ago "
          f"at {last_shot_rpm:.0f} RPM ({muzzle_mps(last_shot_rpm):.2f} m/s)")


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

        if parts[0] in ("short", "in", "long"):
            label_shot(line)
            continue
        if parts[0] == "s":
            label_shot(" ".join(parts[1:]))
            continue
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
            print(f"\n  '{line}' not understood - number, 'kd 0.02', or '?'")
            continue

        if requested > MAX_TARGET_RPM:
            print(f"\n  capping {requested:.0f} -> {MAX_TARGET_RPM:.0f} RPM")
            requested = MAX_TARGET_RPM
        elif requested > MAX_LEGAL_RPM:
            print(f"\n  note: {requested:.0f} RPM is "
                  f"{muzzle_mps(requested):.1f} m/s, above the 12.0 m/s "
                  f"competition limit. Bench only.")
        target_rpm = max(0.0, requested)


# ==========================================================================
# MAIN
# ==========================================================================

def main_loop():
    global rpm, throttle, trim, previous_error, target_rpm

    print(f"Motor STOPPED until you type a target. Max {MAX_TARGET_RPM:.0f} RPM.")
    print(f"BENCH MODE - competition limit is {MAX_LEGAL_RPM:.0f} RPM "
          f"(12.0 m/s); above that the line shows OVER.")
    print("Commands: <rpm> | 0 | kp 0.003 | kd 0.02 | ramp 0.8 | db 20 | ? | q")

    global pending_event, last_shot_time, last_shot_rpm

    log_path = time.strftime("shots_%H%M%S.csv")
    log = open(log_path, "w", buffering=1)
    log.write("t,target,rpm,mps,throttle,trim,event\n")
    print(f"Logging to {log_path}. Shots are detected automatically - just "
          f"type short / in / long after each one.")

    threading.Thread(target=encoder_thread, daemon=True).start()
    threading.Thread(target=read_commands, daemon=True).start()

    log_start = time.monotonic()
    last_log = 0.0
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
                    + (MAX_TARGET_RPM / RPM_PER_THROTTLE) * 1.25)
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

        if now - last_log >= 1.0 / LOG_RATE_HZ or pending_event:
            recent_rpm.append(rpm)

            # Detect the ball: speed over the last 0.3 s dropped clear of
            # where it sat the 0.3 s before that.
            if (target_rpm > 0 and len(recent_rpm) == recent_rpm.maxlen
                    and now - last_shot_time > SHOT_DEBOUNCE_S):
                window = list(recent_rpm)
                half = len(window) // 2
                before = sum(window[:half]) / half
                after = sum(window[half:]) / (len(window) - half)
                # Measurement noise grows roughly with speed squared (the
                # pulse window shrinks as RPM rises and each ms of jitter is
                # worth more RPM), so the threshold must grow with it. A fixed
                # 25 sat inside the noise at 3400 and detected phantom shots.
                dip_threshold = SHOT_DIP_RPM * max(1.0, (target_rpm / 2000.0) ** 2)
                if before - after > dip_threshold:
                    last_shot_time = now
                    last_shot_rpm = before
                    pending_event = f"shot:{before:.0f}:{before - after:.0f}"
                    print(f"\n  shot detected at {before:.0f} RPM "
                          f"({muzzle_mps(before):.2f} m/s), dip {before - after:.0f}")

            log.write(f"{now - log_start:.3f},{target_rpm:.0f},{rpm:.1f},"
                      f"{muzzle_mps(rpm):.3f},{throttle:.3f},{trim:+.3f},"
                      f"{pending_event}\n")
            pending_event = ""
            last_log = now

        # 5 Hz - printing every pass starves the encoder thread.
        if now - last_print >= 0.2:
            elapsed = now - last_poll_time
            if elapsed > 0:
                poll_hz = (poll_count - last_poll_count) / elapsed
            last_poll_count, last_poll_time, last_print = poll_count, now, now
            flag = "OVER" if rpm > MAX_LEGAL_RPM else "    "
            sys.stdout.write(
                f"\rpulses {pulse_count:7d} | RPM {rpm:6.0f} / {target_rpm:<6.0f}"
                f" | {muzzle_mps(rpm):5.2f} m/s {flag} | throttle {throttle:5.2f}"
                f" | trim {trim:+5.2f} | poll {poll_hz / 1000:4.1f}kHz   ")
            sys.stdout.flush()

        time.sleep(0.002)


try:
    main_loop()
except KeyboardInterrupt:
    print("\nExiting...")
finally:
    stop_motor()             # any exit path must neutralise the motor
    print("Motor commanded to neutral.")
