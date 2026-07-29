import sys
import threading
import time
import math
from queue import Queue
from fusion_hat.servo import Servo
from fusion_hat.pin import Pin

# Define the GPIO pins connected to the encoder.
CHANNEL_A = 17
CHANNEL_B = 4
MOTOR_CHANNEL = 2

# ---------------------------------------------------------------------------
# MOTOR OUTPUT
#
# The ESC does NOT stop at angle 0. It stops at NEUTRAL_ANGLE, which every
# other script on this robot puts at 5 (MOTOR_NEUTRAL / SHOOTER_STOP).
# DIRECTION picks which SIDE of neutral drives the shooting direction - it must
# never be multiplied into the neutral value itself, or "stop" becomes a
# throttle command on the opposite side. That is what left the motor running
# after Ctrl-C.
#
#   angle = NEUTRAL_ANGLE + DIRECTION * throttle      throttle 0 = stopped
#
# Sanity check against the chassis files: 5 + (-1 * 25) = -20, which is exactly
# SHOOTER_MAX_SPEED in chassis_1.py and chassis_2.py.
# ---------------------------------------------------------------------------
NEUTRAL_ANGLE = 5.0     # ESC stop. VERIFY: at this value the motor must not turn.
DIRECTION = -1          # -1 drives negative, matching the chassis scripts
MAX_THROTTLE = 25.0     # full throttle magnitude away from neutral

# How often to estimate RPM from the encoder samples.
SAMPLE_INTERVAL_SECONDS = 0.2

# How often to update the motor control output.
CONTROL_INTERVAL_SECONDS = 0.1

# Counts registered by encoder_thread() per full revolution of the flywheel.
#
# MEASURED at 16 over one hand-turned revolution. encoder_thread() counts twice
# per quadrature cycle (once when A rises, once when B changes while A is high),
# so 16 counts implies 8 cycles = 32 quadrature ticks per revolution.
#
# CONFIRM THIS WITH TEN REVOLUTIONS, not one. Restart the script to zero the
# count, turn the shaft through exactly 10 revolutions, and expect 160. A single
# rotation can easily be off by a count or two from contact bounce or from
# starting mid-cycle, and the error direction is not safe: if the true figure is
# lower than what is set here, the RPM readout under-reports, and the shooter
# will pass 12.0 m/s while the screen still says it is legal.
PULSES_PER_REV = 16

# Initialize the global state variables.
pos = 0                 # Running position count from the encoder.
rpm = 0.0               # Most recently calculated RPM.
rotations = 0           # Total rotations counted so far.
sample_rotations = 0    # Rotations counted during the current sample window.
sample_pulses = 0       # Pulse count during the current sample window.
last_a_state = None     # Previous state of channel A for edge detection.
last_b_state = None     # Previous state of channel B for edge detection.
last_sample_time = time.monotonic()   # Time of the last RPM sample.
last_control_time = time.monotonic()  # Time of the last PID update.

# PID settings. All five are LIVE-TUNABLE while running - type e.g. "kp 0.5"
# and press Enter. These values are only the starting point at launch.
#
# Nothing is written back to this file, so when you land on numbers you like,
# copy them in here before you lose the terminal.
# CAREFUL - in this INCREMENTAL controller the terms are not what their names
# suggest. throttle accumulates the output, so:
#     KP term -> adds to throttle every tick   => acts as INTEGRAL action
#     KD term -> responds to change in error   => acts as PROPORTIONAL action
#     KI term -> integral of an integral       => DOUBLE integral, destabilising
#
# That is why KI is zero. Simulated with an inertia wheel, KI = 0.0005 (the old
# value) overshoots by about 1450 RPM once the ramp clamp stops hiding it.
# The old KP of 0.3 was roughly 2800x too high and only appeared stable because
# the clamp saturated it into a slew-limited bang-bang controller.
KP = 0.0010             # integral action  (was 0.3)
KI = 0.0                # MUST stay 0 here (was 0.0005)
KD = 0.020              # proportional damping  (was 0.5)

# Maximum change to the throttle per control update. This is a rate limit that
# sits AFTER the PID, so it caps how fast the flywheel can spin up no matter
# what the gains are: at 0.1 per 0.1 s update, crossing the full 0..25 throttle
# range takes 25 seconds. If spin-up feels hopelessly slow, this is the knob,
# not the gains. Raise it with "ramp 0.5".
MAX_ANGLE_CHANGE = 0.8  # was 0.1, which needed 25 s to cross the throttle range

ERROR_DEADBAND = 15.0   # Ignore small RPM errors to reduce jitter.

# Rule 5.5: tennis balls may not be launched above 12.0 m/s. On a 3 in flywheel
# that is the RPM below. Typed targets are capped at it.
FLYWHEEL_DIAMETER_IN = 3.0
FLYWHEEL_RADIUS_M = FLYWHEEL_DIAMETER_IN * 0.0254 / 2.0
MAX_LEGAL_RPM = (12.0 / FLYWHEEL_RADIUS_M) * 60.0 / (2.0 * math.pi)

# Commanded but not turning for this long -> cut power rather than let the
# integral wind up to full throttle against a dead encoder.
STALL_SECONDS = 3.0

# Roughly how much RPM one throttle unit buys, at the top of the range.
# Only used for the runaway guard below, so it does not need to be exact.
MOTOR_FREE_RPM = 6000.0
RPM_PER_THROTTLE = MOTOR_FREE_RPM / MAX_THROTTLE

# RUNAWAY GUARD.
#
# If the encoder under-reports - dropped pulses, a loose wire, a failing
# sensor - the controller sees a wheel that is too slow and keeps adding
# throttle. The wheel really does accelerate, the screen does not show it, and
# nothing in the loop notices. The stall guard does not catch this either,
# because the wheel IS turning.
#
# So bound the throttle by what the requested speed could plausibly need. A
# target of 1700 RPM should need about 1700/240 = 7 throttle units; there is no
# honest reason to be commanding 20. The 1.6 multiplier leaves room for a low
# battery and for the wheel to be loaded, while still catching a runaway early.
THROTTLE_HEADROOM = 1.6

throttle = 0.0          # Current throttle magnitude, 0 = stopped.
integral_error = 0.0    # Accumulated error for the integral term.
previous_error = 0.0    # Previous error for the derivative term.

# START STOPPED. This used to be TARGET_RPM (900), which meant the motor span
# up the moment the script launched, before anyone typed anything.
target_rpm = 0.0
input_queue = Queue()   # Queue for new target RPM values from the input thread.

# Initialize the hardware objects.
motor = Servo(MOTOR_CHANNEL)
pinA = Pin(CHANNEL_A, mode=Pin.IN)
pinB = Pin(CHANNEL_B, mode=Pin.IN)


def clamp(value, low, high):
    """Keep a value inside a specified range."""
    return max(low, min(high, value))


def write_throttle(value):
    """Command a throttle magnitude. 0 is a genuine stop."""
    value = clamp(value, 0.0, MAX_THROTTLE)
    motor.angle(NEUTRAL_ANGLE + DIRECTION * value)


def stop_motor():
    """Neutral. Never multiply DIRECTION into this."""
    motor.angle(NEUTRAL_ANGLE)


stop_motor()


def calculate_rpm(pulses, elapsed_seconds):
    """Convert the counts observed over a time window into RPM.

    Revolutions per SECOND times 60 gives revolutions per MINUTE. This used to
    multiply by 120, which had no justification and made every reading twice as
    fast as reality. Together with the old PULSES_PER_REV of 14, the display was
    over-reporting by a factor of (120/14) / (60/16) = 2.29.
    """
    if elapsed_seconds <= 0:
        return 0.0
    revolutions = pulses / PULSES_PER_REV
    return (revolutions / elapsed_seconds) * 60.0


encoder_lock = threading.Lock()
poll_count = 0          # how many times the encoder thread has sampled the pins


def encoder_thread():
    """Poll the encoder as fast as possible, on its own thread.

    THIS USED TO LIVE IN THE MAIN LOOP, and that was the cause of the hunting
    at higher speeds. The main loop also does a stdout write, a flush and a
    sleep(0.001) every pass, which caps it near 700-900 Hz. To catch every
    rising edge you have to sample faster than channel A's high time:

        1000 RPM -> 533 Hz needed      (fine)
        1500 RPM -> 800 Hz needed      (marginal - this is where it broke)
        2000 RPM -> 1067 Hz needed     (misses pulses)

    Missing pulses makes the measured RPM read LOW, so the PID pushes harder,
    which raises the speed, which misses MORE pulses. That runs away until the
    count happens to catch up and the controller slams back. The result looks
    exactly like a badly tuned loop, but no gain change can fix it.

    On its own thread with nothing else to do, this polls in the tens of kHz.
    The status line shows the achieved rate - if it ever drops near the numbers
    above, the readings are not trustworthy.
    """
    global pos, rotations, sample_rotations, sample_pulses
    global last_a_state, last_b_state, poll_count

    last_a_state = pinA.value()
    last_b_state = pinB.value()

    while True:
        a_state = pinA.value()
        b_state = pinB.value()
        poll_count += 1

        if (a_state != last_a_state) or (b_state != last_b_state):
            if a_state == 1:
                with encoder_lock:
                    pos += 1
                    sample_pulses += 1
                    if b_state == 1:
                        rotations += 1
                        sample_rotations += 1

        last_a_state = a_state
        last_b_state = b_state


TUNABLES = ("kp", "ki", "kd", "ramp", "db")


def keyboard_input_loop():
    """Read commands from the terminal and place them into the queue."""
    while True:
        try:
            raw_value = input("cmd (RPM / kp / ki / kd / ramp / db / ?): ").strip()
        except EOFError:
            break
        if raw_value:
            input_queue.put(raw_value)


def show_settings():
    print(f"\n  kp {KP:<10g} ki {KI:<10g} kd {KD:<10g}"
          f"\n  ramp {MAX_ANGLE_CHANGE:<8g} db {ERROR_DEADBAND:<10g}"
          f"  target {target_rpm:.0f} RPM")


def handle_keyboard():
    """Apply queued commands.

    A bare number is a target RPM. Anything else is a live gain change, so the
    PID can be tuned without stopping the motor or restarting the program.
    """
    global target_rpm, KP, KI, KD, MAX_ANGLE_CHANGE, ERROR_DEADBAND
    global integral_error

    while not input_queue.empty():
        raw_value = input_queue.get().strip().lower()
        if not raw_value:
            continue

        if raw_value in ("?", "h", "help"):
            show_settings()
            continue

        parts = raw_value.split()

        # Accept "kp 0.5" and "kp0.5" alike.
        if len(parts) == 1:
            for key in TUNABLES:
                if parts[0].startswith(key) and len(parts[0]) > len(key):
                    parts = [key, parts[0][len(key):]]
                    break

        if len(parts) == 2 and parts[0] in TUNABLES:
            key, text = parts
            try:
                value = float(text)
            except ValueError:
                print(f"\n  '{text}' is not a number - ignored")
                continue

            if key == "kp":
                KP = value
            elif key == "ki":
                # Bumpless transfer: the output contains KI * integral_error,
                # so rescale the accumulator to keep that product constant.
                # Without this, changing KI mid-spin kicks the throttle.
                if value > 0 and KI > 0:
                    integral_error *= KI / value
                else:
                    integral_error = 0.0
                KI = value
            elif key == "kd":
                KD = value
            elif key == "ramp":
                MAX_ANGLE_CHANGE = max(0.0, value)
            elif key == "db":
                ERROR_DEADBAND = max(0.0, value)

            show_settings()
            continue

        try:
            requested = float(raw_value)
        except ValueError:
            print(f"\n  '{raw_value}' not understood. Type a number for target "
                  f"RPM, 'kp 0.5' to set a gain, or '?' to show settings.")
            continue

        if requested > MAX_LEGAL_RPM:
            print(f"\n  capping {requested:.0f} -> {MAX_LEGAL_RPM:.0f} RPM "
                  f"(12.0 m/s limit, rule 5.5)")
            requested = MAX_LEGAL_RPM

        target_rpm = max(0.0, requested)


def main_loop():
    """Main loop: read the encoder, estimate RPM, and adjust the motor throttle."""
    global rpm, rotations, sample_rotations, sample_pulses
    global last_sample_time, last_control_time
    global throttle, integral_error, previous_error, target_rpm

    print("Reading encoder... Press Ctrl+C to stop.")
    print(f"Motor is STOPPED until you type a target. Legal max "
          f"{MAX_LEGAL_RPM:.0f} RPM.")
    print("Gains are live - 'kp 0.5', 'ki 0.001', 'kd 0.2', 'ramp 0.5', "
          "'db 15'. '?' shows them.")
    print("Gain changes are NOT saved - copy the good ones into the file.")
    print("Spin the flywheel by hand - 'pulses' should climb.")

    keyboard_thread = threading.Thread(target=keyboard_input_loop, daemon=True)
    keyboard_thread.start()

    # The encoder gets its own thread so the print and sleep below cannot
    # starve it. See encoder_thread() for why that matters.
    threading.Thread(target=encoder_thread, daemon=True).start()

    last_motion = time.monotonic()
    last_print_time = time.monotonic()
    last_poll_count = 0
    last_poll_time = time.monotonic()
    poll_hz = 0.0

    while True:
        now = time.monotonic()

        # Update RPM estimate on a fixed sample interval.
        if now - last_sample_time >= SAMPLE_INTERVAL_SECONDS:
            elapsed_seconds = now - last_sample_time
            with encoder_lock:
                pulses = sample_pulses
                sample_pulses = 0
            rpm = calculate_rpm(pulses, elapsed_seconds)
            last_sample_time = now

        if rpm > 5.0:
            last_motion = now

        # Update the throttle with the PID controller on a fixed control interval.
        if now - last_control_time >= CONTROL_INTERVAL_SECONDS:
            if target_rpm <= 0:
                # Commanded stop: zero everything so the next spin-up starts
                # clean and the integral cannot carry over.
                throttle = 0.0
                integral_error = 0.0
                previous_error = 0.0
                last_motion = now
                stop_motor()
            else:
                error = target_rpm - rpm
                if abs(error) < ERROR_DEADBAND:
                    error = 0.0
                integral_error += error * CONTROL_INTERVAL_SECONDS
                derivative_error = error - previous_error
                control_output = ((KP * error)
                                  + (KI * integral_error)
                                  + (KD * derivative_error))
                angle_change = clamp(control_output,
                                     -MAX_ANGLE_CHANGE, MAX_ANGLE_CHANGE)
                # Never command more throttle than the requested speed could
                # plausibly need. Without this, an under-reporting encoder
                # drives the flywheel away while the screen looks calm.
                throttle_ceiling = min(
                    MAX_THROTTLE,
                    (target_rpm / RPM_PER_THROTTLE) * THROTTLE_HEADROOM + 1.0,
                    # And never past the throttle the LEGAL ceiling should need,
                    # plus a little for battery sag. This is the only thing
                    # standing between a lying encoder and an illegal shot.
                    (MAX_LEGAL_RPM / RPM_PER_THROTTLE) * 1.25)
                throttle = clamp(throttle + angle_change, 0.0, throttle_ceiling)

                if throttle >= throttle_ceiling - 1e-9 and rpm < target_rpm * 0.7:
                    print(f"\n  !! throttle is capped at {throttle_ceiling:.1f} "
                          f"for a {target_rpm:.0f} RPM target, but the encoder "
                          f"only reads {rpm:.0f}.")
                    print("     Suspect dropped pulses - check the poll rate. "
                          "The wheel may be far faster than shown.")

                if now - last_motion > STALL_SECONDS:
                    print(f"\n  !! commanded {target_rpm:.0f} RPM but nothing "
                          f"has turned for {STALL_SECONDS:.0f}s - stopping.")
                    print("     Motor not spinning, or encoder not reading.")
                    target_rpm = 0.0
                    throttle = 0.0
                    integral_error = 0.0
                    previous_error = 0.0
                    stop_motor()
                else:
                    write_throttle(throttle)

                previous_error = error

            last_control_time = now

        handle_keyboard()

        # Print at ~5 Hz, not every pass. The write and flush used to run
        # thousands of times a second and were a large part of what slowed the
        # encoder sampling down. Nothing is lost - the eye cannot read faster.
        if now - last_print_time >= 0.2:
            dt_poll = now - last_poll_time
            if dt_poll > 0:
                poll_hz = (poll_count - last_poll_count) / dt_poll
            last_poll_count, last_poll_time = poll_count, now
            last_print_time = now

            sys.stdout.write(
                f"\rpulses {pos:7d} | RPM: {rpm:6.1f} | Target: {target_rpm:6.1f} "
                f"| Throttle: {throttle:5.2f} "
                f"| angle {NEUTRAL_ANGLE + DIRECTION * throttle:6.1f} "
                f"| poll {poll_hz / 1000:5.1f}kHz   "
            )
            sys.stdout.flush()

        time.sleep(0.002)


try:
    main_loop()
except KeyboardInterrupt:
    print("\nExiting...")
finally:
    # A finally block, not just the KeyboardInterrupt handler: ANY crash must
    # also stop the motor. Previously an unexpected exception left it running.
    stop_motor()
    print("Motor commanded to neutral.")
