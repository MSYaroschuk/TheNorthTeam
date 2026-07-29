import sys
import threading
import time
import math
from collections import deque
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

# Speed is measured by timing this many pulses - see read_rpm().
#
# SIZED FROM LOGGED DATA, not guessed. With 20 pulses the readings had a
# standard deviation of 53 RPM at a 2000 RPM target, and - the giveaway -
# consecutive readings 104 ms apart were UNCORRELATED (autocorrelation -0.04).
# A flywheel with an inertia disc cannot change speed randomly between samples;
# a real one would have shown about +0.9. So that scatter was measurement
# noise, and the controller was faithfully chasing it, dithering the throttle
# by 1.8 units (about 600 RPM of command) to correct a wheel that was already
# steady.
#
# The error works out to about 1 ms of timing jitter, which is the encoder
# thread losing the GIL or being descheduled. The 16 us poll resolution is not
# the limit; the OS is. Jitter is roughly fixed, so a longer window dilutes it:
#
#     pulses   span at 2000 RPM   noise      lag
#         20               38 ms   53 RPM   19 ms
#         48               90 ms   22 RPM   45 ms
#         64              120 ms   16 RPM   60 ms
#
# 64 is chosen to get the noise under the +-25 RPM the shot actually needs.
# The added lag is tens of ms against a plant whose time constant is seconds.
RPM_WINDOW_PULSES = 64
RPM_STALE_SECONDS = 0.3   # no pulses for this long means the wheel has stopped

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
KP = 0.0020             # integral action  (was 0.3)
KI = 0.0                # MUST stay 0 here (was 0.0005)
KD = 0.030              # proportional damping  (was 0.5)

# Maximum change to the throttle per control update. This is a rate limit that
# sits AFTER the PID, so it caps how fast the flywheel can spin up no matter
# what the gains are: at 0.1 per 0.1 s update, crossing the full 0..25 throttle
# range takes 25 seconds. If spin-up feels hopelessly slow, this is the knob,
# not the gains. Raise it with "ramp 0.5".
MAX_ANGLE_CHANGE = 0.8  # was 0.1, which needed 25 s to cross the throttle range

# Ignore errors smaller than this. Should sit just above the measurement noise
# floor, so the controller does not chase it. With a 64-pulse window that floor
# is about 16 RPM, hence 20.
ERROR_DEADBAND = 20.0

# Rule 5.5: tennis balls may not be launched above 12.0 m/s. On a 3 in flywheel
# that is the RPM below. Typed targets are capped at it.
FLYWHEEL_DIAMETER_IN = 3.0
FLYWHEEL_RADIUS_M = FLYWHEEL_DIAMETER_IN * 0.0254 / 2.0
MAX_LEGAL_RPM = (12.0 / FLYWHEEL_RADIUS_M) * 60.0 / (2.0 * math.pi)

# Commanded but not turning for this long -> cut power rather than let the
# integral wind up to full throttle against a dead encoder.
STALL_SECONDS = 3.0

# Every control tick is written here so an oscillation can be plotted rather
# than guessed at. Overwritten on each run - copy it off before restarting.
LOG_PATH = "rpm_log.csv"

# MEASURED PLANT, fitted from throttle/RPM pairs logged on the bench:
#
#     rpm = 340 * throttle - 1544
#
# Two things fall out of that, and both were wrong in earlier versions:
#   - one throttle unit is worth 340 RPM, not the 240 implied by dividing free
#     speed by max throttle
#   - the ESC produces NOTHING until about throttle 4.5. Everything below that
#     is dead travel.
#
# RE-MEASURE THESE if the battery, ESC, wheel or belt changes. Command a few
# fixed speeds, note the settled throttle, and fit a straight line.
RPM_PER_THROTTLE = 340.0
THROTTLE_DEADBAND = 4.54        # throttle at which the wheel first turns


def feedforward_throttle(target):
    """The throttle that should, on its own, produce the requested speed.

    Without this the controller has to discover the operating point from zero
    every single time, walking the throttle up by KP*error per tick. That is
    slow, and it means the integral term is doing all the work at steady state
    rather than just trimming.
    """
    if target <= 0:
        return 0.0
    return THROTTLE_DEADBAND + target / RPM_PER_THROTTLE


# How far the PID may pull away from the feedforward estimate. Wide enough to
# cover a sagging battery and a loaded wheel, tight enough that a bad estimate
# cannot run away.
MAX_TRIM = 4.0

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
trim = 0.0              # PID correction on top of the feedforward estimate.
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


encoder_lock = threading.Lock()
poll_count = 0          # how many times the encoder thread has sampled the pins
pulse_times = deque(maxlen=RPM_WINDOW_PULSES)   # monotonic time of recent pulses


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
                    pulse_times.append(time.monotonic())
                    if b_state == 1:
                        rotations += 1
                        sample_rotations += 1

        last_a_state = a_state
        last_b_state = b_state


def read_rpm():
    """Speed from the TIME SPANNED by the last RPM_WINDOW_PULSES pulses.

    The old method counted whole pulses inside a fixed 0.2 s window. That
    quantises hard: one pulse is (60 / PULSES_PER_REV) / 0.2 = 18.75 RPM, so at
    a 2000 RPM target the only readings available were 1981, 2000 and 2019.
    Nothing downstream can hold tighter than the ruler can measure, and the
    controller was chasing single-count wobble as if it were real speed change.

    Timing a fixed NUMBER of pulses instead uses the 62 kHz poll clock as the
    ruler. At 2000 RPM twenty pulses span about 38 ms, which the poll resolves
    to roughly +-0.9 RPM - about twenty times finer than counting.

    It also updates every control tick rather than every 0.2 s, so the
    derivative term sees a fresh number each time instead of alternating
    between a real change and a stale zero.
    """
    with encoder_lock:
        if len(pulse_times) < 2:
            return 0.0
        stamps = list(pulse_times)

    # Nothing recently -> stopped. Without this the last computed speed would
    # persist forever after the wheel halted.
    if time.monotonic() - stamps[-1] > RPM_STALE_SECONDS:
        return 0.0

    span = stamps[-1] - stamps[0]
    if span <= 0:
        return 0.0
    return ((len(stamps) - 1) / PULSES_PER_REV) / span * 60.0


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
    global last_control_time
    global throttle, trim, integral_error, previous_error, target_rpm

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

    log_file = open(LOG_PATH, "w", buffering=1)
    log_file.write("t,target,rpm,throttle,trim,kp,kd,db\n")
    log_start = time.monotonic()

    last_motion = time.monotonic()
    last_print_time = time.monotonic()
    last_poll_count = 0
    last_poll_time = time.monotonic()
    poll_hz = 0.0

    while True:
        now = time.monotonic()

        # Speed is available continuously now, not once per 0.2 s window.
        rpm = read_rpm()

        if rpm > 5.0:
            last_motion = now

        # Update the throttle with the PID controller on a fixed control interval.
        if now - last_control_time >= CONTROL_INTERVAL_SECONDS:
            if target_rpm <= 0:
                # Commanded stop: zero everything so the next spin-up starts
                # clean and the integral cannot carry over.
                throttle = 0.0
                trim = 0.0
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

                # Feedforward carries the operating point; the PID only trims
                # around it. Previously the PID had to walk the throttle up
                # from zero itself, which is slow and leaves the integral
                # holding the whole load at steady state.
                trim = clamp(trim + angle_change, -MAX_TRIM, MAX_TRIM)

                # Never command more throttle than the requested speed could
                # plausibly need. Without this, an under-reporting encoder
                # drives the flywheel away while the screen looks calm.
                # Headroom applies only to the part ABOVE the ESC's dead
                # travel - scaling the deadband itself would just inflate the
                # ceiling for no reason.
                throttle_ceiling = min(
                    MAX_THROTTLE,
                    THROTTLE_DEADBAND
                    + (target_rpm / RPM_PER_THROTTLE) * THROTTLE_HEADROOM,
                    # And never past the throttle the LEGAL ceiling should need,
                    # plus a little for battery sag. This is the only thing
                    # standing between a lying encoder and an illegal shot.
                    THROTTLE_DEADBAND
                    + (MAX_LEGAL_RPM / RPM_PER_THROTTLE) * 1.25)
                throttle = clamp(feedforward_throttle(target_rpm) + trim,
                                 0.0, throttle_ceiling)

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

            # Log every control tick, not every print. The 5 Hz status line is
            # far too slow to show the shape of an oscillation - this is what
            # you plot afterwards to see its period and amplitude.
            if log_file is not None:
                log_file.write(f"{now - log_start:.3f},{target_rpm:.1f},"
                               f"{rpm:.1f},{throttle:.3f},{trim:+.3f},"
                               f"{KP:g},{KD:g},{ERROR_DEADBAND:g}\n")

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
