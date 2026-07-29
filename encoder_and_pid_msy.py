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

# Number of quadrature pulses used for one full revolution in this setup.
#
# UNVERIFIED - see calculate_rpm(). update_encoder() counts twice per quadrature
# cycle (once when A rises, once when B changes while A is high), so 14 counts
# per revolution implies 7 quadrature cycles per revolution. That is consistent
# with a 28-tick goBILDA encoder. Confirm it with the pulse count on screen:
# zero it by restarting, turn the shaft exactly 10 revolutions, expect 140.
PULSES_PER_REV = 14

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

# PID settings for controlling motor speed.
KP = 0.3                # Proportional gain. .00015
KI = 0.0005             # Integral gain.
KD = 0.5                # Derivative gain.
MAX_ANGLE_CHANGE = 0.1  # Maximum change to the throttle per PID update.
ERROR_DEADBAND = 10.0   # Ignore small RPM errors to reduce jitter.

# Rule 5.5: tennis balls may not be launched above 12.0 m/s. On a 3 in flywheel
# that is the RPM below. Typed targets are capped at it.
FLYWHEEL_DIAMETER_IN = 3.0
FLYWHEEL_RADIUS_M = FLYWHEEL_DIAMETER_IN * 0.0254 / 2.0
MAX_LEGAL_RPM = (12.0 / FLYWHEEL_RADIUS_M) * 60.0 / (2.0 * math.pi)

# Commanded but not turning for this long -> cut power rather than let the
# integral wind up to full throttle against a dead encoder.
STALL_SECONDS = 3.0

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
    """Convert the number of quadrature pulses observed over a time window into RPM.

    NOTE the 120.0. Converting pulses/second to RPM needs 60. The extra factor
    of 2 here has never been checked against a known speed, so this may read
    double the true RPM. It is left as-is because the current gains were tuned
    against this number - but before trusting the m/s figure for a legal shot,
    verify it (count 10 revolutions by hand, or film the wheel).
    """
    if elapsed_seconds <= 0:
        return 0.0
    revolutions = pulses / PULSES_PER_REV
    return (revolutions / elapsed_seconds) * 120.0


def update_encoder():
    """Read the encoder channels and count a quadrature transition."""
    global pos, rotations, sample_rotations, sample_pulses
    global last_a_state, last_b_state

    a_state = pinA.value()
    b_state = pinB.value()

    if last_a_state is not None and (
            (a_state != last_a_state) or (b_state != last_b_state)):
        if a_state == 1:
            pos += 1
            sample_pulses += 1

            # Count one full revolution when the quadrature state reaches the high-high pattern.
            if b_state == 1 and a_state == 1:
                rotations += 1
                sample_rotations += 1

    last_a_state = a_state
    last_b_state = b_state


def keyboard_input_loop():
    """Read a new target RPM from the terminal and place it into the queue."""
    while True:
        try:
            raw_value = input("Target RPM (0 to stop): ").strip()
        except EOFError:
            break
        if raw_value:
            input_queue.put(raw_value)


def handle_keyboard():
    """Apply any newly queued target RPM values to the control loop."""
    global target_rpm

    while not input_queue.empty():
        raw_value = input_queue.get()
        try:
            requested = float(raw_value)
        except ValueError:
            print("\n  not a number - ignored")
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
    print("Spin the flywheel by hand - 'pulses' should climb.")

    keyboard_thread = threading.Thread(target=keyboard_input_loop, daemon=True)
    keyboard_thread.start()

    last_motion = time.monotonic()

    while True:
        update_encoder()

        now = time.monotonic()

        # Update RPM estimate on a fixed sample interval.
        if now - last_sample_time >= SAMPLE_INTERVAL_SECONDS:
            elapsed_seconds = now - last_sample_time
            rpm = calculate_rpm(sample_pulses, elapsed_seconds)
            sample_pulses = 0
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
                throttle = clamp(throttle + angle_change, 0.0, MAX_THROTTLE)

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

        sys.stdout.write(
            f"\rpulses {pos:7d} | RPM: {rpm:6.1f} | Target: {target_rpm:6.1f} "
            f"| Throttle: {throttle:5.2f} | angle {NEUTRAL_ANGLE + DIRECTION * throttle:6.1f}   "
        )
        sys.stdout.flush()
        time.sleep(0.001)


try:
    main_loop()
except KeyboardInterrupt:
    print("\nExiting...")
finally:
    # A finally block, not just the KeyboardInterrupt handler: ANY crash must
    # also stop the motor. Previously an unexpected exception left it running.
    stop_motor()
    print("Motor commanded to neutral.")
