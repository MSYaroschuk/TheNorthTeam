 #!/usr/bin/env python3
import sys
import threading
import time
from queue import Queue
from fusion_hat.servo import Servo
from fusion_hat.pin import Pin

# Define the GPIO pins connected to the encoder.
CHANNEL_A = 22
CHANNEL_B = 27
MOTOR_CHANNEL = 0

# How often to estimate RPM from the encoder samples.
SAMPLE_INTERVAL_SECONDS = 0.2

# How often to update the motor control output.
CONTROL_INTERVAL_SECONDS = 0.1

# Number of quadrature pulses used for one full revolution in this setup.
PULSES_PER_REV = 14

# Initialize the global state variables.
pos = 0                 # Running position count from the encoder.
rpm = 0.0               # Most recently calculated RPM.
rotations = 0           # Total rotations counted so far.
sample_rotations = 0    # Rotations counted during the current sample window.
sample_pulses = 0       # Pulse count during the current sample window.
last_a_state = None     # Previous state of channel A for edge detection.
last_sample_time = time.monotonic()   # Time of the last RPM sample.
last_control_time = time.monotonic()  # Time of the last PID update.

# PID settings for controlling motor speed.
TARGET_RPM = 900.0          # Desired speed in RPM.
KP = 0.3                # Proportional gain. .00015
KI = 0.0005               # Integral gain.
KD = 0.5               # Derivative gain.
MIN_ANGLE = 6.0             # Minimum servo angle.
MAX_ANGLE = 25.0            # Maximum servo angle.
MAX_ANGLE_CHANGE = 0.1     # Maximum change to the motor angle per PID update.
ERROR_DEADBAND = 10.0       # Ignore small RPM errors to reduce jitter.

motor_angle = 10.0         # Current motor/servo angle.
integral_error = 0.0       # Accumulated error for the integral term.
previous_error = 0.0        # Previous error for the derivative term.
target_rpm = TARGET_RPM     # Current target RPM that can be changed at runtime.
input_queue = Queue()      # Queue for receiving new target RPM values from the input thread.

# Initialize the hardware objects.
motor = Servo(MOTOR_CHANNEL)
pinA = Pin(CHANNEL_A, mode=Pin.IN)
pinB = Pin(CHANNEL_B, mode=Pin.IN)
motor.angle(motor_angle)


def clamp(value, low, high):
    """Keep a value inside a specified range."""
    return max(low, min(high, value))


def calculate_rpm(pulses, elapsed_seconds):
    """Convert the number of quadrature pulses observed over a time window into RPM."""
    if elapsed_seconds <= 0:
        return 0.0
    revolutions = pulses / PULSES_PER_REV
    return (revolutions / elapsed_seconds) * 120.0


def update_encoder():
    """Read the encoder channels and count a quadrature transition."""
    global pos, rotations, sample_rotations, sample_pulses, last_a_state

    a_state = pinA.value()
    b_state = pinB.value()

    if last_a_state is not None and a_state != last_a_state:
        if a_state == 1:
            pos += 1
            sample_pulses += 1

            # Count one full revolution when the quadrature state reaches the high-high pattern.
            if b_state == 1 and a_state == 1:
                rotations += 1
                sample_rotations += 1

    last_a_state = a_state


def keyboard_input_loop():
    """Read a new target RPM from the terminal and place it into the queue."""
    global target_rpm

    while True:
        try:
            raw_value = input("Target RPM: ").strip()
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
            target_rpm = float(raw_value)
        except ValueError:
            target_rpm = TARGET_RPM


def main_loop():
    """Main loop: read the encoder, estimate RPM, and adjust the motor angle."""
    global rpm, rotations, sample_rotations, sample_pulses, last_sample_time, last_control_time
    global motor_angle, integral_error, previous_error, target_rpm

    print("Reading encoder... Press Ctrl+C to stop.")
    print("Type a target RPM and press Enter to apply it.")

    last_a_state = pinA.value()
    keyboard_thread = threading.Thread(target=keyboard_input_loop, daemon=True)
    keyboard_thread.start()

    while True:
        update_encoder()

        now = time.monotonic()

        # Update RPM estimate on a fixed sample interval.
        if now - last_sample_time >= SAMPLE_INTERVAL_SECONDS:
            elapsed_seconds = now - last_sample_time
            rpm = calculate_rpm(sample_pulses, elapsed_seconds)
            sample_pulses = 0
            last_sample_time = now

        # Update the motor angle with the PID controller on a fixed control interval.
        if now - last_control_time >= CONTROL_INTERVAL_SECONDS:
            error = target_rpm - rpm
            if abs(error) < ERROR_DEADBAND:
                error = 0.0
            integral_error += error * CONTROL_INTERVAL_SECONDS
            derivative_error = error - previous_error
            control_output = (KP * error) + (KI * integral_error) + (KD * derivative_error)
            angle_change = clamp(control_output, -MAX_ANGLE_CHANGE, MAX_ANGLE_CHANGE)
            motor_angle = clamp(motor_angle + angle_change, MIN_ANGLE, MAX_ANGLE)
            motor.angle(motor_angle)
            previous_error = error
            last_control_time = now
            v_ang = (rpm * 2 * 3.1415)/60

        handle_keyboard()

        sys.stdout.write(
            f"\rRPM: {rpm:6.1f} | Angular velocity: {v_ang:6.1f} | Target: {target_rpm:6.1f} | Angle: {motor_angle:5.2f}"
        )
        sys.stdout.flush()
        time.sleep(0.001)


try:
    main_loop()
except KeyboardInterrupt:
    print("\nExiting...")
    motor.angle(5)
