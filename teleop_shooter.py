#!/usr/bin/env python3

# teleop_shooter.py
# Teleop drive + intake + continuous-servo feeder + CLOSED-LOOP PID shooter.
#
# v1.0 - practice field build
# (C) Team Northeast
#
# This is chassis_2.py's teleop with the open-loop shooter replaced by the RPM
# PID from encoderPIDRPM-2motor.py. chassis_2.py is untouched - keep it as the
# fallback if this misbehaves on the field.
#
# ==========================================================================
#  CONFIRM EVERY VALUE IN THE "WIRING" BLOCK BEFORE YOU POWER ON.
#
#  Two conflicts had to be resolved to merge these two programs, and both
#  are wiring changes, not code changes:
#
#  1. ENCODER PINS. encoderPIDRPM-2motor.py had the flywheel encoder on
#     GPIO 22 and 27. chassis_2.py has the RC steering and throttle on
#     GPIO 22 and 27. They cannot share. The encoder has been moved to
#     GPIO 23/24 here - REWIRE THE ENCODER TO MATCH, or change these
#     constants to whatever free pins you actually use.
#
#  2. CHANNEL MAP. chassis_1/2.py drive 2 motors with the shooter on PWM
#     0-1; auto_chassis.py drives 4 motors with the shooter on 6-7. Set
#     CHASSIS_LAYOUT below to whichever robot you are taking to the field.
# ==========================================================================

import math
import sys
import threading
import time
from collections import deque
from time import sleep

import pigpio
from fusion_hat.servo import Servo

# ==========================================================================
# WIRING - CONFIRM ALL OF THIS
# ==========================================================================

# "4motor" = auto_chassis.py layout. "2motor" = chassis_1/chassis_2.py layout.
CHASSIS_LAYOUT = "2motor"

if CHASSIS_LAYOUT == "4motor":
    LEFT_DRIVE_CH = [0, 1]
    RIGHT_DRIVE_CH = [2, 3]
    INTAKE_CH = 4
    FEEDER_CH = 5
    SHOOTER_CH = [6, 7]
else:
    SHOOTER_CH = [0, 1]
    FEEDER_CH = 2
    INTAKE_CH = 3
    LEFT_DRIVE_CH = [4]
    RIGHT_DRIVE_CH = [5]

# RC receiver inputs (BCM numbering). Matches chassis_2.py.
# NOTE chassis_1.py has STEERING and THROTTLE swapped relative to this.
SLIDER_PIN = 4        # flywheel speed slider
SWITCH_PIN = 17       # 3-position trigger switch
STEERING_PIN = 22     # left/right joystick
THROTTLE_PIN = 27     # forward/backward joystick

# Flywheel encoder. MOVED off 22/27 to clear the RC inputs - see header.
# Only channel A is used; B is read for direction but the shooter is
# single-direction so it does not currently matter.
ENCODER_A_PIN = 23
ENCODER_B_PIN = 24

# ==========================================================================
# SHOOTER - RPM CONTROL
# ==========================================================================

# See encoderPIDRPM-2motor.py for the derivation of all of this.
PULSES_PER_REV = 7            # rising edges of channel A per flywheel rev
MOTOR_FREE_RPM = 6000.0       # goBILDA bare motor
FLYWHEEL_DIAMETER_IN = 3.0
FLYWHEEL_RADIUS_M = FLYWHEEL_DIAMETER_IN * 0.0254 / 2.0
LAUNCH_EFFICIENCY = 1.0
MAX_LEGAL_LAUNCH_MPS = 12.0   # NEXT 2026 rule 5.5

# The motor free-speeds at roughly TWICE the legal launch speed, so this cap
# is the difference between a legal shot and a disqualified one.
MAX_LEGAL_RPM = (MAX_LEGAL_LAUNCH_MPS /
                 (FLYWHEEL_RADIUS_M * LAUNCH_EFFICIENCY)) * 60.0 / (2.0 * math.pi)

# Slider maps across this band. Low end is a soft practice shot, high end is
# the legal ceiling. Never raise SLIDER_MAX_RPM above MAX_LEGAL_RPM.
SLIDER_MIN_RPM = 1200.0
SLIDER_MAX_RPM = MAX_LEGAL_RPM

# ESC angles for the shooter.
# CONFLICT TO RESOLVE ON THE BENCH: encoderPIDRPM-2motor.py used positive
# 6..25 for this direction, chassis_1/2.py used NEGATIVE (-20) for full speed
# with 5 as stop. Whichever is right depends on ESC wiring. These follow the
# PID script because that is the one that actually closed the loop.
SHOOTER_STOP_ANGLE = 5.0
SHOOTER_MIN_ANGLE = 6.0
SHOOTER_MAX_ANGLE = 25.0

# Angle that corresponds to the legal RPM ceiling, assuming angle maps roughly
# linearly to free speed. The control loop clamps to THIS, not to
# SHOOTER_MAX_ANGLE, so the loop can never command an illegal speed even
# during a spin-up transient.
#
# Simulated without this clamp, a target at the legal ceiling overshot to
# ~3330 RPM (13.3 m/s) on the way up. The feeder would not have run there (it
# waits for at-speed), but the flywheel physically reaches that speed, and
# anything already touching it goes out illegally.
#
# If the shooter cannot reach its target on the field, the honest fix is to
# re-measure LAUNCH_EFFICIENCY - NOT to raise this.
SHOOTER_LEGAL_MAX_ANGLE = SHOOTER_MIN_ANGLE + min(
    1.0, MAX_LEGAL_RPM / MOTOR_FREE_RPM) * (SHOOTER_MAX_ANGLE - SHOOTER_MIN_ANGLE)

# If measured RPM ever exceeds the legal ceiling anyway - wrong efficiency
# assumption, nonlinear ESC, tailwind on the encoder - cut the trim hard
# rather than waiting for the PID to walk it back.
OVERSPEED_CUTBACK = 1.0       # angle units removed per control tick

# chassis_1/2.py send the SAME angle to both shooter motors, so the ESCs or
# the mounting must already mirror one of them. encoderPIDRPM-2motor.py
# instead had SECOND_MOTOR_REVERSED = True. Following the chassis files here
# because that is the code that has actually run on the robot.
SECOND_SHOOTER_REVERSED = False

# PID gains, from encoderPIDRPM-2motor.py.
KP = 0.3
KI = 0.0005
KD = 0.5
ERROR_DEADBAND = 10.0         # RPM
MAX_TRIM_CHANGE = 0.1         # max angle trim change per control tick
MAX_TRIM = 8.0                # how far PID may pull away from feedforward

CONTROL_INTERVAL = 0.05       # 20 Hz shooter control

# Feedforward. Without it the trim rate above needs ~19 s to cross the angle
# range, so spin-up would be unusably slow. Feedforward jumps straight to the
# estimated angle and lets the PID trim from there.
USE_FEEDFORWARD = True

# "At speed" window. The feeder only runs while measured RPM is within
# AT_SPEED_TOLERANCE of target for AT_SPEED_SAMPLES consecutive control ticks.
# This is the whole "only feed once the PID reaches target" rule: below the
# window the feeder is off, so no ball is ever fed into a slow flywheel.
AT_SPEED_TOLERANCE = 75.0     # RPM
AT_SPEED_SAMPLES = 3          # consecutive samples inside tolerance

# RPM measurement. Timing between pulses rather than counting per window:
# at 3000 RPM channel A only produces ~350 edges/s, so a fixed 50 ms window
# would quantise to ~170 RPM. A 20-pulse sliding window is ~57 ms at that
# speed with far better resolution.
RPM_WINDOW_PULSES = 20
RPM_STALE_S = 0.25            # no pulses for this long means stopped

# ==========================================================================
# FEEDER / INTAKE   (both are CONTINUOUS-ROTATION servos, driven by .speed())
# ==========================================================================

# The feeder is a continuous servo, not a positional gate. It runs to push
# balls into the flywheel and stops otherwise - it is never commanded to an
# "open"/"closed" position. It only ever runs while the shooter is at speed.
#
# .speed() takes the same range as the intake. STOP is the trim point where a
# continuous servo actually holds still - confirm it on the bench, it may not
# be 0. If the feeder runs the wrong way, negate FEEDER_RUN.
FEEDER_RUN = 60
FEEDER_STOP = 5

INTAKE_SPEED = 30
INTAKE_STOP = 5

# ==========================================================================
# CHASSIS
# ==========================================================================

MOTOR_MAX_FWD = 50            # chassis_1.py value. chassis_2.py used 90.
MOTOR_NEUTRAL = 5
ACCEL_LIMIT = 0.014
DECEL_LIMIT = 0.014

THROTTLE_MIN_FWD = 1250
THROTTLE_TYP = 1500
THROTTLE_MAX_REV = 1750

STEERING_MAX_LFT = 1750
STEERING_TYP = 1500
STEERING_MIN_RGT = 1250

SHOOTER_TRIGGER_VAL = 2000
INTAKE_TRIGGER_VAL = 1000
TRIGGER_MARGIN = 50

# RC failsafe. chassis_1/2.py had none: if the transmitter was switched off or
# went out of range, the last commanded speed was held and the robot kept
# driving. Anything older than this and everything goes neutral.
RC_TIMEOUT = 0.5

# ==========================================================================
# HARDWARE
# ==========================================================================

pi = pigpio.pi()
if not pi.connected:
    print("E: Could not connect to pigpiod daemon! Run 'sudo pigpiod' first.")
    sys.exit(1)

left_motors = [Servo(ch) for ch in LEFT_DRIVE_CH]
right_motors = [Servo(ch) for ch in RIGHT_DRIVE_CH]
shooter_motors = [Servo(ch) for ch in SHOOTER_CH]
feeder_motor = Servo(FEEDER_CH)
intake_motor = Servo(INTAKE_CH)

running = threading.Event()
running.set()

# RC state
slider_pwm = 1500
switch_pwm = 1500
throttle_pwm = 1500
steering_pwm = 1500
start_ticks = {}
last_rc_time = 0.0

prev_left_mix = 0.0
prev_right_mix = 0.0

# Shooter state
_pulse_ticks = deque(maxlen=RPM_WINDOW_PULSES)
_pulse_lock = threading.Lock()
target_rpm = 0.0
measured_rpm = 0.0
shooter_angle = SHOOTER_STOP_ANGLE
trim = 0.0
integral_error = 0.0
previous_error = 0.0
at_speed_count = 0


def clamp(value, low, high):
    return max(low, min(high, value))


# ==========================================================================
# RC INPUT
# ==========================================================================

def rc_callback(gpio, level, tick):
    global slider_pwm, switch_pwm, throttle_pwm, steering_pwm, last_rc_time

    if level == 1:
        start_ticks[gpio] = tick
    elif level == 0 and gpio in start_ticks:
        width = pigpio.tickDiff(start_ticks[gpio], tick)
        if 900 <= width <= 2100:
            if gpio == SLIDER_PIN:
                slider_pwm = width
            elif gpio == SWITCH_PIN:
                switch_pwm = width
            elif gpio == THROTTLE_PIN:
                throttle_pwm = width
            elif gpio == STEERING_PIN:
                steering_pwm = width
            last_rc_time = time.perf_counter()


def rc_is_live():
    return (time.perf_counter() - last_rc_time) < RC_TIMEOUT


for pin in (SLIDER_PIN, SWITCH_PIN, THROTTLE_PIN, STEERING_PIN):
    pi.set_mode(pin, pigpio.INPUT)
    pi.callback(pin, pigpio.EITHER_EDGE, rc_callback)


# ==========================================================================
# ENCODER
# ==========================================================================

def encoder_callback(gpio, level, tick):
    with _pulse_lock:
        _pulse_ticks.append(tick)


pi.set_mode(ENCODER_A_PIN, pigpio.INPUT)
pi.set_pull_up_down(ENCODER_A_PIN, pigpio.PUD_UP)
pi.callback(ENCODER_A_PIN, pigpio.RISING_EDGE, encoder_callback)

pi.set_mode(ENCODER_B_PIN, pigpio.INPUT)
pi.set_pull_up_down(ENCODER_B_PIN, pigpio.PUD_UP)


def read_rpm():
    """Flywheel RPM from the spacing of recent encoder pulses.

    Uses pigpio edge callbacks rather than polling. encoderPIDRPM-2motor.py
    polled the pin in a 1 ms loop, which cannot keep up here: channel A emits
    ~700 edges/s at the motor's 6000 RPM free speed, so a polled loop silently
    drops pulses and reads LOW. On a shooter that is dangerous - the PID sees
    an under-speed flywheel and drives it harder.
    """
    with _pulse_lock:
        if len(_pulse_ticks) < 2:
            return 0.0
        ticks = list(_pulse_ticks)

    # pigpio ticks wrap every ~72 minutes; tickDiff handles the wrap.
    if pigpio.tickDiff(ticks[-1], pi.get_current_tick()) > RPM_STALE_S * 1e6:
        return 0.0

    span_us = pigpio.tickDiff(ticks[0], ticks[-1])
    if span_us <= 0:
        return 0.0

    revolutions = (len(ticks) - 1) / PULSES_PER_REV
    return revolutions / (span_us / 1e6) * 60.0


def muzzle_speed(rpm):
    return (rpm * 2.0 * math.pi / 60.0) * FLYWHEEL_RADIUS_M * LAUNCH_EFFICIENCY


# ==========================================================================
# SHOOTER CONTROL
# ==========================================================================

def slider_to_rpm():
    """Map the slider to a target RPM inside the legal band."""
    proportion = clamp((slider_pwm - 1000) / 1000.0, 0.0, 1.0)
    rpm = SLIDER_MIN_RPM + proportion * (SLIDER_MAX_RPM - SLIDER_MIN_RPM)
    return min(rpm, MAX_LEGAL_RPM)


def feedforward_angle(rpm):
    if not USE_FEEDFORWARD or rpm <= 0:
        return SHOOTER_MIN_ANGLE
    proportion = clamp(rpm / MOTOR_FREE_RPM, 0.0, 1.0)
    return SHOOTER_MIN_ANGLE + proportion * (SHOOTER_MAX_ANGLE - SHOOTER_MIN_ANGLE)


def write_shooter(angle):
    shooter_motors[0].angle(angle)
    if len(shooter_motors) > 1:
        second = -angle if SECOND_SHOOTER_REVERSED else angle
        shooter_motors[1].angle(second)


def shooter_control_loop():
    """Closed-loop flywheel speed. Runs in its own thread at CONTROL_INTERVAL."""
    global measured_rpm, shooter_angle, trim
    global integral_error, previous_error, at_speed_count

    while running.is_set():
        loop_start = time.perf_counter()
        measured_rpm = read_rpm()

        if target_rpm <= 0:
            # Coast down. Reset the PID so the next spin-up starts clean.
            trim = 0.0
            integral_error = 0.0
            previous_error = 0.0
            at_speed_count = 0
            shooter_angle = SHOOTER_STOP_ANGLE
            write_shooter(SHOOTER_STOP_ANGLE)
        else:
            error = target_rpm - measured_rpm
            if abs(error) < ERROR_DEADBAND:
                error = 0.0

            integral_error += error * CONTROL_INTERVAL
            # Anti-windup: the trim is clamped, so let the integral wind no
            # further than the trim can actually express.
            integral_error = clamp(integral_error, -MAX_TRIM / max(KI, 1e-9),
                                   MAX_TRIM / max(KI, 1e-9))

            derivative = error - previous_error
            previous_error = error

            output = (KP * error) + (KI * integral_error) + (KD * derivative)
            trim = clamp(trim + clamp(output, -MAX_TRIM_CHANGE, MAX_TRIM_CHANGE),
                         -MAX_TRIM, MAX_TRIM)

            # Overspeed guard on MEASURED speed. Independent of the angle model
            # above, so it still holds if the linear assumption is wrong.
            if measured_rpm > MAX_LEGAL_RPM:
                trim -= OVERSPEED_CUTBACK
                integral_error = 0.0

            shooter_angle = clamp(feedforward_angle(target_rpm) + trim,
                                  SHOOTER_MIN_ANGLE, SHOOTER_LEGAL_MAX_ANGLE)
            write_shooter(shooter_angle)

            if abs(target_rpm - measured_rpm) <= AT_SPEED_TOLERANCE:
                at_speed_count += 1
            else:
                at_speed_count = 0

        elapsed = time.perf_counter() - loop_start
        sleep(max(0.0, CONTROL_INTERVAL - elapsed))


def at_speed():
    return at_speed_count >= AT_SPEED_SAMPLES


# ==========================================================================
# SHOOTER / FEEDER / INTAKE  -  called every main-loop tick
# ==========================================================================

def update_manipulators():
    """Decide what the shooter, feeder and intake should do this tick.

    Switch positions are mutually exclusive (one 3-position switch), so exactly
    one of shooter / intake is ever active. The shooter flywheel is spun by the
    background PID thread; here we only set its TARGET and gate the feeder.

    The feeder (a continuous servo) runs ONLY while the flywheel is measured to
    be at speed. That single rule is what "only feed once the PID reaches target
    RPM" means in code: during spin-up, and during the RPM sag right after a
    ball loads the flywheel, the feeder is off, so no ball is ever pushed into a
    slow wheel and thrown short.
    """
    global target_rpm

    # Lost RC (rule 5.9 spirit): everything the shooter touches goes safe.
    if not rc_is_live():
        target_rpm = 0.0
        feeder_motor.speed(FEEDER_STOP)
        intake_motor.speed(INTAKE_STOP)
        return

    if shooter_is_triggered():
        target_rpm = slider_to_rpm()
        intake_motor.speed(INTAKE_STOP)          # never intake while shooting
        feeder_motor.speed(FEEDER_RUN if at_speed() else FEEDER_STOP)
    else:
        target_rpm = 0.0                         # spin the flywheel down
        feeder_motor.speed(FEEDER_STOP)
        if intake_is_triggered():
            intake_motor.speed(INTAKE_SPEED)
        else:
            intake_motor.speed(INTAKE_STOP)


def shooter_is_triggered():
    return switch_pwm >= (SHOOTER_TRIGGER_VAL - TRIGGER_MARGIN)


def intake_is_triggered():
    return switch_pwm <= (INTAKE_TRIGGER_VAL + TRIGGER_MARGIN)


# ==========================================================================
# CHASSIS
# ==========================================================================

def write_drive(left_angle, right_angle):
    for motor in left_motors:
        motor.angle(left_angle)
    for motor in right_motors:
        motor.angle(right_angle)


def update_chassis():
    global prev_left_mix, prev_right_mix

    if not rc_is_live():
        prev_left_mix = 0.0
        prev_right_mix = 0.0
        write_drive(MOTOR_NEUTRAL, MOTOR_NEUTRAL)
        return

    if throttle_pwm <= THROTTLE_TYP:
        div = THROTTLE_TYP - THROTTLE_MIN_FWD
        throttle = -(THROTTLE_TYP - throttle_pwm) / div if div else 0.0
    else:
        div = THROTTLE_MAX_REV - THROTTLE_TYP
        throttle = (throttle_pwm - THROTTLE_TYP) / div if div else 0.0

    if steering_pwm >= STEERING_TYP:
        div = STEERING_MAX_LFT - STEERING_TYP
        steering = (steering_pwm - STEERING_TYP) / div if div else 0.0
    else:
        div = STEERING_TYP - STEERING_MIN_RGT
        steering = -(STEERING_TYP - steering_pwm) / div if div else 0.0

    throttle = clamp(throttle, -1.0, 1.0)
    steering = clamp(steering, -1.0, 1.0)

    if abs(throttle) < 0.05:
        throttle = 0.0
    if abs(steering) < 0.05:
        steering = 0.0

    left_target = clamp(throttle + steering, -1.0, 1.0)
    right_target = clamp(throttle - steering, -1.0, 1.0)

    def limit_accel(prev, target):
        d = target - prev
        if d > ACCEL_LIMIT:
            return prev + ACCEL_LIMIT
        if d < -DECEL_LIMIT:
            return prev - DECEL_LIMIT
        return target

    left_mix = limit_accel(prev_left_mix, left_target)
    right_mix = limit_accel(prev_right_mix, right_target)
    prev_left_mix = left_mix
    prev_right_mix = right_mix

    span = MOTOR_MAX_FWD - MOTOR_NEUTRAL
    write_drive(MOTOR_NEUTRAL + left_mix * span,
                MOTOR_NEUTRAL - right_mix * span)


# ==========================================================================
# MAIN
# ==========================================================================

def all_neutral():
    write_drive(MOTOR_NEUTRAL, MOTOR_NEUTRAL)
    write_shooter(SHOOTER_STOP_ANGLE)
    feeder_motor.speed(FEEDER_STOP)
    intake_motor.speed(INTAKE_STOP)


def cleanup(shooter_thread=None):
    global target_rpm
    running.clear()
    target_rpm = 0.0
    if shooter_thread is not None:
        shooter_thread.join(timeout=2.0)
    all_neutral()
    pi.stop()
    print("\nAll motors neutral. STOP")


def main():
    all_neutral()

    print("=============================================")
    print("      TELEOP - DRIVE / INTAKE / SHOOTER      ")
    print("   Team Northeast   teleop_shooter.py v1.0   ")
    print("=============================================")
    print(f"Chassis layout : {CHASSIS_LAYOUT}")
    print(f"Encoder pins   : A={ENCODER_A_PIN} B={ENCODER_B_PIN} "
          f"(RC uses {SLIDER_PIN},{SWITCH_PIN},{STEERING_PIN},{THROTTLE_PIN})")
    print(f"Slider band    : {SLIDER_MIN_RPM:.0f} - {SLIDER_MAX_RPM:.0f} RPM")
    print(f"Legal ceiling  : {MAX_LEGAL_RPM:.0f} RPM "
          f"= {MAX_LEGAL_LAUNCH_MPS:.1f} m/s (rule 5.5)")
    print("Feeder         : continuous servo, runs only when flywheel at speed")
    print("=============================================")

    shooter_thread = threading.Thread(target=shooter_control_loop, daemon=True)
    shooter_thread.start()

    try:
        while running.is_set():
            update_chassis()
            update_manipulators()

            live = "RC OK " if rc_is_live() else "RC LOST"
            feed = "FEED" if (shooter_is_triggered() and at_speed()) else "----"
            sys.stdout.write(
                f"\r{live} | RPM {measured_rpm:6.0f}/{target_rpm:6.0f}"
                f" | {muzzle_speed(measured_rpm):5.2f} m/s"
                f" | angle {shooter_angle:5.2f} | trim {trim:+5.2f}"
                f" | {feed} | slider {slider_to_rpm():4.0f}   ")
            sys.stdout.flush()

            sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup(shooter_thread)


if __name__ == "__main__":
    main()
