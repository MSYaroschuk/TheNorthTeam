#!/usr/bin/env python3

# chassis_2.py
# Last updated to Pi 07/03 3:40 PM
# (C) 07/13/2026 Team East, All Rights Reserved

import time
import threading
from time import sleep
import pigpio
from fusion_hat.servo import Servo

# GPIO inputs
SLIDER_PIN = 4           # flywheel speed slider
SWITCH_PIN = 17          # 3-value trigger switch
STEERING_PIN = 22        # left/right joystick
THROTTLE_PIN = 27        # forward/backward joystick

# Fusion HAT+ PWM output channels
SHOOTER_CH_1 = 0
SHOOTER_CH_2 = 1
GATE_CHANNEL = 2
INTAKE_CH = 3
MOTOR_LEFT_CH = 4
MOTOR_RIGHT_CH = 5

# Shooter settings
SHOOTER_MAX_SPEED = -20  # Fast speed angle
SHOOTER_STOP = 5         # Stop / neutral angle
GATE_OPEN = 0            # Gate open position
GATE_CLOSED = 100        # Gate closed position

SPINUP_TIME = 2.0        # Time flywheels spin up before gate opens
GATE_OPEN_TIME = 0.217   # Time gate stays open

SHOOTER_TRIGGER_VAL = 2000

# Intake settings
INTAKE_TRIGGER_VAL = 1000
INTAKE_SPEED = 30
INTAKE_STOP = 5

TRIGGER_MARGIN = 50      # Fires if switch reads +- 50 of the trigger value

# Chassis settings
MOTOR_MAX_FWD = 90
MOTOR_MAX_REV = -90
MOTOR_NEUTRAL = 5        # Stopping speed

# Remote controller input values
THROTTLE_MIN_FWD = 1250
THROTTLE_TYP     = 1500
THROTTLE_MAX_REV = 1750

STEERING_MAX_LFT = 1750
STEERING_TYP     = 1500
STEERING_MIN_RGT = 1250

# Acceleration / deceleration limits
ACCEL_LIMIT = 0.014
DECEL_LIMIT = 0.014

# pigpiod check
pi = pigpio.pi()
if not pi.connected:
    print("E: Could not connect to pigpiod daemon! Run 'sudo pigpiod' first.")
    exit(1)

shooter_1 = Servo(SHOOTER_CH_1)
shooter_2 = Servo(SHOOTER_CH_2)
gate = Servo(GATE_CHANNEL)
left_motor = Servo(MOTOR_LEFT_CH)
right_motor = Servo(MOTOR_RIGHT_CH)
intake_motor = Servo(INTAKE_CH)

# PWM values
slider_pwm = 1500
switch_pwm = 1500 
throttle_pwm = 1500
steering_pwm = 1490

start_ticks = {}
firing_in_progress = False
running = True
prev_left_mix = 0.0
prev_right_mix = 0.0

# PWM callback
def recv_hw_cb(gpio, level, tick):
    global slider_pwm, switch_pwm, throttle_pwm, steering_pwm, start_ticks
    
    if level == 1:
        start_ticks[gpio] = tick
    elif level == 0:
        if gpio in start_ticks:
            width = pigpio.tickDiff(start_ticks[gpio], tick)
            if 900 <= width <= 2100:
                if gpio == SLIDER_PIN: slider_pwm = width
                elif gpio == SWITCH_PIN: switch_pwm = width
                elif gpio == THROTTLE_PIN: throttle_pwm = width
                elif gpio == STEERING_PIN: steering_pwm = width

# Add the callback function
for pin in [SLIDER_PIN, SWITCH_PIN, THROTTLE_PIN, STEERING_PIN]:
    pi.set_mode(pin, pigpio.INPUT)
    pi.callback(pin, pigpio.EITHER_EDGE, recv_hw_cb)

# Shooter sequence
def async_fire_sequence():
    global firing_in_progress
    firing_in_progress = True
    
    try:
        val = max(1000, min(2000, slider_pwm))
        proportion = (val - 1000) / 1000.0
        target_angle = SHOOTER_STOP + proportion * (SHOOTER_MAX_SPEED - SHOOTER_STOP)
        
        print(f"\n[SHOOTER] Spinning up flywheels to: {target_angle:.1f}")
        shooter_1.angle(target_angle)
        shooter_2.angle(target_angle)
        sleep(SPINUP_TIME)

        print("[SHOOTER] Opening feed gate")
        gate.angle(GATE_OPEN)
        sleep(GATE_OPEN_TIME)

        print("[SHOOTER] Closing feed gate")
        gate.angle(GATE_CLOSED)
        sleep(1)

        print("[SHOOTER] Stopping flywheels\n")
        shooter_1.angle(SHOOTER_STOP)
        shooter_2.angle(SHOOTER_STOP)
        
    finally:
        print("[SHOOTER] Done, reset physical switch now")
        while shooter_is_triggered() and running:
            sleep(0.1)
        firing_in_progress = False

def shooter_is_triggered():
    return switch_pwm >= (SHOOTER_TRIGGER_VAL - TRIGGER_MARGIN)

def intake_is_triggered():
    return switch_pwm <= (INTAKE_TRIGGER_VAL + TRIGGER_MARGIN)

# Chassis movement
def update_chassis():    
    # 1. Map Throttle (1250 Low is Forward (+1.0), 1750 High is Backward (-1.0))
    if throttle_pwm <= THROTTLE_TYP:
        # Scale 1500 down to 1250 as 0.0 to +1.0
        div = (THROTTLE_TYP - THROTTLE_MIN_FWD)
        throttle = -(THROTTLE_TYP - throttle_pwm) / div if div != 0 else 0
    else:
        # Scale 1500 up to 1750 as 0.0 to -1.0
        div = (THROTTLE_MAX_REV - THROTTLE_TYP)
        throttle = (throttle_pwm - THROTTLE_TYP) / div if div != 0 else 0

    # 2. Map Steering (1730 High is Left (+1.0), 1230 Low is Right (-1.0))
    if steering_pwm >= STEERING_TYP:
        # Scale 1490 up to 1730 as 0.0 to +1.0
        div = (STEERING_MAX_LFT - STEERING_TYP)
        steering = (steering_pwm - STEERING_TYP) / div if div != 0 else 0
    else:
        # Scale 1490 down to 1230 as 0.0 to -1.0
        div = (STEERING_TYP - STEERING_MIN_RGT)
        steering = -(STEERING_TYP - steering_pwm) / div if div != 0 else 0

    # Strict software boundaries
    throttle = max(-1.0, min(1.0, throttle))
    steering = max(-1.0, min(1.0, steering))

    # Deadzone 
    if abs(throttle) < 0.05: throttle = 0.0
    if abs(steering) < 0.05: steering = 0.0

    # 3. Arcade Mixing Math
    left_mix_target = max(-1.0, min(1.0, throttle + steering))
    right_mix_target = max(-1.0, min(1.0, throttle - steering))

    # 3b. Apply acceleration/deceleration limiter to prevent sudden changes
    global prev_left_mix, prev_right_mix

    def limit_accel(prev, target):
        d = target - prev
        if d > 0:
            max_d = ACCEL_LIMIT
            if d > max_d:
                return prev + max_d
        else:
            max_d = DECEL_LIMIT
            if d < -max_d:
                return prev - max_d
        return target

    left_mix = limit_accel(prev_left_mix, left_mix_target)
    right_mix = limit_accel(prev_right_mix, right_mix_target)

    # store for next iteration
    prev_left_mix = left_mix
    prev_right_mix = right_mix

    # 4. Map normalized calculations directly to hardware angles
    left_angle = MOTOR_NEUTRAL + (left_mix * (MOTOR_MAX_FWD - MOTOR_NEUTRAL))
    right_angle = MOTOR_NEUTRAL + (-right_mix * (MOTOR_MAX_FWD - MOTOR_NEUTRAL))

    # 5. Output commands to the Fusion Hat (with the reversed right motor inverted)
    left_motor.angle(left_angle)
    right_motor.angle(right_angle)

# Cleanup when stopped
def cleanup():
    global running
    running = False
    sleep(0.2)
    gate.angle(GATE_CLOSED)
    shooter_1.angle(SHOOTER_STOP)
    shooter_2.angle(SHOOTER_STOP)
    left_motor.angle(MOTOR_NEUTRAL)
    right_motor.angle(MOTOR_NEUTRAL)
    intake_motor.speed(INTAKE_STOP)
    pi.stop()
    print("\nSTOP")

def main():
    # Structural safe startup state forcing absolute stops
    gate.angle(GATE_CLOSED)
    shooter_1.angle(SHOOTER_STOP)
    shooter_2.angle(SHOOTER_STOP)
    left_motor.angle(MOTOR_NEUTRAL)
    right_motor.angle(MOTOR_NEUTRAL)
    intake_motor.speed(INTAKE_STOP)
    
    print("=============================================")
    print("          CHASSIS & SHOOTER CONTROL          ")
    print("  Team East  chassis_2.py  07/03/2026 15:40  ")
    print("=============================================")

    while True:
        # Drive functions update fast (every 20ms)
        update_chassis()

        # Handle independent shooter trigger checks
        if shooter_is_triggered() and not firing_in_progress:
            shooter_thread = threading.Thread(target=async_fire_sequence, daemon=True)
            shooter_thread.start()

        # Handle intake trigger checks
        if intake_is_triggered() and not firing_in_progress:
            print("[INTAKE] Intake triggered")
            intake_motor.speed(INTAKE_SPEED)
        else:
            intake_motor.speed(INTAKE_STOP)

        sleep(0.02)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cleanup()