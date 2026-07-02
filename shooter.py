#Launcher code
#!/usr/bin/env python3

# Sequence: wait for trigger (PWM == 2000) -> spin shooter to full speed
#           -> wait for spin-up -> open gate for X sec -> close gate -> stop shooter

from fusion_hat.servo import Servo
from time import sleep

# ---------------------------------------------------------------
# SETTINGS TO TUNE
# ---------------------------------------------------------------
SHOOTER_CHANNEL1 = 0  # PWM channel the shooter motor/ESC is on
SHOOTER_CHANNEL2 = 1  # PWM channel the shooter motor/ESC is on
GATE_CHANNEL = 2  # PWM channel the gate (feeder) servo is on

# Fusion HAT angle range is -90 to +90.
# "180 = full speed" -> +90.  If it spins the wrong way ("inverted"),
# change SHOOTER_FULL to -90 (no speed).
SHOOTER_FULL1 = -10  # full speed.  If wrong direction, use -90
SHOOTER_STOP1 = 5  # stop / neutral
SHOOTER_FULL2 = -4  # full speed.  If wrong direction, use -90
SHOOTER_STOP2 = 5  # stop / neutral

GATE_OPEN = 0  # gate open position
GATE_CLOSED = 55  # gate closed position

# Timing (seconds)
SPINUP_TIME = 3  # wait after starting motor before opening gate ( “3 sec")
GATE_OPEN_TIME = 4.0  # how long the gate stays open  ("amount of time")

# Trigger
TRIGGER_VALUE = 2000  # fire when the PWM signal reaches this
TRIGGER_MARGIN = 50  # allow a little jitter (fires at >= 1950)

# ---------------------------------------------------------------
# HARDWARE OBJECTS
# ---------------------------------------------------------------
shooter1 = Servo(SHOOTER_CHANNEL1)
shooter2 = Servo(SHOOTER_CHANNEL2)
gate = Servo(GATE_CHANNEL)


# ---------------------------------------------------------------
# >> NEED TO FILL THIS IN  For PWM<<<
# Return the current incoming PWM value as a number.
# The Fusion HAT does PWM OUTPUT, so reading an incoming 2000 signal
# depends on:
#   - an RC receiver channel wired to a GPIO pin, measured with pigpio/gpiozero
#   - your own variable set elsewhere in the program
# For now this returns 0 (never fires) so the program is safe to run.
# ---------------------------------------------------------------
def read_trigger_pwm():
    return 0


def is_triggered():
    return read_trigger_pwm() >= (TRIGGER_VALUE - TRIGGER_MARGIN)


# ---------------------------------------------------------------
# THE FIRE SEQUENCE
# ---------------------------------------------------------------
def fire():
    print("Trigger detected -> starting shooter")
    shooter1.angle(SHOOTER_FULL1)  # motor ON at full speed
    shooter2.angle(SHOOTER_FULL2)  # motor ON at full speed
    sleep(SPINUP_TIME)  # let it spin up ("2-3 sec after motor starting")

    print("Opening gate")
    gate.angle(GATE_OPEN)  # open the feeder
    sleep(GATE_OPEN_TIME)  # keep open for X seconds

    print("Closing gate")
    gate.angle(GATE_CLOSED)  # close the feeder

    print("Stopping shooter")
    shooter1.angle(SHOOTER_STOP1)  # stop the motor
    shooter2.angle(SHOOTER_STOP2)  # stop the motor


def cleanup():
    gate.angle(GATE_CLOSED)
    shooter1.angle(SHOOTER_STOP1)
    shooter2.angle(SHOOTER_STOP2)
    sleep(0.1)


# ---------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------
def main():
    # to make sure we start in a safe state
    shooter1.angle(SHOOTER_STOP1)
    shooter2.angle(SHOOTER_STOP2)
    gate.angle(GATE_CLOSED)
    # print("Ready. Waiting for trigger (PWM == %d)..." % TRIGGER_VALUE)

    while True:
        #        if is_triggered():
        fire()
        # wait until the trigger is released so it doesn't fire repeatedly
        while is_triggered():
            sleep(0.05)
        print("Ready. Waiting for trigger...")
        sleep(5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping safely.")
        cleanup()
