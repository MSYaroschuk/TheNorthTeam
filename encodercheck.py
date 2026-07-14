#!/usr/bin/env python3
import time
import pigpio

# Define the GPIO pins connected to the encoder
CHANNEL_A = 17
CHANNEL_B = 27

# Initialize the global tick counter
pos = 0

# Callback function to handle encoder state changes
def encoder_callback(gpio, level, tick):
    global pos
    # Read the current states of both channels
    a_state = pi.read(CHANNEL_A)
    b_state = pi.read(CHANNEL_B)
    
    # Determine direction based on quadrature phasing
    if a_state == b_state:
        pos += 1
    else:
        pos -= 1

# Start the pigpio connection
pi = pigpio.pi()
if not pi.connected:
    exit()

# Set up pin modes and enable internal pull-up resistors
pi.set_mode(CHANNEL_A, pigpio.INPUT)
pi.set_mode(CHANNEL_B, pigpio.INPUT)
pi.set_pull_up_down(CHANNEL_A, pigpio.PUD_UP)
pi.set_pull_up_down(CHANNEL_B, pigpio.PUD_UP)

# Attach interrupts to both channels, triggering on any edge
cb_a = pi.callback(CHANNEL_A, pigpio.EITHER_EDGE, encoder_callback)
cb_b = pi.callback(CHANNEL_B, pigpio.EITHER_EDGE, encoder_callback)

try:
    print("Reading encoder... Press Ctrl+C to stop.")
    while True:
        print(f"Encoder Ticks: {pos}")
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nExiting...")
    cb_a.cancel()
    cb_b.cancel()
    pi.stop()