#!/usr/bin/env python3

# auto_chassis.py
# Autonomous chassis and intake

# v2.5 - Intake improvements
# (C) Team Northeast

# pi4 main bot #1
# 07/29/2026

import time
import threading
from time import sleep
import pigpio
from fusion_hat.servo import Servo

import cv2
import numpy as np
from picamera2 import Picamera2
from edge_impulse_linux.image import ImageImpulseRunner

from dataclasses import dataclass

# default .eim model path
model_path = "/home/pi/northeast/yolo_pro_model.eim"

# select the model to use
model = input("Select 1 to run FOMO; 2 to run YOLO-Pro (default): ")

if model == '1':
    model_path = "/home/pi/northeast/fomo_model_2.eim"

print(f"Running model: {model_path}")

# select the confidence threshold
CONFIDENCE_THRESHOLD = float(input("Enter confidence threshold: "))
print(f"Setting confidence: {CONFIDENCE_THRESHOLD}")

# no ball detections - wait then rotate to scan
NO_DETECT_WAIT_TIME = 5.0

# screen resolution
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

STOP_LINE_Y_PCT = 0.70

# Fusion HAT+ PWM output channels
SHOOTER_CH_1 = 7
SHOOTER_CH_2 = 6
GATE_CHANNEL = 5
INTAKE_CH = 3

MOTOR_LEFT_CH_1 = 0
MOTOR_LEFT_CH_2 = 8
MOTOR_RIGHT_CH_1 = 1
MOTOR_RIGHT_CH_2 = 9

# speed settings
MOTOR_MAX_THROTTLE = 20  # max speed
MOTOR_NEUTRAL = 5        # stopping speed
INTAKE_SPEED = -30

# acceleration / deceleration limits per tick
ACCEL_LIMIT = 0.1
DECEL_LIMIT = 0.1

left_motor_1 = Servo(MOTOR_LEFT_CH_1)
left_motor_2 = Servo(MOTOR_LEFT_CH_2)
right_motor_1 = Servo(MOTOR_RIGHT_CH_1)
right_motor_2 = Servo(MOTOR_RIGHT_CH_2)
intake_motor = Servo(INTAKE_CH)

left_speed = 0.0
right_speed = 0.0

running = True
intake_running = False

# detected object struct
@dataclass
class DetectedObject:
    x: int
    y: int
    confidence: float
    label: str

# list of detected balls
ball_list = []

# set the speed of motors
def set_speed(left_target, right_target):
    global left_speed, right_speed

    left_target = max(-1.0, min(1.0, left_target))
    right_target = max(-1.0, min(1.0, right_target))

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
    
    left_speed = limit_accel(left_speed, left_target)
    right_speed = limit_accel(right_speed, right_target)

    left_angle = MOTOR_NEUTRAL + (left_speed * MOTOR_MAX_THROTTLE)
    right_angle = MOTOR_NEUTRAL - (right_speed * MOTOR_MAX_THROTTLE)

    left_motor_1.angle(left_angle)
    left_motor_2.angle(left_angle)
    right_motor_1.angle(right_angle)
    right_motor_2.angle(right_angle)

def intake_start():
    global intake_running
    intake_motor.angle(INTAKE_SPEED)
    intake_running = True

def intake_stop():
    global intake_running
    intake_motor.angle(MOTOR_NEUTRAL)
    intake_running = False

# EI inference loop
def inference():
    global running, left_speed, right_speed

    last_ball_time = time.perf_counter()
    intake_until_time = 0.0  # Non-blocking intake timer

    with ImageImpulseRunner(model_path) as runner:
        try:
            # load the model
            model_info = runner.init()
            print(f'Loaded model: {model_info["project"]["owner"]} / {model_info["project"]["name"]}')
            
            model_width = model_info["model_parameters"]["image_input_width"]
            model_height = model_info["model_parameters"]["image_input_height"]

            picam2 = Picamera2()
            picam2.preview_configuration.main.size = (SCREEN_WIDTH, SCREEN_HEIGHT)
            picam2.preview_configuration.main.format = "RGB888" 
            picam2.preview_configuration.align()
            picam2.configure("preview")
            picam2.start()

            print("Edge Impulse Detector Started! Press 'q' to quit.")

            window_name = "Edge Impulse Detection"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            prev_time = time.perf_counter()

            # detection loop
            while running:
                now = time.perf_counter()
                ball_list.clear()

                frame = picam2.capture_array()
                frame_h, frame_w, _ = frame.shape
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                features, cropped = runner.get_features_from_image(rgb_frame)
                
                start_inference = time.perf_counter()
                res = runner.classify(features)
                end_inference = time.perf_counter()
                
                inference_time_ms = (end_inference - start_inference) * 1000

                if "bounding_boxes" in res["result"]:
                    for bb in res["result"]["bounding_boxes"]:
                        score = bb['value']
                        if score < CONFIDENCE_THRESHOLD:  # ignore low confidence
                            continue
                        
                        # scale up model resolution to camera input size
                        scale_x = frame_w / model_width
                        scale_y = frame_h / model_height
                        
                        # get centroid
                        center_x = int((bb['x'] + (bb['width'] / 2)) * scale_x)
                        center_y = int((bb['y'] + (bb['height'] / 2)) * scale_y)
                        
                        label = bb['label'].lower()

                        obj = DetectedObject(center_x, center_y, score, label)
                        
                        if "ball" in label:
                            color = (0, 255, 255)
                            display_name = "Tennis Ball"
                            ball_list.append(obj)
                        elif "blue" in label:
                            color = (0, 100, 255)
                            display_name = "Blue Bucket"
                        elif "orange" in label:
                            color = (255, 50, 50)
                            display_name = "Orange Bucket"
                        else:
                            color = (0, 255, 0)
                            display_name = label.capitalize()

                        cv2.circle(frame, (center_x, center_y), 6, color, -1)
                        cv2.circle(frame, (center_x, center_y), 18, color, 3)
                        
                        text = f"{display_name} ({score:.2f})"
                        cv2.putText(frame, text, (center_x + 15, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # draw stopping line & pre-intake threshold
                stop_line_y = int(SCREEN_HEIGHT * STOP_LINE_Y_PCT)
                cv2.line(frame, (0, stop_line_y), (SCREEN_WIDTH, stop_line_y), (0, 0, 255), 2)

                # ball tracking logic if ball(s) detected
                if ball_list:
                    last_ball_time = now
                    closest_ball = max(ball_list, key=lambda b: b.y)  # closest ball is target
                    
                    cv2.putText(frame, f"Closest Ball: ({closest_ball.x}, {closest_ball.y})", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
                    cv2.circle(frame, (closest_ball.x, closest_ball.y), 10, (255, 0, 0), 3)

                    # INTAKE BALL: ball passed stop line
                    if closest_ball.y >= stop_line_y:
                        intake_until_time = now + 7.0  # Keep intake running for 0.8s without blocking
                        intake_start()
                        status = "INTAKING"

                    # APPROACHING BALL: proportional steering
                    else:
                        status = "TRACKING"
                        
                        # Smooth speed reduction: slow down as ball gets closer so intake can grab it
                        proximity_ratio = min(1.0, closest_ball.y / stop_line_y)
                        
                        BASE_CRUISE_SPEED = 0.75 - (0.35 * proximity_ratio)
                        STEERING_GAIN = 0.75
                        
                        # calculate error
                        screen_center_x = SCREEN_WIDTH // 2
                        error_x = closest_ball.x - screen_center_x
                        normalized_error = error_x / screen_center_x
                        
                        steering_correction = normalized_error * STEERING_GAIN
                        
                        # calculate left/right speeds
                        target_left = BASE_CRUISE_SPEED + steering_correction
                        target_right = BASE_CRUISE_SPEED - steering_correction
                        
                        # set the speed
                        set_speed(target_left, target_right)
                        
                        dir_text = "RIGHT" if error_x > 0 else "LEFT" if error_x < 0 else "STRAIGHT"
                        cv2.putText(frame, f"TRACKING: {dir_text} (Err: {error_x})", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                        cv2.putText(frame, f"Steer Correction: {steering_correction:.2f}", (15, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                # No balls detected
                else:
                    no_ball_elapsed = now - last_ball_time
                    cv2.putText(frame, f"No balls detected ({no_ball_elapsed:.1f}s)", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

                    # Maintain active intake for brief run-on window after ball vanishes under intake hood
                    if now <= intake_until_time:
                        intake_start()
                        set_speed(0.9, 0.6)  # Gentle forward glide to finish pulling in the ball
                        cv2.putText(frame, "INTAKE RUN-ON", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    else:
                        intake_stop()
                        if no_ball_elapsed > NO_DETECT_WAIT_TIME:  # Rotate to scan
                            set_speed(0.25, -0.25)
                            cv2.putText(frame, "SCANNING...", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        else:
                            set_speed(0.0, 0.0)
                            cv2.putText(frame, "WAITING...", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

                # performance log
                end_loop_time = time.perf_counter()
                loop_fps = 1.0 / max(0.001, (end_loop_time - prev_time))
                prev_time = end_loop_time

                cv2.putText(frame, f"FPS: {loop_fps:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
                cv2.putText(frame, f"Inference: {inference_time_ms:.1f} ms", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
                cv2.putText(frame, f"Left: {left_speed:.2f} | Right: {right_speed:.2f}", (15, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                
                intake_text = "ON" if intake_running else "OFF"
                cv2.putText(frame, f"Intake: {intake_text}", (15, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.imshow(window_name, frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    running = False
                    break

        finally:
            print("Cleaning up...")
            cv2.destroyAllWindows()
            picam2.stop()
            runner.stop()

def cleanup():
    global running
    running = False
    print("\nDecelerating motors...")
    while abs(left_speed) > 0.01 or abs(right_speed) > 0.01:
        set_speed(0.0, 0.0)
        sleep(0.05)

    left_motor_1.angle(MOTOR_NEUTRAL)
    left_motor_2.angle(MOTOR_NEUTRAL)
    right_motor_1.angle(MOTOR_NEUTRAL)
    right_motor_2.angle(MOTOR_NEUTRAL)
    intake_stop()

    print("Motors stopped")

def main():
    left_motor_1.angle(MOTOR_NEUTRAL)
    left_motor_2.angle(MOTOR_NEUTRAL)
    right_motor_1.angle(MOTOR_NEUTRAL)
    right_motor_2.angle(MOTOR_NEUTRAL)
    intake_stop()

    print("=============================================")
    print("         AUTONOMOUS BALL INTAKING            ")
    print("    Team Northeast  auto_chassis.py  v2.5    ")
    print("=============================================")

    inference_thread = threading.Thread(target=inference, daemon=True)
    inference_thread.start()

    try:
        while running:
            sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()

if __name__ == "__main__":
    main()

