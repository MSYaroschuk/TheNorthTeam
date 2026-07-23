#!/usr/bin/env python3

# auto_chassis.py
# Autonomous chassis control

# v2.2 - 07/22/2026 23:00
# (C) Team Northeast

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
model_path = "/home/Lucas/edge_impulse/yolo_pro_model.eim"

# select the model to use
model = input("Select 1 to run FOMO; 2 to run YOLO-Pro (default): ")

if model == '1':
    model_path = "/home/Lucas/edge_impulse/fomo_model_2.eim"

print(f"Running model: {model_path}")

# select the confidence threshold
CONFIDENCE_THRESHOLD = float(input("Enter confidence threshold: "))
print(f"Setting confidence: {CONFIDENCE_THRESHOLD}")

# no ball detections - wait then rotate to scan
NO_DETECT_WAIT_TIME = 5.0

# screen resolution
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

STOP_LINE_Y_PCT = 0.62

# Fusion HAT+ PWM output channels
SHOOTER_CH_1 = 7
SHOOTER_CH_2 = 6
GATE_CHANNEL = 5
INTAKE_CH = 4

MOTOR_LEFT_CH_1 = 0
MOTOR_LEFT_CH_2 = 1
MOTOR_RIGHT_CH_1 = 2
MOTOR_RIGHT_CH_2 = 3

# chassis settings
MOTOR_MAX_THROTTLE = 20  # max speed
MOTOR_NEUTRAL = 5        # stopping speed

# acceleration / deceleration limits per tick
ACCEL_LIMIT = 0.1
DECEL_LIMIT = 0.1

left_motor_1 = Servo(MOTOR_LEFT_CH_1)
left_motor_2 = Servo(MOTOR_LEFT_CH_2)
right_motor_1 = Servo(MOTOR_RIGHT_CH_1)
right_motor_2 = Servo(MOTOR_RIGHT_CH_2)

left_speed = 0.0
right_speed = 0.0

running = True

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

# EI inference loop
def inference():
    global running, left_speed, right_speed

    last_ball_time = time.perf_counter()

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

            print("Edge Impulse FOMO Centroid Detector Started! Press 'q' to quit.")

            window_name = "Edge Impulse Detection"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            prev_time = time.perf_counter()

            # detection loop
            while running:
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
                
                # draw stopping line
                stop_line_y = int(SCREEN_HEIGHT * STOP_LINE_Y_PCT)
                cv2.line(frame, (0, stop_line_y), (SCREEN_WIDTH, stop_line_y), (0, 0, 255), 2)

                # ball tracking logic if ball(s) detected
                if ball_list:
                    last_ball_time = time.perf_counter()  # to enable rotating when no balls detected for NO_DETECT_WAIT_TIME seconds
                    closest_ball = max(ball_list, key=lambda b: b.y)  # closest ball will be the target
                    # TODO: make cluster density optimized target
                    
                    cv2.putText(frame, f"Closest Ball: ({closest_ball.x}, {closest_ball.y})", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
                    cv2.circle(frame, (closest_ball.x, closest_ball.y), 10, (255, 0, 0), 3)

                    # stop line check
                    if closest_ball.y >= stop_line_y:
                        set_speed(0.0, 0.0)
                        cv2.putText(frame, "STOP LINE REACHED", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # proportional steering
                    else:
                        BASE_CRUISE_SPEED = 0.8  # base speed
                        STEERING_GAIN = 0.8      # steering correction
                        
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
                
                else:
                    no_ball_elapsed = time.perf_counter() - last_ball_time  # calculate time with no detections
                    cv2.putText(frame, f"No balls detected ({no_ball_elapsed:.1f}s)", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

                    if no_ball_elapsed > NO_DETECT_WAIT_TIME:  # greater than NO_DETECT_WAIT_TIME, rotate to scan
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
    # loop until speeds drop
    while abs(left_speed) > 0.01 or abs(right_speed) > 0.01:
        set_speed(0.0, 0.0)
        sleep(0.05)

    # stop all
    left_motor_1.angle(MOTOR_NEUTRAL)
    left_motor_2.angle(MOTOR_NEUTRAL)
    right_motor_1.angle(MOTOR_NEUTRAL)
    right_motor_2.angle(MOTOR_NEUTRAL)
    print("Motors stopped")

def main():
    left_motor_1.angle(MOTOR_NEUTRAL)
    left_motor_2.angle(MOTOR_NEUTRAL)
    right_motor_1.angle(MOTOR_NEUTRAL)
    right_motor_2.angle(MOTOR_NEUTRAL)

    print("=============================================")
    print("         AUTONOMOUS BALL TRACKING            ")
    print(" Team Northeast  auto_chassis.py  07/22/2026 ")
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
