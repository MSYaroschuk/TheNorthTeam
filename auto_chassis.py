#!/usr/bin/env python3

# auto_chassis.py
# Autonomous chassis and intake

# v2.4 - Smooth Non-Blocking Intake Update
# (C) Team Northeast

import time
import threading
from time import sleep
from fusion_hat.servo import Servo

import cv2
import numpy as np
from picamera2 import Picamera2
from picamera2.devices import IMX500

from dataclasses import dataclass

# IMX500 AI Camera: the network runs ON THE SENSOR, so there is no model
# picker any more - one .rpk is loaded into the camera at open time.
NETWORK_PATH = "/home/pi/northeast/imx500/network.rpk"
LABELS_PATH = "/home/pi/northeast/imx500/labels.txt"

print(f"Loading network onto sensor: {NETWORK_PATH}")

# select the confidence threshold
# ponytail: the export ran NMS on-sensor at conf=0.001, so the sensor emits
# nearly everything and THIS is the only real filter. Do not set it to 0.
CONFIDENCE_THRESHOLD = float(input("Enter confidence threshold: "))
print(f"Setting confidence: {CONFIDENCE_THRESHOLD}")

# no ball detections - wait then rotate to scan
NO_DETECT_WAIT_TIME = 5.0

# screen resolution
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

STOP_LINE_Y_PCT = 0.70

# ponytail: a real ball persists across frames, a false positive flickers.
# Confidence alone cannot separate them - time can. Tune these, not the threshold.
BALL_CONFIRM_FRAMES = 2    # consecutive hits before we act on a ball
BALL_MEMORY_S = 0.4        # keep steering at a remembered ball this long after it drops out
BALL_MATCH_PX = 80         # a hit within this many px counts as the same ball

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
INTAKE_SPEED = -20

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
inference_thread = None   # set by main(); cleanup() waits on it

def unpack_edgemdt_nms(outputs):
    """4 sensor tensors -> (boxes, scores, labels, n). Copied from RoverSoftware
    robot/sensors/imx500.py, which solved this properly - do not "simplify" it.

    Our YOLO export wraps the network in edge-mdt NMS, so NMS runs on-sensor and
    FOUR tensors come out (not the model-zoo three):
        boxes   (max_det, 4)  xyxy corners, in NETWORK INPUT pixels
        scores  (max_det,)    descending
        labels  (max_det,)    class index
        n_valid (1,)          how many are real; the rest are zero padding that
                              would otherwise decode into junk boxes at (0,0)

    Tensors are identified BY SHAPE, not position: the ONNX graph and the
    converter's dnnParams.xml list them in opposite orders and which one the
    sensor follows is only observable on-device. boxes is the only 2-D tensor
    and n_valid the only 1-element one; where those two land tells us whether
    the list is forward or reversed, which settles scores vs labels too.
    """
    if outputs is None or len(outputs) != 4:
        return None
    arrs = [o[0] if getattr(o, "ndim", 0) and o.shape[0] == 1 and o.ndim > 1 else o
            for o in outputs]
    box_at = [i for i, a in enumerate(arrs) if a.ndim == 2 and a.shape[-1] == 4]
    one_at = [i for i, a in enumerate(arrs) if a.size == 1]
    if len(box_at) != 1 or len(one_at) != 1:
        return None
    bi, ni = box_at[0], one_at[0]
    if bi == 0 and ni == 3:
        boxes, scores, labels, n_valid = arrs      # as the ONNX graph declares
    elif bi == 3 and ni == 0:
        n_valid, labels, scores, boxes = arrs      # as dnnParams.xml lists
    else:
        return None
    n = int(n_valid.reshape(-1)[0])
    n = max(0, min(n, len(scores)))
    return boxes[:n], scores[:n], labels[:n], n


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def parse_detections(imx500, picam2, metadata, labels, frame_w, frame_h):
    """One frame's sensor metadata -> [(cx, cy, label, score), ...] in frame px.

    Returns centroids because that is all the control loop uses. The sensor
    reports boxes in letterboxed network-input space; convert_inference_coords()
    maps them back against the SAME request's metadata, so unlike the Edge
    Impulse path there is no manual scale_x/scale_y to get wrong.
    """
    outputs = imx500.get_outputs(metadata, add_batch=True)
    if outputs is None:
        return []                    # no inference on this frame - normal at startup

    nms = unpack_edgemdt_nms(outputs)
    if nms is None:
        return []
    boxes, scores, classes, _n = nms

    input_w, input_h = imx500.get_input_size()
    # Corner pixels in network input space -> the normalized (y0,x0,y1,x1)
    # that convert_inference_coords() expects.
    boxes = np.asarray(boxes, dtype=np.float32) / float(input_h)
    boxes = boxes[:, [1, 0, 3, 2]]
    boxes = zip(*np.array_split(boxes, 4, axis=1))

    out = []
    for coords, score, category in zip(boxes, scores, classes):
        if score < CONFIDENCE_THRESHOLD:
            continue
        try:
            label = labels[int(category)].lower()
        except (IndexError, ValueError):
            label = str(int(category))
        x, y, w, h = imx500.convert_inference_coords(coords, metadata, picam2)
        # Clip: the network sees a letterboxed field the ISP output does not
        # exactly cover, so a box can land partly outside the frame.
        x = int(_clamp(x, 0, max(frame_w - 1, 0)))
        y = int(_clamp(y, 0, max(frame_h - 1, 0)))
        w = int(_clamp(w, 0, frame_w - x))
        h = int(_clamp(h, 0, frame_h - y))
        if w <= 0 or h <= 0:
            continue
        out.append((x + w // 2, y + h // 2, label, float(score)))
    return out


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

    # ponytail: 3 vars instead of a tracker class. Upgrade to a real
    # tracker only if we ever need to follow more than one ball.
    confirmed_ball = None    # last ball we trust
    confirmed_at = 0.0       # when we last saw it
    hit_streak = 0           # consecutive frames matching confirmed_ball

    picam2 = None   # ponytail: finally runs even if construction throws



    # Loading the .rpk pushes ~3MB into the sensor over I2C - slow, once, here.
    imx500 = IMX500(NETWORK_PATH)
    with open(LABELS_PATH) as f:
        labels = [ln.strip() for ln in f if ln.strip()]
    print(f"Labels: {labels}")

    # ponytail: was `with ImageImpulseRunner(...)`. Kept as a bare block so the
    # whole body did not need re-indenting for the swap.
    if True:
        try:
            picam2 = Picamera2(imx500.camera_num)
            config = picam2.create_preview_configuration(
                main={"size": (SCREEN_WIDTH, SCREEN_HEIGHT), "format": "RGB888"},
                buffer_count=6)
            picam2.start(config)
            imx500.show_network_fw_progress_bar()

            print("Edge Impulse Detector Started! Press 'q' to quit.")

            window_name = "Edge Impulse Detection"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, SCREEN_WIDTH, SCREEN_HEIGHT)  # ponytail: fullscreen off, so size it to the frame
#            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            prev_time = time.perf_counter()

            # detection loop
            while running:
                now = time.perf_counter()
                ball_list.clear()

                # One request carries the frame AND the boxes the sensor already
                # produced for it - they cannot drift apart the way a separate
                # capture + CPU inference could.
                request = picam2.capture_request()
                try:
                    frame = request.make_array("main")
                    metadata = request.get_metadata()
                finally:
                    request.release()
                frame_h, frame_w, _ = frame.shape

                start_decode = time.perf_counter()
                detections = parse_detections(
                    imx500, picam2, metadata, labels, frame_w, frame_h)
                inference_time_ms = (time.perf_counter() - start_decode) * 1000

                if True:
                    for center_x, center_y, label, score in detections:
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
                # ponytail: use the real frame size - picam2 align() may not
                # hand back exactly SCREEN_WIDTH x SCREEN_HEIGHT.
                stop_line_y = int(frame_h * STOP_LINE_Y_PCT)
                cv2.line(frame, (0, stop_line_y), (frame_w, stop_line_y), (0, 0, 255), 2)

                # ponytail: confirm-then-remember. A hit near the last known ball
                # builds a streak; a ball is only acted on once the streak is long
                # enough. Dropouts decay instead of clearing, so one missed frame
                # does not brake the robot. Kills phantom chasing AND lurching.
                raw_ball = max(ball_list, key=lambda b: b.y) if ball_list else None

                if raw_ball is not None:
                    same_ball = (
                        confirmed_ball is not None
                        and abs(raw_ball.x - confirmed_ball.x) <= BALL_MATCH_PX
                        and abs(raw_ball.y - confirmed_ball.y) <= BALL_MATCH_PX
                    )
                    hit_streak = hit_streak + 1 if same_ball else 1
                    confirmed_ball = raw_ball
                    confirmed_at = now
                elif hit_streak >= BALL_CONFIRM_FRAMES:
                    # already trusted: coast through a short dropout
                    if (now - confirmed_at) > BALL_MEMORY_S:
                        confirmed_ball = None   # memory expired, really gone
                        hit_streak = 0
                else:
                    # ponytail: unconfirmed candidate. A miss breaks the streak at
                    # once - otherwise two phantoms BALL_MEMORY_S apart confirm
                    # each other and "consecutive" means nothing.
                    confirmed_ball = None
                    hit_streak = 0

                # only act on a ball we have seen enough times in a row
                tracked_ball = confirmed_ball if hit_streak >= BALL_CONFIRM_FRAMES else None

                if tracked_ball is not None:
                    last_ball_time = now
                    closest_ball = tracked_ball
                    
                    cv2.putText(frame, f"Closest Ball: ({closest_ball.x}, {closest_ball.y})", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
                    cv2.circle(frame, (closest_ball.x, closest_ball.y), 10, (255, 0, 0), 3)

                    # INTAKE BALL: ball passed stop line
                    if closest_ball.y >= stop_line_y:
                        intake_until_time = now + 0.8  # ponytail: 0.8s blind run-on; tune to how long the hood needs
                        set_speed(0.3, 0.3)  # ponytail: creep straight while swallowing; PWM latches, so silence != stop
                        intake_start()

                    # APPROACHING BALL: proportional steering
                    else:
                        
                        # Smooth speed reduction: slow down as ball gets closer so intake can grab it
                        proximity_ratio = min(1.0, closest_ball.y / stop_line_y)
                        
                        BASE_CRUISE_SPEED = 0.75 - (0.35 * proximity_ratio)
                        STEERING_GAIN = 0.75
                        
                        # calculate error
                        screen_center_x = frame_w // 2
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
                        set_speed(0.3, 0.3)  # ponytail: slow and STRAIGHT (equal L/R) - blind, so keep it short
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
                cv2.putText(frame, f"Decode: {inference_time_ms:.1f} ms (on-sensor)", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
                cv2.putText(frame, f"Left: {left_speed:.2f} | Right: {right_speed:.2f}", (15, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                
                trk = "LOCKED" if tracked_ball is not None else f"cand {hit_streak}/{BALL_CONFIRM_FRAMES}"
                cv2.putText(frame, f"Track: {trk}", (15, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

                intake_text = "ON" if intake_running else "OFF"
                cv2.putText(frame, f"Intake: {intake_text}", (15, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.imshow(window_name, frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    running = False
                    break

        finally:
            print("Cleaning up...")
            cv2.destroyAllWindows()
            if picam2 is not None:
                picam2.stop()

def cleanup():
    global running
    running = False

    # ponytail: the inference thread also calls set_speed. Let it exit before
    # we neutralise, or it re-commands the motors after we have stopped them.
    if inference_thread is not None:
        inference_thread.join(timeout=2.0)

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
    global inference_thread

    left_motor_1.angle(MOTOR_NEUTRAL)
    left_motor_2.angle(MOTOR_NEUTRAL)
    right_motor_1.angle(MOTOR_NEUTRAL)
    right_motor_2.angle(MOTOR_NEUTRAL)
    intake_stop()

    print("=============================================")
    print("         AUTONOMOUS BALL INTAKING            ")
    print("    Team Northeast  auto_chassis.py  v2.4    ")
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

