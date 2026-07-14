import cv2
import numpy as np
from fusion_hat import Servo
from time import sleep

DRIVE_LEFT_CHANNEL = 3
DRIVE_RIGHT_CHANNEL = 4
drive_left = Servo(DRIVE_LEFT_CHANNEL)
drive_right = Servo(DRIVE_RIGHT_CHANNEL)

try:
    from picamera2 import Picamera2
    USE_PICAMERA2 = True
except ImportError:
    USE_PICAMERA2 = False
    print("picamera2 not found, falling back to cv2.VideoCapture")

# --- 1. CALIBRATION CONSTANTS ---
KNOWN_HEIGHT = 36.8             # Physical height of Home Depot bucket (cm)
CALIBRATED_PIXEL_HEIGHT = 300.0 # Replace with your measured pixel height at 100cm
CALIBRATED_DISTANCE = 100.0     # Distance used during calibration (cm)

# Calculate focal length constant (F) based on height
FOCAL_LENGTH = (CALIBRATED_PIXEL_HEIGHT * CALIBRATED_DISTANCE) / KNOWN_HEIGHT

# --- 2. OFF-CENTER MOUNTING VARIABLES (TUNE THESE!) ---
# If your camera is mounted to the RIGHT of the car's center, increase this positive number.
# If mounted to the LEFT, make this number negative.
CAMERA_OFFSET_X = 40  # Value in pixels

#Pixel width at proper distance
PIX_WIDTH = 136

# How wide the "center target zone" is. Larger = less steering jitter.
STEERING_DEADZONE = 20 

# --- 3. CAMERA SETUP ---
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 5  # Reduced framerate

if USE_PICAMERA2:
    print("Initializing picamera2...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": 'BGR888', "size": (FRAME_WIDTH, FRAME_HEIGHT)},
        controls={"FrameRate": TARGET_FPS}
    )
    picam2.configure(config)
    picam2.start()
    print(f"Camera started successfully at {TARGET_FPS} FPS")
else:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

# Rolling average buffer for distance smoothing
distance_history = []

try:
    while True:
        if USE_PICAMERA2:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
            # ^^ commented out on computer, only needed on Raspberry Pi with picamera2
        else:
            ret, frame = cap.read()
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if not ret:
                break
        # Calculate where the CAR's physical centerline is in the video frame
        car_center_x = (FRAME_WIDTH // 2) + CAMERA_OFFSET_X

        # Target zone boundaries for steering
        zone_left = car_center_x - STEERING_DEADZONE
        zone_right = car_center_x + STEERING_DEADZONE

        # Color processing (HSV Orange tracking)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_orange = np.array([5, 150, 100])
        upper_orange = np.array([22, 255, 255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        
        # Clean up image noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest_contour) > 500:
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # 1. Calculate the center point of the bucket in the frame
                bucket_center_x = x + (w // 2)
                bucket_center_y = y + (h // 2)

                # 2. Distance Calculation (Height-based)
                raw_distance = (KNOWN_HEIGHT * FOCAL_LENGTH) / float(h)
                flag1 = False
                flag2 = False
                # Smooth the distance reading to filter out chassis bumps
                distance_history.append(raw_distance)
                if len(distance_history) > 5:
                    distance_history.pop(0)
                smoothed_distance = sum(distance_history) / len(distance_history)

                # 3. Steering Logic Decision
                if bucket_center_x < zone_left:
                    steering_instruction = "STEER LEFT"
                    color = (0, 0, 255) # Red bounding box (Needs correction)
                elif bucket_center_x > zone_right:
                    steering_instruction = "STEER RIGHT"
                    color = (0, 0, 255)
                else:
                    steering_instruction = "TARGET ALIGNED"
                    color = (255, 255, 0) # Purple bounding box (On target)
                    flag1 = True


                if w > PIX_WIDTH + 10: # add buffer, watch for jitter in testing
                    distance_status = "BUCKET TOO CLOSE"
                    color = (0, 0, 255) # Red bounding box (Too close)
                elif w < PIX_WIDTH - 10: # add buffer, watch for jitter in testing
                    distance_status = "BUCKET TOO FAR"
                    color = (0, 0, 255) # Red bounding box (Too far)
                else:
                    distance_status = "BUCKET AT IDEAL DISTANCE"
                    color = (0, 255, 255) # Yellow bounding box (Ideal distance)
                    flag2 = True

                if flag1 and flag2:
                    color = (0, 255, 0) # Green bounding box (Aligned and at ideal distance)

                # Draw Bucket data
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.circle(frame, (bucket_center_x, bucket_center_y), 5, color, -1)
                
                # Display metrics
                cv2.putText(frame, f"Dist: {smoothed_distance:.1f} cm", (x, y - 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, steering_instruction, (x, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, distance_status, (x, y - 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if steering_instruction == "STEER LEFT":
                    drive_left.angle(-30)  # Adjust as needed
                    drive_right.angle(30)  # Adjust as needed 
                elif steering_instruction == "STEER RIGHT":
                    drive_left.angle(30)  # Adjust as needed
                    drive_right.angle(-30)  # Adjust as needed
                else:
                    drive_left.angle(0)  # Stop or go straight
                    drive_right.angle(0)  # Stop or go straight

                if distance_status == "BUCKET TOO CLOSE":
                    drive_left.angle(-30)  # Adjust as needed
                    drive_right.angle(-30)  # Adjust as needed
                elif distance_status == "BUCKET TOO FAR":
                    drive_left.angle(30)  # Adjust as needed
                    drive_right.angle(30)  # Adjust as needed
                else:
                    drive_left.angle(0)  # Stop or go straight
                    drive_right.angle(0)  # Stop or go straight

        # --- VISUALIZE OFF-CENTER CONFIGURATION ---
        # Draw the true center of the video frame (Yellow dotted line)
        cv2.line(frame, (FRAME_WIDTH // 2, 0), (FRAME_WIDTH // 2, FRAME_HEIGHT), (0, 255, 255), 1, cv2.LINE_AA)
        
        # Draw the Car's offset center line (Blue solid line)
        cv2.line(frame, (car_center_x, 0), (car_center_x, FRAME_HEIGHT), (255, 0, 0), 2)
        
        # Draw the Deadzone boundaries (Light Blue lines)
        cv2.line(frame, (zone_left, 0), (zone_left, FRAME_HEIGHT), (255, 255, 0), 1)
        cv2.line(frame, (zone_right, 0), (zone_right, FRAME_HEIGHT), (255, 255, 0), 1)

        cv2.imshow("RC Car Navigation View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program interrupted by user")
except Exception as e:
    print(f"Error: {e}")
finally:
    if USE_PICAMERA2:
        picam2.stop()
    else:
        cap.release()
    cv2.destroyAllWindows()
    print("Camera closed")