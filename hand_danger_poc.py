import cv2
import numpy as np
import time
import math

# ---------- CONFIGURATION ----------

# Choose camera index (0 for default webcam)
CAMERA_INDEX = 0

# Virtual object (a circle in the middle of the screen)
VIRTUAL_RADIUS = 70  # radius of virtual object

# Distance thresholds (in pixels) for state logic
# distance > D_SAFE      -> SAFE
# D_DANGER < distance <= D_SAFE  -> WARNING
# distance <= D_DANGER   -> DANGER
D_SAFE = 150
D_DANGER = 80

# HSV range for color segmentation (example: green band / paper)
# You can change this to track another color if needed.
LOWER_COLOR = np.array([40, 40, 40])   # lower HSV bound for green-ish
UPPER_COLOR = np.array([80, 255, 255]) # upper HSV bound for green-ish

# Minimum contour area to consider as "hand / marker"
MIN_CONTOUR_AREA = 1500

# -----------------------------------

def classify_state(distance):
    if distance is None:
        # No hand detected -> treat as SAFE or separate "NO HAND"
        return "SAFE"
    if distance > D_SAFE:
        return "SAFE"
    elif distance > D_DANGER:
        return "WARNING"
    else:
        return "DANGER"

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Resize for speed & consistency
        frame = cv2.resize(frame, (640, 480))

        # Get frame dimensions and center (virtual object center)
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2

        # Convert to HSV for simpler color segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for the chosen color (e.g., green band on hand)
        mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)

        # Clean mask - remove noise
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((5, 5), np.uint8), iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE,
                                np.ones((5, 5), np.uint8), iterations=1)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        hand_center = None
        distance = None

        if contours:
            # Take the largest contour as the "hand / marker"
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area > MIN_CONTOUR_AREA:
                # Draw contour
                cv2.drawContours(frame, [largest], -1, (255, 0, 0), 2)

                # Compute center of contour using moments
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    hand_center = (cx, cy)

                    # Draw center point
                    cv2.circle(frame, hand_center, 7, (255, 255, 255), -1)
                    cv2.putText(frame, "HAND", (cx + 10, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # Distance from hand center to virtual object center
                    distance = math.dist((cx, cy), (center_x, center_y))

        # Classify interaction state
        state = classify_state(distance)

        # Draw virtual object (circle) and change its color by state
        if state == "SAFE":
            color = (0, 255, 0)      # green
        elif state == "WARNING":
            color = (0, 255, 255)    # yellow
        else:
            color = (0, 0, 255)      # red

        cv2.circle(frame, (center_x, center_y),
                   VIRTUAL_RADIUS, color, 3)
        cv2.putText(frame, "VIRTUAL OBJECT",
                    (center_x - 90, center_y - VIRTUAL_RADIUS - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw distance info if we have a hand detected
        if distance is not None:
            cv2.putText(frame, f"Distance: {int(distance)}",
                        (10, 430), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)

        # Show current state at top
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
        state_color = (0, 255, 0) if state == "SAFE" else \
                      (0, 255, 255) if state == "WARNING" else (0, 0, 255)
        cv2.putText(frame, f"STATE: {state}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, state_color, 2)

        # Show DANGER DANGER overlay
        if state == "DANGER":
            cv2.putText(frame, "DANGER  DANGER",
                        (int(w * 0.18), int(h * 0.55)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

        # FPS calculation
        current_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (current_time - prev_time))
        prev_time = current_time

        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

        cv2.imshow("Hand Danger POC - Arvyax", frame)
        cv2.imshow("Mask (debug)", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            # ESC or q to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
