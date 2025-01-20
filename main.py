import numpy as np
import argparse
import cv2
from ultralytics import YOLO
from collections import defaultdict

parser = argparse.ArgumentParser(description="Object Tracking with YOLO")
parser.add_argument("-v", "--video", required=True, help="Path to the video file")
args = parser.parse_args()

# Load the YOLO11 model
model = YOLO("yolo11s.pt")

video_path = args.video
if video_path == "0":
    video_path = 0

cap = cv2.VideoCapture(video_path)

track_history = defaultdict(lambda: [])
directions = {}

while cap.isOpened():
    # Read a single frame from the video
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True)

        # Get the boxes and track IDs
        if results[0].boxes is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id
            if track_ids is not None:
                track_ids = track_ids.int().cpu().tolist()
            else:
                track_ids = []
        else:
            boxes = []
            track_ids = []

        annotated_frame = results[0].plot()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            center = (float(x), float(y))

            # Update track history
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:
                track.pop(0)

            # Calculate movement direction
            if len(track) > 1:
                prev_x, prev_y = track[-2]
                dx = center[0] - prev_x
                dy = center[1] - prev_y

                # Determine direction based on dx and dy
                if abs(dx) > abs(dy):  # Horizontal movement
                    if dx > 0:
                        directions[track_id] = "Right"
                    else:
                        directions[track_id] = "Left"
                else:  # Vertical movement
                    if dy > 0:
                        directions[track_id] = "Down"
                    else:
                        directions[track_id] = "Up"
            else:
                directions[track_id] = "Stationary"

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(240, 240, 240), thickness=10)

            # Add direction label
            direction_label = directions.get(track_id, "Unknown")
            cv2.putText(
                annotated_frame,
                direction_label,
                (int(x - w / 2), int(y - h / 2) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (250, 250, 250),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("YOLO11 Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

cap.release()
cv2.destroyAllWindows()
