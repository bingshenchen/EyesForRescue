from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model
model = YOLO('yolo11n-pose.pt')

# Initialize dictionary to store keypoint trajectories
keypoint_trajectories = {}

# Open the video file
video_path = 'fall.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)

    for result in results:
        # Process detected persons
        for idx, (box, keypoints) in enumerate(zip(result.boxes.xyxy, result.keypoints.xy)):
            # Draw keypoints and update trajectories
            for kp_idx, kp in enumerate(keypoints):
                if kp is not None and len(kp) >= 2:
                    x, y = map(int, kp[:2])

                    # Draw the keypoint
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                    # Update the trajectory
                    if kp_idx not in keypoint_trajectories:
                        keypoint_trajectories[kp_idx] = []
                    keypoint_trajectories[kp_idx].append((x, y))

                    # Draw the trajectory
                    for i in range(1, len(keypoint_trajectories[kp_idx])):
                        pt1 = keypoint_trajectories[kp_idx][i - 1]
                        pt2 = keypoint_trajectories[kp_idx][i]
                        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Pose Detection with Trajectories", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Print the trajectories of keypoints after processing
for kp_idx, trajectory in keypoint_trajectories.items():
    print(f"Keypoint {kp_idx} trajectory: {trajectory}")

cap.release()
cv2.destroyAllWindows()
