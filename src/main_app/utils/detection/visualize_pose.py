from ultralytics import YOLO
import cv2
import numpy as np

# Initialize class colors
class_colors = {
    "person": (0, 255, 0),
    "sitting_person": (255, 255, 0),
    "falling_person": (0, 0, 255),
    "lying_person": (255, 0, 255)
}


# Function to infer action based on keypoints
def infer_action(keypoints):
    """
    Infer the action (e.g., sitting, falling) based on keypoints.
    Args:
        keypoints (np.ndarray): Keypoints detected for a person.
    Returns:
        str: Predicted action.
    """
    if keypoints is None or len(keypoints) == 0:
        return "Unknown"

    # Example: Use relative positions to determine actions
    # Keypoint indices (COCO format): [11=left hip, 12=right hip, 13=left knee, 14=right knee]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    left_knee = keypoints[13]
    right_knee = keypoints[14]

    if left_hip[1] > left_knee[1] and right_hip[1] > right_knee[1]:
        return "sitting_person"  # Hips are above knees -> Sitting
    elif left_knee[1] > left_hip[1] and right_knee[1] > right_hip[1]:
        return "falling_person"  # Knees are below hips -> Falling
    elif np.abs(left_hip[1] - right_hip[1]) < 20:  # Nearly horizontal
        return "lying_person"  # Hips are almost level -> Lying
    else:
        return "person"  # Default to standing or unknown action


# Load YOLO model
model = YOLO('yolo11n-pose.pt')

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
            x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
            action = infer_action(keypoints)  # Infer action from keypoints
            color = class_colors.get(action, (255, 255, 255))  # Get color for action

            # Draw bounding box and action label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, action, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw keypoints and skeleton
            for kp in keypoints:
                if kp is not None and len(kp) >= 2:
                    x, y = map(int, kp[:2])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow("Pose Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
