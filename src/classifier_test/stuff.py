import os
import cv2
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv

RED = (0, 0, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)

PURPLE = (255, 0, 255)

tracks_history = {}
next_track_id = 0

load_dotenv()

model_base = os.getenv('MODEL_BASE')
model_file = os.getenv('MODEL_FILE')
if not model_base or not model_file:
    raise EnvironmentError("MODEL_BASE and MODEL_FILE environment variables must be set.")
model_path = os.path.join(model_base, model_file)
model = YOLO(model_path)

alert_frames = []


def get_yolo_pose_detections(frame):
    results = model(frame, verbose=False)
    detections = []

    # Iterate over results and extract pose keypoints for persons only
    for r in results:
        if hasattr(r, 'keypoints') and r.keypoints is not None:
            keypoints = r.keypoints.data.cpu().numpy()  # Access keypoints data as numpy array
            cls_list = r.boxes.cls if hasattr(r.boxes, 'cls') else []

            for keypoint_data, cls in zip(keypoints, cls_list):
                cls = int(cls.item())

                # this makes sure the detection is a person
                if cls == 0:
                    detections.append(keypoint_data)

    return detections


def estimate_head_position(left_shoulder, right_shoulder, left_hip, right_hip):
    # Calculate the center of the shoulders and hips
    shoulders_center = (left_shoulder[:2] + right_shoulder[:2]) / 2
    hips_center = (left_hip[:2] + right_hip[:2]) / 2

    # Vector from hips to shoulders
    vector_hip_to_shoulders = shoulders_center - hips_center

    # Estimate head position by projecting upward from shoulders
    estimated_head = shoulders_center + vector_hip_to_shoulders

    estimated_head[1] -= 0.5 * vector_hip_to_shoulders[1]  # Adjust the head position slightly

    return estimated_head


def process_videos(video_paths, frame_skip=23, fall_duration_threshold=5):
    global tracks_history
    global alert_frames
    required_frames = fall_duration_threshold // (frame_skip / 23)
    print(f"Required frames to trigger alert: {required_frames} frames")

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            continue

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            keypoints_list = get_yolo_pose_detections(frame)

            for keypoints in keypoints_list:
                track_id = assign_track_id_pose(keypoints)

                # Update tracking history and determine if fall is detected
                tracks_history, color = update_tracks_history_pose(
                    track_id, keypoints, required_frames=int(required_frames), video_name=video_path
                )

                # Draw keypoints on the frame
                keypoints_reshaped = keypoints.reshape(-1, 3)  # Reshape to (num_keypoints, 3)
                head = keypoints_reshaped[0]
                left_shoulder = keypoints_reshaped[5]
                right_shoulder = keypoints_reshaped[6]
                left_hip = keypoints_reshaped[11]
                right_hip = keypoints_reshaped[12]
                for kp in keypoints_reshaped:
                    x, y, confidence = kp
                    if confidence > 0.5:  # Draw only if confidence is sufficient
                        cv2.circle(frame, (int(x), int(y)), radius=3, color=color, thickness=-1)

                # If the head is missing or confidence is low, estimate its position
                if head[2] <= 0.5 and all(x is not None for x in [left_shoulder, right_shoulder, left_hip, right_hip]):  # Confidence is low
                    estimated_head = estimate_head_position(left_shoulder, right_shoulder, left_hip, right_hip)
                    # Draw the estimated head position in purple
                    cv2.circle(frame, (int(estimated_head[0]), int(estimated_head[1])), radius=3, color=PURPLE,
                               thickness=-1)

                if color == RED:
                    alert_frames.append(frame)

                # Logging for debugging purposes
                # print(
                #     f"Track ID {track_id} | Fallen frame count: {tracks_history[track_id]['fall_frame_count']} / {required_frames}"
                # )

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1

        cap.release()

    cv2.destroyAllWindows()


def assign_track_id_pose(keypoints, distance_threshold=100):
    global next_track_id
    keypoint_center = np.mean(keypoints[:, :2], axis=0)  # Calculate the mean position of all keypoints

    # Try to match with existing tracks based on proximity
    for track_id, data in tracks_history.items():
        last_position = data["last_position"]
        dist = np.linalg.norm(keypoint_center - np.array(last_position))

        # Use distance matching to reuse an existing track ID
        if dist < distance_threshold:
            tracks_history[track_id]["last_position"] = keypoint_center
            return track_id

    # Create a new track ID if no match is found
    next_track_id += 1
    tracks_history[next_track_id] = {"last_position": keypoint_center, "static_frame_count": 0}
    return next_track_id


# Function to calculate the angle between two points relative to the horizontal plane
def calculate_angle(point1, point2):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    return np.degrees(np.arctan2(dy, dx))


def detect_fall(keypoints, confidence_threshold=0.3, height_ratio_threshold=0.4, symmetry_threshold=10,
                temporal_threshold=5):
    keypoints_reshaped = keypoints.reshape(-1, 3)

    head = keypoints_reshaped[0] if keypoints_reshaped[0, 2] > confidence_threshold else None
    left_shoulder = keypoints_reshaped[5] if keypoints_reshaped[5, 2] > confidence_threshold else None
    right_shoulder = keypoints_reshaped[6] if keypoints_reshaped[6, 2] > confidence_threshold else None
    left_hip = keypoints_reshaped[11] if keypoints_reshaped[11, 2] > confidence_threshold else None
    right_hip = keypoints_reshaped[12] if keypoints_reshaped[12, 2] > confidence_threshold else None
    left_foot = keypoints_reshaped[15] if keypoints_reshaped[15, 2] > confidence_threshold else None
    right_foot = keypoints_reshaped[16] if keypoints_reshaped[16, 2] > confidence_threshold else None

    if head is None and all(x is not None for x in [left_shoulder, right_shoulder, left_hip, right_hip]):
        head = estimate_head_position(left_shoulder, right_shoulder, left_hip, right_hip)

    if any(kp is None for kp in [head, left_shoulder, right_shoulder, left_hip, right_hip, left_foot, right_foot]):
        return False  # Insufficient data

    body_height = np.linalg.norm(head[:2] - ((left_foot[:2] + right_foot[:2]) / 2))
    if body_height == 0:  # Avoid division by zero
        return False

    hips_y = (left_hip[1] + right_hip[1]) / 2
    head_to_hips = abs(head[1] - hips_y)
    head_to_hips_ratio = head_to_hips / body_height

    shoulders_center = (left_shoulder[0] + right_shoulder[0]) / 2
    hips_center = (left_hip[0] + right_hip[0]) / 2
    symmetry_deviation = abs(shoulders_center - hips_center)

    hips_ratio_trigger = head_to_hips_ratio < height_ratio_threshold

    symmetry_trigger = symmetry_deviation > symmetry_threshold

    is_fallen = (
            hips_ratio_trigger and
            symmetry_trigger
    )

    return is_fallen


def update_tracks_history_pose(track_id, keypoints, required_frames, video_name):
    global tracks_history
    track = tracks_history.get(track_id, {"fall_frame_count": 0, "alert_triggered": False, "static_frame_count": 0})

    # Calculate the vertical displacement of keypoints (e.g., head and shoulders vs. hips and feet)
    keypoints_reshaped = keypoints.reshape(-1, 3)  # Reshape to (num_keypoints, 3)

    is_fallen = detect_fall(keypoints)

    # Track movement to determine if the person is stationary
    keypoint_center = np.mean(keypoints_reshaped[:, :2], axis=0)
    last_position = tracks_history[track_id]["last_position"]
    movement_distance = np.linalg.norm(keypoint_center - last_position)

    if movement_distance < 20:  # Threshold to determine if the person is stationary
        track["static_frame_count"] += 1
    else:
        track["static_frame_count"] = 0

    tracks_history[track_id]["last_position"] = keypoint_center

    # Check if the person has fallen and is stationary for the required duration
    if is_fallen and track["static_frame_count"] >= required_frames:
        track["fall_frame_count"] += 1
        if track["fall_frame_count"] >= required_frames and not track["alert_triggered"]:
            track["alert_triggered"] = True
            print(f"ALERT: Person {track_id} on video {video_name} needs help!")
    else:
        track["fall_frame_count"] = 0
        track["alert_triggered"] = False

    tracks_history[track_id] = track

    color = RED if track["alert_triggered"] else (YELLOW if is_fallen else GREEN)
    return tracks_history, color


if __name__ == "__main__":
    video_paths = os.getenv("VIDEO_PATHS").split(',')
    formatted_video_paths = "\n".join(video_paths)
    print(f"Running fall detection on videos:\n{formatted_video_paths}")
    process_videos(video_paths, frame_skip=5, fall_duration_threshold=1)
    # for frame in alert_frames:
    #     cv2.imshow("Alert Frame", frame)
    #     cv2.waitKey(0)
