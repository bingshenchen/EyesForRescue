import os
import cv2
import pandas as pd
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()
model_path = os.getenv('YOLO_MODEL_PATH')
model = YOLO(model_path)

MOVEMENT_SENSITIVITY = 0.1
STATIC_THRESHOLD = 10
ALERT_DELAY_FRAMES = 30


def is_same_position(xmin, ymin, width, height, prev_data):
    prev_xmin, prev_ymin, prev_width, prev_height = prev_data
    return (abs(xmin - prev_xmin) < MOVEMENT_SENSITIVITY * width and
            abs(ymin - prev_ymin) < MOVEMENT_SENSITIVITY * height)


def update_tracks_history(tracks_history, track_id, frame_nb, xmin, ymin, width, height, area, aspect_ratio):
    """Update the tracking history."""
    same_pos = False
    fall_event = False
    call_911 = False

    if track_id not in tracks_history:
        # New track
        new_record = pd.DataFrame(
            [[frame_nb, xmin, ymin, width, height, area, aspect_ratio, same_pos, fall_event, False, False]],
            columns=['frame_nb', 'xmin', 'ymin', 'width', 'height', 'area', 'aspect_ratio', 'same_pos', 'fall_event',
                     'alert_state', 'call_911'])
        tracks_history[track_id] = new_record
    else:
        # Update existing record
        prev_data = tracks_history[track_id].iloc[-1]
        same_pos = is_same_position(xmin, ymin, width, height,
                                    (prev_data['xmin'], prev_data['ymin'], prev_data['width'], prev_data['height']))
        fall_event = (aspect_ratio < 0.5 and area > prev_data['area'] * 1.2)
        stationary_for_while = (same_pos and (frame_nb - prev_data['frame_nb']) > STATIC_THRESHOLD)
        alert_state = fall_event and stationary_for_while
        call_911 = alert_state and (frame_nb - prev_data['frame_nb']) > ALERT_DELAY_FRAMES

        new_record = pd.DataFrame(
            [[frame_nb, xmin, ymin, width, height, area, aspect_ratio, same_pos, fall_event, alert_state, call_911]],
            columns=['frame_nb', 'xmin', 'ymin', 'width', 'height', 'area', 'aspect_ratio', 'same_pos', 'fall_event',
                     'alert_state', 'call_911'])
        tracks_history[track_id] = pd.concat([tracks_history[track_id], new_record], ignore_index=True)

    return tracks_history, same_pos, fall_event, call_911


def main():
    video_path = os.getenv('TEST_VIDEO_PATH')
    cap = cv2.VideoCapture(video_path)
    tracks_history = {}

    if not cap.isOpened():
        print("Video not found")
        return

    frame_cnt = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_cnt += 1
        results = model.predict(frame, verbose=False)[0]  # Use predict instead of track

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width, height = x2 - x1, y2 - y1
            area = width * height
            aspect_ratio = width / height
            track_id = frame_cnt
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            tracks_history, same_pos, fall_event, call_911 = update_tracks_history(
                tracks_history, track_id, frame_cnt, x1, y1, width, height, area, aspect_ratio)

            color = (0, 255, 0)
            if call_911:
                color = (0, 0, 255)
            elif fall_event:
                color = (0, 165, 255)
            elif same_pos:
                color = (0, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    output_path = "tracking_results.csv"
    all_data = pd.concat(tracks_history.values(), ignore_index=True)
    all_data.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
