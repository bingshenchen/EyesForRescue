import os
import cv2
import pandas as pd
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()


def evaluate_model_on_video(model, video_path, output_video_path, results_csv_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, int(cap.get(cv2.CAP_PROP_FPS)),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    frame_cnt = 0
    results_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_cnt += 1
        results = model.predict(frame, verbose=False)[0]  # Run detection on the current frame

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            # Draw bounding boxes on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{model.names[class_id]}: {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Record results for evaluation metrics
            results_list.append({
                'frame': frame_cnt,
                'class_id': class_id,
                'confidence': confidence,
                'coordinates': (x1, y1, x2, y2)
            })

        out.write(frame)

        # Optionally, display the frame
        # cv2.imshow('Frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Save results to CSV
    df_results = pd.DataFrame(results_list)
    df_results.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")


if __name__ == "__main__":
    model_path = os.getenv('YOLO_MODEL_PATH')
    video_path = os.getenv('TEST_VIDEO_PATH')
    output_video_path = "output_evaluation_video.avi"
    results_csv_path = "video_evaluation_results.csv"

    model = YOLO(model_path)
    evaluate_model_on_video(model, video_path, output_video_path, results_csv_path)
