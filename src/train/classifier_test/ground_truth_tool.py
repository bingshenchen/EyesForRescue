import os

import cv2
from dotenv import load_dotenv

load_dotenv()

def process_videos():
    for video_path in video_paths:
        video_name = os.path.basename(video_path)
        print(f"Processing video: {video_name}")
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            cv2.putText(frame, f"Frame: {frame_idx}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255,255,255), 2)
            cv2.imshow("Frame", frame)
            # Wait until a specific key (e.g., spacebar) is pressed to show the next frame
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):  # Spacebar key
                    break
                elif key == ord('q'):  # Quit if 'q' is pressed
                    cap.release()
                    cv2.destroyAllWindows()
                    break
        cap.release()

cv2.destroyAllWindows()


if __name__ == "__main__":
    video_paths = os.getenv("VIDEO_PATHS").split(',')
    formatted_video_paths = "\n".join(video_paths)

    process_videos()