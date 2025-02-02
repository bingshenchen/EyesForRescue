import os
import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk
from dotenv import load_dotenv
from src.utils.calculate_danger import calculate_danger
import tkinter as tk
from threading import Event


def process_video(video_path, danger_label, canvas, root, stop_event: Event, output_video_path=None):
    print("Starting video processing...")
    load_dotenv()

    model_path = os.getenv('YOLO_MODEL_PATH')
    if not model_path:
        print("Error: YOLO_MODEL_PATH is not set in the .env file")
        return

    local_model = YOLO(model_path)
    print("Video thread started.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return []

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_width = width // 2
    new_height = height // 2
    canvas.config(width=new_width, height=new_height)

    if output_video_path and isinstance(output_video_path, str):
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        try:
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (new_width, new_height))
        except cv2.error as e:
            print(f"Error initializing VideoWriter: {e}")
            out = None
    else:
        out = None

    results_list = []
    after_id = None

    class_colors = {
        "person": (0, 255, 0),
        "falling_person": (0, 0, 255),
        "sitting_person": (255, 255, 0),
        "lying_person": (255, 0, 255)
    }

    def read_frame():
        nonlocal results_list, after_id

        if stop_event.is_set():
            print("Stopping video processing...")
            cap.release()
            if out:
                out.release()
            if after_id:
                print(f"Cancelling after_id: {after_id}")
                root.after_cancel(after_id)
            else:
                print("No after_id to cancel.")
            cv2.destroyAllWindows()
            return  # Exit the frame reading loop

        ret, frame = cap.read()
        if not ret:
            print("No more frames or video ended.")
            cap.release()
            if out:
                out.release()
            if after_id:
                print(f"Cancelling after_id: {after_id}")
                root.after_cancel(after_id)
            cv2.destroyAllWindows()
            return

        frame = cv2.resize(frame, (new_width, new_height))
        results = local_model.predict(frame, conf=0.7, stream=True)  # Increased confidence threshold
        frame_results = []

        for result in results:
            if result.boxes:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    try:
                        class_name = ["person", "falling_person", "lying_person", "sitting_person"][class_id]
                    except IndexError:
                        class_name = "Unknown"
                        print(f"Warning: class_id {class_id} is out of range.")
                    color = class_colors.get(class_name, (255, 255, 255))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{class_name}: {confidence:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if class_name == "falling_person":
                        fall_detected = 1
                    else:
                        fall_detected = 0

                    frame_results.append(fall_detected)

                    # Debugging information
                    print(f"Detected {class_name} with confidence {confidence:.2f}")

        results_list.append(frame_results or [0])

        danger_score = calculate_danger(results_list)
        update_danger_score(danger_score, danger_label)

        show_frame_on_canvas(frame, canvas, root)
        after_id = root.after(10, read_frame)

    # Start reading frames
    after_id = root.after(10, read_frame)


def show_frame_on_canvas(frame, canvas, root):
    """Convert OpenCV frame to Tkinter-compatible format and display on canvas."""
    if canvas.winfo_exists():
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = Image.fromarray(frame_rgb)  # Convert to PIL Image
        imgtk = ImageTk.PhotoImage(image=img)
        canvas.imgtk = imgtk  # Keep a reference to avoid garbage collection
        canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        root.update()


def update_danger_score(score, danger_label):
    """Update the Danger score label with corresponding color based on the score."""
    if danger_label.winfo_exists():
        try:
            danger_label.config(text=f"Danger Score: {score}")
            if score < 3:
                danger_label.config(fg="green")
            elif 3 <= score < 6:
                danger_label.config(fg="orange")
            else:
                danger_label.config(fg="red")
        except Exception as e:
            print("Error updating the danger score:", e)
