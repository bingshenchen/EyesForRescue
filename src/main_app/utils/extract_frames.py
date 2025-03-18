import os
import cv2
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def save_video_frames(video_file_path, output_directory_path, frame_interval=5):
    """
    Extract frames from a video at a specific frame interval and save them as PNG images.

    Args:
        video_file_path (str): Path to the video file.
        output_directory_path (str): Directory where frames will be saved.
        frame_interval (int): The interval between frames to be saved (default: 5).
    """
    # Ensure output directory exists
    if not os.path.exists(output_directory_path):
        os.makedirs(output_directory_path)

    # Open the video file
    cap = cv2.VideoCapture(video_file_path)

    if not cap.isOpened():
        logging.error(f"Error: Unable to open video file {video_file_path}")
        return

    logging.info(f"Video {video_file_path} opened successfully.")

    frame_count = 0
    video_name = os.path.splitext(os.path.basename(video_file_path))[0]  # Extract the video name without extension

    # Loop through the video
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            logging.info("Reached the end of the video or cannot read the frame.")
            break

        # Save the frame at the specified interval
        if frame_count % frame_interval == 0:
            frame_name = os.path.join(output_directory_path, f"{video_name}_{frame_count}.png")  # Save as PNG format
            cv2.imwrite(frame_name, frame)
            logging.info(f"Saved {frame_name}")

        frame_count += 1

    # Release video capture
    cap.release()
    cv2.destroyAllWindows()


def get_video_files(directory, valid_extensions=(".mp4", ".avi")):
    """
    Retrieve all video files in the specified directory and its subdirectories.

    Args:
        directory (str): The directory to search for video files.
        valid_extensions (tuple): The video file extensions to look for (default: (".mp4", ".avi")).

    Returns:
        List of video file paths.
    """
    video_files_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(valid_extensions):
                video_files_list.append(os.path.join(root, file))
    return video_files_list


def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Retrieve paths from environment variables
    video_directory = os.getenv('VIDEO_DIR')  # Directory where video files are located
    output_directory_root = os.getenv('TRAINING_IMAGES_PATH')  # Root directory for training images

    if not video_directory or not output_directory_root:
        logging.error("VIDEO_DIR or TRAINING_IMAGES_PATH environment variables are not set.")
        return

    # Retrieve all video files (e.g., .mp4, .avi) from the directory
    video_files = get_video_files(video_directory)

    # Extract frames from each video
    for video_path in video_files:
        # Define output directory for the frames
        video_name = os.path.splitext(os.path.basename(video_path))[0]  # Get the video name without extension
        output_directory = os.path.join(output_directory_root, video_name)  # Output path for frames

        # Save frames from the video with the specified frame interval
        save_video_frames(video_path, output_directory, frame_interval=5)


if __name__ == "__main__":
    main()
