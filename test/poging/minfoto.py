import os
import shutil

def extract_and_copy_images(src_directory, dest_directory):
    """Extract images and labels from Subject directories and copy every 5th image and its label to a new directory on the desktop.
       Create separate folders for each Subject and include a copy of classes.txt in each."""
    for subject_folder in os.listdir(src_directory):
        subject_path = os.path.join(src_directory, subject_folder)
        if os.path.isdir(subject_path) and subject_folder.startswith('Subject.'):
            subject_dest_path = os.path.join(dest_directory, subject_folder)
            if not os.path.exists(subject_dest_path):
                os.makedirs(subject_dest_path)

            # Copy classes.txt to the subject destination folder
            classes_file = os.path.join(src_directory, 'classes.txt')
            if os.path.exists(classes_file):
                shutil.copy2(classes_file, subject_dest_path)
                print(f"Copied classes.txt to {subject_dest_path}")

            for action_folder in os.listdir(subject_path):
                action_path = os.path.join(subject_path, action_folder)
                if os.path.isdir(action_path):
                    # Get image files only (no labels)
                    image_files = sorted([f for f in os.listdir(action_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    for i, image_file in enumerate(image_files):
                        if (i + 1) % 9 == 0:  # Every 5th file
                            image_path = os.path.join(action_path, image_file)
                            image_name, _ = os.path.splitext(image_file)
                            label_file = image_name + '.txt'
                            label_path = os.path.join(action_path, label_file)

                            # Copy the image file
                            dest_image_path = os.path.join(subject_dest_path, image_file)
                            shutil.copy2(image_path, dest_image_path)
                            print(f"Copied {image_file} to {subject_dest_path}")

                            # Copy the label file if it exists
                            if os.path.exists(label_path):
                                dest_label_path = os.path.join(subject_dest_path, label_file)
                                shutil.copy2(label_path, dest_label_path)
                                print(f"Copied {label_file} to {subject_dest_path}")

# Define paths
src_directory = r"C:\Users\Bingshen\Desktop\Ucll verkorte toegepast informatica\Fase1\Semester 1\AI Applications\AI-Applications\assets\datasets\fall_detection\frames"
dest_directory = os.path.expanduser(r"~\Desktop\Extracted_Every5_Images")

# Run the function
extract_and_copy_images(src_directory, dest_directory)
