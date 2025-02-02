import os

def rename_images_and_labels(directory, prefix="AI4ID_valid_"):
    """Rename image and label files in the specified directory with a given prefix and sequential numbering."""
    image_extensions = ['.jpg', '.png']
    label_extension = '.txt'
    counter = 1

    # Sort files to ensure matching between images and labels
    files = sorted(os.listdir(directory))

    for file in files:
        file_path = os.path.join(directory, file)
        file_name, file_ext = os.path.splitext(file)

        # Check if file is an image or a label
        if file_ext.lower() in image_extensions:
            new_name = f"{prefix}{counter}{file_ext}"
            new_label_name = f"{prefix}{counter}{label_extension}"

            # Rename the image if the target name does not already exist
            if os.path.exists(file_path):
                new_image_path = os.path.join(directory, new_name)
                if not os.path.exists(new_image_path):
                    os.rename(file_path, new_image_path)
                    print(f"Renamed {file} to {new_name}")
                else:
                    print(f"Skipped renaming {file} as {new_name} already exists")
            else:
                print(f"File not found: {file_path}")

            # Check if a corresponding label exists and rename it
            label_path = os.path.join(directory, f"{file_name}{label_extension}")
            new_label_path = os.path.join(directory, new_label_name)
            if os.path.exists(label_path):
                if not os.path.exists(new_label_path):
                    os.rename(label_path, new_label_path)
                    print(f"Renamed {file_name}{label_extension} to {new_label_name}")
                else:
                    print(f"Skipped renaming {file_name}{label_extension} as {new_label_name} already exists")

            counter += 1

# Usage
directory_path = r"C:\Users\Bingshen\Desktop\Ucll verkorte toegepast informatica\Fase1\Semester 1\AI Applications\AI-Applications\assets\datasets\fall_detection\frames\AI4ID\valid"
rename_images_and_labels(directory_path)
