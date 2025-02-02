import os

def delete_avi_files(directory):
    """Delete all .avi files in the subdirectories of the specified directory."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.avi'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

# Usage
directory_path = r"C:\Users\Bingshen\Desktop\Ucll verkorte toegepast informatica\Fase1\Semester 1\AI Applications\AI-Applications\assets\datasets\fall_detection\frames"
delete_avi_files(directory_path)
