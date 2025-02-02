import os


def replace_first_digit_in_txt(directory_path):
    """
    Read all .txt files in the specified directory, replace the first digit in each line (0 to 1, or 1 to 0).

    Args:
        directory_path (str): Path to the directory containing .txt files.
    """
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                new_lines = []
                for line in lines:
                    parts = line.strip().split(' ')
                    if parts[0] == '0':
                        parts[0] = '3'
                    elif parts[0] == '1':
                        parts[0] = '0'
                    new_lines.append(' '.join(parts))

                with open(file_path, 'w') as f:
                    f.write('\n'.join(new_lines))

                print(f"Processed file: {file_path}")


def search_classier_in_txt(directory_path):
    """
    Search for all classier numbers in .txt files and print them in the specified format.

    Args:
        directory_path (str): Path to the directory containing .txt files.
    """
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                classiers = []
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split(' ')
                        if parts:
                            classiers.append(parts[0])

                if classiers:
                    print(f"{classiers}: {file}")


# Example usage:
directory = r"C:\Users\Bingshen\Desktop\Ucll verkorte toegepast informatica\Fase1\Semester 1\AI Applications\AI-Applications\assets\datasets\fall_detection\frames\Subject.10\Sit down"
# replace_first_digit_in_txt(directory)
search_classier_in_txt(directory)
