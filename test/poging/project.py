import os


def list_files(startpath):
    ignore_dirs = {'data', 'datasets', 'venv', 'test', 'model', '.git', '.idea', '__pycache__', 'processed_data',
                   'runs'}  # 需要忽略的目录名称

    for root, dirs, files in os.walk(startpath):
        # filter
        dirs[:] = [d for d in dirs if d.lower() not in ignore_dirs]

        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)

        for f in files:
            # ignore 'data', 'datasets', 'models' file
            if not any(f.lower().startswith(prefix) for prefix in ignore_dirs):
                print(f"{subindent}{f}")


project_path = r'C:\Users\Bingshen\Desktop\Ucll verkorte toegepast informatica\Fase1\Semester 1\AI Applications\AI-Applications'
list_files(project_path)
