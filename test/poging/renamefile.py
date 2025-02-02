import os

# 指定路径，使用 UNC 路径格式
folder_path = r"\\?\C:\Users\Bingshen\Desktop\Ucll verkorte toegepast informatica\Fase1\Semester 1\AI Applications\AI-Applications\assets\datasets\people_static_fine_and_needhelp\train\needhelp"

# 检查文件夹是否存在
if os.path.exists(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)

    # 遍历文件
    for idx, file in enumerate(files, start=1):
        # 确保只重命名图片文件（例如：jpg, png）
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            # 获取文件扩展名
            file_extension = file.split('.')[-1]

            # 创建新的文件名
            new_filename = f"needhelp_{idx}.{file_extension}"

            # 获取旧文件和新文件的完整路径
            old_file_path = os.path.join(folder_path, file)
            new_file_path = os.path.join(folder_path, new_filename)

            # 检查目标文件是否已存在，如果存在，则跳过
            if not os.path.exists(new_file_path):
                # 重命名文件
                os.rename(old_file_path, new_file_path)

                # 输出重命名的文件信息
                print(f"Renamed: {file} -> {new_filename}")
            else:
                print(f"Skipped: {new_filename} already exists.")
else:
    print(f"The folder at {folder_path} does not exist.")
