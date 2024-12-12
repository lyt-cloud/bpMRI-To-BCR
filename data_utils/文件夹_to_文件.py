import os
import shutil

def move_files_to_target_folder(source_folder, target_folder):
    os.makedirs(target_folder,exist_ok=True)
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            source_path = os.path.join(root, file)
            target_path = os.path.join(target_folder, file)
            shutil.move(source_path, target_path)

if __name__ == "__main__":
    # 请将下面的路径替换为你实际的文件夹路径
    source_folder = "../datasets_figure/datasets_changhai_n4/diease_folder"
    target_folder = "../datasets_figure/datasets_changhai_n4/diease_all"

    move_files_to_target_folder(source_folder, target_folder)
