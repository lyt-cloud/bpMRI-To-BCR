import os
import shutil


def organize_files_into_subfolders(folder_path,save_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 跳过文件夹和子文件夹
        if os.path.isdir(file_path):
            continue

        # 提取文件名（不包括扩展名）
        # file_name = os.path.splitext(filename)[0]
        file_name = filename.split('_')[0]
        # 构建子文件夹路径
        subfolder_path = os.path.join(save_path, file_name)
        # 创建子文件夹（如果不存在）
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        # 移动文件到子文件夹
        new_file_path = os.path.join(subfolder_path, filename)
        shutil.copy(file_path, new_file_path)
        print(f"copy {filename} to {subfolder_path}")

# 指定文件夹路径
folder_path = "../外部验证数据/长海_260/diease_folder_rgb/R"
save_path = "../外部验证数据/长海_260/diease_folder_rgb/R_folder"
# 调用函数进行文件整理
organize_files_into_subfolders(folder_path, save_path)
