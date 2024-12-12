import os
import shutil

# 定义两个文件夹的路径
folder1_path = "../new_vis/1_roi"
folder2_path = "../new_vis/T2WI_roi"

# 指定新文件夹的路径
new_folder_path = "../new_vis/roi_folder"

# 获取folder1中的文件名列表
folder1_files = os.listdir(folder1_path)

# 获取folder2中的文件名列表
folder2_files = os.listdir(folder2_path)

# 创建一个字典来存储文件名的重复部分及其对应的文件列表
file_dict = {}

# 遍历folder1中的文件
for file_name in folder1_files:
    # 获取文件名的重复部分
    common_part = os.path.splitext(file_name)[0]
    # 将文件名和对应的文件路径添加到字典中
    if common_part in file_dict:
        file_dict[common_part].append(os.path.join(folder1_path, file_name))
    else:
        file_dict[common_part] = [os.path.join(folder1_path, file_name)]

# 遍历folder2中的文件
for file_name in folder2_files:
    # 获取文件名的重复部分
    common_part = os.path.splitext(file_name)[0]
    # 将文件名和对应的文件路径添加到字典中
    if common_part in file_dict:
        file_dict[common_part].append(os.path.join(folder2_path, file_name))
    else:
        file_dict[common_part] = [os.path.join(folder2_path, file_name)]

# 遍历字典中的每一项
for common_part, files_list in file_dict.items():
    # 创建新文件夹的路径
    new_subfolder_path = os.path.join(new_folder_path, common_part)
    # 创建新文件夹
    os.makedirs(new_subfolder_path, exist_ok=True)
    # 复制文件到新文件夹中
    for file_path in files_list:
        shutil.copy(file_path, new_subfolder_path)
