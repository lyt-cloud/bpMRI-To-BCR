import os
import shutil

# 源文件夹路径
source_folder = '../bcr_datasets/2D_datasets/nnUNet_raw_data_roi/Task_changhai_pca/labelsTr'

# 目标文件夹路径
destination_folder = '../bcr_datasets/2D_datasets/nnUNet_raw_data_roi/Task_changhai_pca/imagesTr'

# 获取源文件夹中的所有文件名
source_files = os.listdir(source_folder)

# 遍历源文件夹中的文件
for source_file in source_files:
    if source_file.endswith('.nii.gz'):
        # 提取前缀
        prefix = source_file.split('.')[0]

        # 构建匹配前缀的目标文件名列表
        matching_files = [f for f in os.listdir(destination_folder) if f.startswith(prefix)]

        # 复制匹配的文件到目标文件夹
        for matching_file in matching_files:
            source_file_path = os.path.join(source_folder, source_file)
            destination_file_path = os.path.join(destination_folder, matching_file)
            shutil.copy(source_file_path, destination_file_path)

print("文件复制完成！")

