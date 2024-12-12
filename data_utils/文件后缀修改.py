import os

def rename_files_with_suffix(folder_path, old_suffix, new_suffix):
    for filename in os.listdir(folder_path):
        if filename.endswith(old_suffix):
            new_filename = filename.replace(old_suffix, new_suffix)
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)
            os.rename(old_path, new_path)

# 用法示例：
folder_path = "../datasets_figure/score_cam/score_cam_norm"  # 替换成实际文件夹路径
# old_suffix = ".nii.gz"
# new_suffix = "_0002.nii.gz"
old_suffix = ".png"
new_suffix = "_cam.png"
rename_files_with_suffix(folder_path, old_suffix, new_suffix)
