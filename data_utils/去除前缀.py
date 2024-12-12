import os

def remove_prefix_from_files(folder_path, prefix):
    for filename in os.listdir(folder_path):
        if filename.startswith(prefix):
            new_filename = filename[len(prefix):]
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_filename)
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {filename} -> {new_filename}")

# 调用函数，并指定文件夹路径和要去除的前缀
folder_path = '../new_vis/0_roi'
prefix_to_remove = '1_result_'
remove_prefix_from_files(folder_path, prefix_to_remove)
