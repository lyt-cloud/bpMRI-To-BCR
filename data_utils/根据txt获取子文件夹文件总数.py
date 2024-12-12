import os

# 定义根文件夹路径
root_folder = "../datasets_all/datasets_suzhou_n4/diease_folder"

# 读取文本文件内容
file_path = "../datasets_all/datasets_suzhou_n4/bcr_label_sz.txt"  # 请替换成你的文本文件路径
with open(file_path, "r") as file:
    lines = file.readlines()

# 初始化图片总数
total_images = 0

# 遍历每一行，获取子文件夹路径，并统计图片数量
for line in lines:
    folder_path, label = line.strip().split(",")
    folder_path = os.path.join(root_folder, folder_path)

    # 统计子文件夹中图片数量
    if os.path.exists(folder_path):
        image_count = sum([len(files) for _, _, files in os.walk(folder_path)])
        total_images += image_count
        print(f"子文件夹 {folder_path} 中的label：{label}，图片数量：{image_count} ")

# 打印总图片数量
print(f"所有子文件夹中的图片总数：{total_images}")
