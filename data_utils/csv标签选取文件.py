import csv
import os
import shutil

# 读取CSV文件
csv_file_path = '../datasets_figure/score_cam/T2_cam.csv'  # 请替换成你的CSV文件路径
output_folder = '../datasets_figure/score_cam/T2_cam01'  # 输出文件夹

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 读取CSV文件并分类文件夹
with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # 跳过标题行

    for row in reader:
        folder_name, label = row
        source_folder = os.path.join('../datasets_figure/score_cam/T2_cam', folder_name)  # 请替换成你的文件夹路径

        # 创建输出文件夹，如果不存在的话
        output_subfolder = os.path.join(output_folder, label)
        os.makedirs(output_subfolder, exist_ok=True)

        # 移动文件夹到相应的输出文件夹
        destination_folder = os.path.join(output_subfolder, folder_name)
        shutil.copytree(source_folder, destination_folder)

print("分类完成")
