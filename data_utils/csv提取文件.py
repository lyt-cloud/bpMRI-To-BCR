import csv
import os
import shutil

# 定义输入csv文件路径和两个输出文件夹路径
csv_file_path = '../bcr_datasets/2D_datasets/classification_roi/picai_illness_3c.csv'
output_folder_0 = '../bcr_datasets/2D_datasets/classification_roi/0'
output_folder_1 = '../bcr_datasets/2D_datasets/classification_roi/diease'

# 创建输出文件夹，如果它们不存在
os.makedirs(output_folder_0, exist_ok=True)
os.makedirs(output_folder_1, exist_ok=True)

# 读取csv文件并根据标签复制图片到不同的输出文件夹中
with open(csv_file_path, 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        img_path, label = row
        # print(row)
        label = int(label)

        # 将图片文件复制到相应的输出文件夹中
        if label == 0:
            shutil.copy(img_path, os.path.join(output_folder_0, os.path.basename(img_path)))
        elif label == 1:
            shutil.copy(img_path, os.path.join(output_folder_1, os.path.basename(img_path)))
        else:
            print(f"Invalid label ({label}) for image: {img_path}")
