import os

import numpy as np
from PIL import Image


# def calculate_image_stats(folder_path):
#     image_paths = []
#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             if file.endswith(('.png', '.jpg', '.jpeg')):
#                 image_paths.append(os.path.join(root, file))
#
#     mean_list = []
#     variance_list = []
#     for path in image_paths:
#         image = Image.open(path).convert('RGB')
#         image_array = np.array(image)
#         mean_list.append(np.mean(image_array))
#         variance_list.append(np.var(image_array))
#
#     total_mean = np.mean(mean_list)
#     total_variance = np.mean(variance_list)
#
#     # 归一化到0和1之间
#     max_value = max(total_mean + total_variance, 1.0)
#     total_mean_normalized = total_mean / max_value
#     total_variance_normalized = total_variance / max_value
#
#     return total_mean_normalized, total_variance_normalized
#
#
# # 指定文件夹路径
# folder_path = '../datasets_all/datasets_changahi/diease_folder'
#
# # 计算图像文件的总均值和总方差，并归一化
# total_mean, total_variance = calculate_image_stats(folder_path)
#
# print(f"Total Mean: {total_mean}")
# print(f"Total Variance: {total_variance}")

"""

rgb每个通道
"""
import os
from PIL import Image
import numpy as np


def calculate_channel_stats(folder_path):
    image_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    red_mean_list = []
    green_mean_list = []
    blue_mean_list = []
    red_variance_list = []
    green_variance_list = []
    blue_variance_list = []

    for path in image_paths:
        image = Image.open(path).convert('RGB')
        image_array = np.array(image)

        red_channel = image_array[:, :, 0]
        green_channel = image_array[:, :, 1]
        blue_channel = image_array[:, :, 2]

        red_mean_list.append(np.mean(red_channel))
        green_mean_list.append(np.mean(green_channel))
        blue_mean_list.append(np.mean(blue_channel))

        red_variance_list.append(np.var(red_channel))
        green_variance_list.append(np.var(green_channel))
        blue_variance_list.append(np.var(blue_channel))

    red_mean = np.mean(red_mean_list)
    green_mean = np.mean(green_mean_list)
    blue_mean = np.mean(blue_mean_list)

    red_variance = np.mean(red_variance_list)
    green_variance = np.mean(green_variance_list)
    blue_variance = np.mean(blue_variance_list)

    # 归一化到0和1之间
    max_value = max(max(red_mean, green_mean, blue_mean) + max(red_variance, green_variance, blue_variance), 1.0)
    red_mean_normalized = round(red_mean / max_value,3)
    green_mean_normalized = round(green_mean / max_value,3)
    blue_mean_normalized = round(blue_mean / max_value,3)

    red_variance_normalized = round(red_variance / max_value, 3)
    green_variance_normalized = round(green_variance / max_value,3)
    blue_variance_normalized = round(blue_variance / max_value,3)

    return red_mean_normalized, green_mean_normalized, blue_mean_normalized, \
           red_variance_normalized, green_variance_normalized, blue_variance_normalized


# 指定文件夹路径
folder_path = '../datasets/changhaiall_207/diease_folder'

# 计算RGB通道的均值和方差，并归一化
red_mean, green_mean, blue_mean, red_variance, green_variance, blue_variance = calculate_channel_stats(folder_path)

print(f"Red Mean: {red_mean}")
print(f"Green Mean: {green_mean}")
print(f"Blue Mean: {blue_mean}")
print(f"Red Variance: {red_variance}")
print(f"Green Variance: {green_variance}")
print(f"Blue Variance: {blue_variance}")

rgb_means = [red_mean, green_mean, blue_mean]
rgb_variances = [red_variance, green_variance, blue_variance]
print(rgb_means,",",rgb_variances)
