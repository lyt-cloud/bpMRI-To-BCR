import cv2
import numpy as np
from PIL import Image



# def t2_d_to_rgb(img1,img2):
#     gray_img1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
#     gray_img2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)
#     gray_img1 = np.expand_dims(gray_img1, axis=-1)
#     gray_img2 = np.expand_dims(gray_img2, axis=-1)
#     zero2_gray = np.zeros_like(gray_img2)
#
#     rgb_img = np.concatenate((zero2_gray, gray_img2, gray_img1), axis=-1)  # BGR
#     print(rgb_img[:, :, 0])
#     # print(rgb_img.shape)
#     png_save = './'
#     cv2.imwrite(png_save, rgb_img)
#
#
# img1_folder = './datasets_30_suzhou/diease_folder_rgb/G'



import cv2
import numpy as np
import os

import cv2
import numpy as np
import os
import glob

def combine_gray_to_rgb(folder_path, output_folder):
    # 获取文件夹下的所有子文件夹
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    for subfolder in os.listdir(subfolders[0]):
        # 获取子文件夹名称，例如 '6323889'
        folder_name = subfolder

        for img_name in os.listdir(os.path.join(subfolders[0],folder_name)):
            img1_path = os.path.join(folder_path, 'R', folder_name, img_name)
            img2_path = os.path.join(folder_path, 'G', folder_name, img_name)
            # 读取图像
            gray_img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            gray_img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
            # 将灰度图像扩展为三通道
            gray_img1 = np.expand_dims(gray_img1, axis=-1)
            gray_img2 = np.expand_dims(gray_img2, axis=-1)

            # 创建一个全零的灰度图像
            zero2_gray = np.zeros_like(gray_img2)

            # 将三个通道合并为RGB图像 (BGR格式)
            rgb_img = np.concatenate((zero2_gray, gray_img2, gray_img1), axis=-1)

            # 构建输出文件名
            output_filename = os.path.basename(img1_path)
            sub_folders = os.path.join(output_folder, folder_name)
            os.makedirs(sub_folders, exist_ok=True)
            output_path = os.path.join(sub_folders, output_filename)

            # 保存RGB图像
            cv2.imwrite(output_path, rgb_img)

# 输入文件夹和输出文件夹的路径
input_folder = '../datasets_all/datasets_changhai_n4/diease_folder_rgb'
output_folder = '../datasets_all/datasets_changhai_n4/diease_folder_rgb/T2WI_DWI' # 替换为你想要保存RGB图像的文件夹路径
os.makedirs(output_folder, exist_ok=True)

# 调用函数来处理图像
combine_gray_to_rgb(input_folder, output_folder)

