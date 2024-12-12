import os
import cv2
import numpy as np

# 输入文件夹路径
original_folder = "../最终的三个模型/exp1_noclinic_vis_use/vis_rgb_128_new/1"
roi_folder = "../datasets_figure/datasets_changhai_n4/bcr_diease_vis_rgb/B"

# 输出文件夹路径
output_folder = "../最终的三个模型/exp1_noclinic_vis_use/vis_rgb_128_new/1_roi"

# 确保输出文件夹存在
os.makedirs(output_folder,exist_ok=True)

# 遍历两个文件夹中的图像
for filename in os.listdir(original_folder):
    if filename.endswith(".png"):  # 假设图像是PNG格式
        # 读取原图像
        original_image = cv2.imread(os.path.join(original_folder, filename))

        # 读取ROI标签图像
        roi_image = cv2.imread(os.path.join(roi_folder, filename[9:]), cv2.IMREAD_GRAYSCALE)
        roi_image = cv2.resize(roi_image, (224, 224), interpolation=cv2.INTER_NEAREST)
        # 寻找ROI标签图像中的轮廓
        contours, _ = cv2.findContours(roi_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 在原图像上绘制绿色轮廓
        result_image = original_image.copy()  # 创建原图像的副本
        cv2.drawContours(result_image, contours, -1, (0, 255, 0), 1)

        # 保存结果图像
        output_filename = os.path.join(output_folder, filename)
        cv2.imwrite(output_filename, result_image)

print("处理完成，结果图像保存在", output_folder)
