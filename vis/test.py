import cv2
import numpy as np

# img_path = './N160302_8.png'
# rgb_img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # 使用cv2.IMREAD_COLOR以确保加载为RGB图像
# rgb_img = cv2.resize(rgb_img, (224, 224))
# rgb_img = np.float32(rgb_img) / 255
# # 提取R通道数据
# gray_img = rgb_img[:, :, 2]  # 通道顺序为BGR，R通道在索引2处
#
# # 创建一个包含R通道数据的三通道图像
# gray_2_rgb = np.zeros_like(rgb_img)
# gray_2_rgb[:, :, 0] = gray_img
# gray_2_rgb[:, :, 1] = gray_img
# gray_2_rgb[:, :, 2] = gray_img
#
# gray_2_rgb = np.float32(gray_2_rgb)
#
# # 保存包含R通道数据的三通道图像为PNG
# cv2.imwrite('output_gray_image.png', gray_2_rgb * 255)  # 乘以255将数据还原到0-255范围

a = [2,3]
print(a * 2)
