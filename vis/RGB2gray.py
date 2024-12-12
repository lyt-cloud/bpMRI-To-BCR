from PIL import Image
import os

# 输入图片路径
input_image_path = "./result_6363022_22.png"  # 替换为你的RGB图片路径

# 打开RGB图像
image = Image.open(input_image_path)

# 分离R、G、B通道
r_channel, g_channel, b_channel = image.split()

# 创建保存灰度图像的文件夹
output_folder_r = "gray_R"
output_folder_g = "gray_G"
output_folder_b = "gray_B"

os.makedirs(output_folder_r, exist_ok=True)
os.makedirs(output_folder_g, exist_ok=True)
os.makedirs(output_folder_b, exist_ok=True)

# 保存每个通道的灰度图像，保持原始文件名
image_filename = os.path.splitext(os.path.basename(input_image_path))[0]
r_channel.save(os.path.join(output_folder_r, f"{image_filename}_R.png"))
g_channel.save(os.path.join(output_folder_g, f"{image_filename}_G.png"))
b_channel.save(os.path.join(output_folder_b, f"{image_filename}_B.png"))

print("灰度图像已保存到文件夹:", output_folder_r, output_folder_g, output_folder_b)
