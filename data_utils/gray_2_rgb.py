from PIL import Image
import os

# 根文件夹路径
root_folder = '../datasets_figure/datasets_changhai_n4/bcr_diease_vis_rgb/R'
output_root = '../datasets_figure/datasets_changhai_n4/bcr_diease_vis_rgb/T2WI'  # 新的根文件夹路径

# 遍历根文件夹中的所有子文件夹和文件
for root, dirs, files in os.walk(root_folder):
    for file in files:
        # 检查文件扩展名是否为图片文件（例如，.jpg, .png）
        if file.endswith(('.jpg', '.png', '.bmp', '.jpeg', '.gif', '.tiff')):
            file_path = os.path.join(root, file)

            # 打开灰度图片
            gray_image = Image.open(file_path).convert('L')

            # 创建一个新的RGB图片
            rgb_image = Image.new("RGB", gray_image.size)
            rgb_image.paste(gray_image)

            # 保存RGB图片到新的根文件夹下，保持原文件名
            relative_path = os.path.relpath(file_path, root_folder)
            output_path = os.path.join(output_root, relative_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 确保目录存在
            rgb_image.save(output_path)

print("转换完成！")