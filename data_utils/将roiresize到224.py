from PIL import Image
import os

# 输入文件夹和输出文件夹
input_folder = '../datasets_128/bcr_diease_vis_rgb/B'
output_folder = '../datasets_128/bcr_diease_vis_rgb/B_224'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 列出输入文件夹中的所有文件
file_list = os.listdir(input_folder)

for filename in file_list:
    if filename.endswith('.png'):
        # 打开原始图片
        input_path = os.path.join(input_folder, filename)
        image = Image.open(input_path)

        # 调整大小为224x224
        new_image = image.resize((224, 224), Image.NEAREST)

        # 保存调整大小后的图片到输出文件夹
        output_path = os.path.join(output_folder, filename)
        new_image.save(output_path)

print("图片调整大小完成")
