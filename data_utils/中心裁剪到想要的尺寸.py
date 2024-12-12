from PIL import Image
import os

# 指定输入文件夹和输出文件夹
input_folder = '../datasets_figure/datasets_changhai_n4/diease_all'
output_folder = '../datasets_figure/datasets_changhai_n4/diease_128'

# 指定目标尺寸
target_width = 128
target_height = 128

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的PNG图像文件
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 打开图像文件
        img = Image.open(input_path)

        # 获取图像的宽度和高度
        width, height = img.size

        # 计算裁剪框的位置
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2

        # 中心裁剪图像
        img_cropped = img.crop((left, top, right, bottom))

        # 保存裁剪后的图像
        img_cropped.save(output_path)

        print(f"裁剪并保存：{output_path}")

print("任务完成")
