from PIL import Image
import os

# 源文件夹和目标文件夹的根目录
source_root = "../datasets/suda2_266/vis_image/diease_folder"
target_root = "../datasets/suda2_266/vis_image/diease_folder_rgb"

# 创建目标文件夹根目录
os.makedirs(target_root, exist_ok=True)
# 遍历子文件夹中的图片文件
for filename in os.listdir(source_root):
    if filename.endswith(".png"):
        image_path = os.path.join(source_root, filename)
        img = Image.open(image_path)

        target_R = os.path.join(target_root, 'R')
        target_G = os.path.join(target_root, 'G')
        target_B = os.path.join(target_root, 'B')
        # 创建文件夹的R、G、B文件夹
        os.makedirs(target_R, exist_ok=True)
        os.makedirs(target_G, exist_ok=True)
        os.makedirs(target_B, exist_ok=True)
        # 拆分RGB通道
        r, g, b = img.split()
        # 保存到目标文件夹
        r.save(os.path.join(target_R,  filename))
        g.save(os.path.join(target_G,  filename))
        b.save(os.path.join(target_B,  filename))

        print(f"Processed: {image_path}")

print("Done")
