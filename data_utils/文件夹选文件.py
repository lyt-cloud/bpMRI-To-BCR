import os
import random
import shutil

def copy_random_images(source_folder, destination_folder, num_images_per_folder=1):
    os.makedirs(destination_folder, exist_ok=True)
    for root, dirs, files in os.walk(source_folder):
        image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        if not image_files:
            continue
        selected_images = random.sample(image_files, min(num_images_per_folder, len(image_files)))
        for image_file in selected_images:
            source_path = os.path.join(root, image_file)
            destination_path = os.path.join(destination_folder,image_file)
            shutil.copy(source_path, destination_path)
            print(f"Random image copied from {source_path} to {destination_path}")

if __name__ == "__main__":
    source_folder = "../外部验证数据/长海_260/diease_folder_rgb/R_folder"  # 替换为你的源文件夹路径
    destination_folder = "../外部验证数据/长海_260/diease_folder_rgb/R_folder_vis"  # 替换为你的目标文件夹路径
    num_images_per_folder = 1  # 每个子文件夹中想要复制的图片数量，可以调整为需要的数量
    copy_random_images(source_folder, destination_folder, num_images_per_folder)
