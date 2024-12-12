import cv2
import numpy as np
from PIL import Image


image1 = './gray/6323889_8_t2.png'
image2 = './gray/6323889_8_d.png'

def t2_d_to_rgb(img1,img2):
    gray_img1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
    gray_img2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)
    gray_img1 = np.expand_dims(gray_img1, axis=-1)
    gray_img2 = np.expand_dims(gray_img2, axis=-1)
    zero2_gray = np.zeros_like(gray_img2)

    rgb_img = np.concatenate((zero2_gray, gray_img2, gray_img1), axis=-1)  # BGR
    print(rgb_img[:, :, 0])
    # print(rgb_img.shape)
    png_save = './'
    cv2.imwrite(png_save, rgb_img)


img1_folder = './datasets_30_suzhou/diease_folder_rgb/G'

