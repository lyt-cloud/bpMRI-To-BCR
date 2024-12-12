import argparse
import json
import os
import warnings

import numpy as np
from PIL import Image
from torchvision import transforms

import torch
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

from models.swin_transformer import swin_tiny_patch4_window7_224 as swin_tiny
from models.swin_transformer_orgin import swin_tiny_patch4_window7_224 as swin_tiny_orgin

from models.swin_transformer import swin_base_patch4_window7_224_in22k as swin_base
from models.swin_transformer_orgin import swin_base_patch4_window7_224_in22k as swin_base_orgin


warnings.filterwarnings("ignore")  # 忽略警告
import argparse
import cv2
import numpy as np
import torch


from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit


def reshape_transform(tensor, height=7, width=7):
    ##第一层height、wideth设置为28，第二层height、wideth设置为14，第三、四层height、wideth设置为7
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def crop_center(image, crop_width, crop_height):
    # 读取图像
    # image = cv2.imread(image_path)

    # 获取图像尺寸
    height, width, _ = image.shape

    # 计算中心点
    center_x, center_y = width // 2, height // 2

    # 计算裁剪区域
    x1 = max(center_x - crop_width // 2, 0)
    y1 = max(center_y - crop_height // 2, 0)
    x2 = min(center_x + crop_width // 2, width)
    y2 = min(center_y + crop_height // 2, height)

    # 裁剪图像
    cropped_image = image[y1:y2, x1:x2]

    return cropped_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=
                        '../final_model/clinic'
                        , help='实验保存的路径主目录')
    parser.add_argument('--best_fold', type=int, default=4, help='选取最佳结果的一折绘图')
    parser.add_argument('--model_name', type=str, default='swin_base', help='resnet50 和 eff 和 mobile swin_tiny')

    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--show_plt', type=bool, default=False, help='是否展示出来每个图片的注意力图,false就是只保存下来,而不会展示')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--clinical', type=bool, default=True, help='是否使用临床信息')  # 不传入默认False 传入就是True
    parser.add_argument('--clinical_json_path', type=str, default='../datasets_all/bcr_clinic.json', help='临床信息所在的位置')

    parser.add_argument('--use-cuda', action='store_true', default=True,help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png', help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true', help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true', help='Reduce noise by taking the first principle componenet of cam_weights*activations')

    parser.add_argument('--method', type=str, default='scorecam', help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')
    args = parser.parse_args()

    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')
    device = args.device
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    if args.model_name == 'swin_tiny':
        if args.clinical == True:
            model = swin_tiny(num_classes=args.num_classes).to(args.device)
        if args.clinical == False:
            model = swin_tiny_orgin(num_classes=args.num_classes).to(args.device)
    elif args.model_name == 'swin_base':
        if args.clinical == True:
            model = swin_base(num_classes=args.num_classes).to(args.device)
        if args.clinical == False:
            model = swin_base_orgin(num_classes=args.num_classes).to(args.device)

    if args.use_cuda:
        model = model.cuda()
    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")


    # load model weights
    weights_path = os.path.join(args.exp_name, f"save_weights/best_model_fold{args.best_fold}.pth")
    # weights_path = os.path.join(args.exp_name, f"best_model_fold2.pth")
    weights_dict = torch.load(weights_path, map_location=device)
    fc_keys = [k for k in weights_dict.keys() if "clinical" or "mcb" in k]
    for k in fc_keys:
        del weights_dict[k]

    # weights_path = "./best_model_fold6.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)

    # model.load_state_dict(torch.load(weights_path, map_location=device))
    model.load_state_dict(weights_dict, strict=False)

    # 导入图片
    # vis_folder = '../datasets_figure/datasets_changhai_n4/bcr_diease_vis_rgb/R'
    vis_folder = '../datasets/suda2_266/vis_image/diease_folder'
    predict_vis = args.exp_name + '/vis_rgb'
    os.makedirs(predict_vis, exist_ok=True)
    for root, dirs, files in os.walk(vis_folder):
        image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        if not image_files:
            continue
        for image_file in image_files:
            img_path = os.path.join(root, image_file)

            target_layers = [model.layers[-1].blocks[-1].norm2]
            # target_layers = [model.norm]    ##swin最后一层
            print("目标层：", target_layers)
            model.eval()
            if args.method == "ablationcam":
                cam = methods[args.method](model=model,
                                           target_layers=target_layers,
                                           use_cuda=args.use_cuda,
                                           reshape_transform=reshape_transform,
                                           ablation_layer=AblationLayerVit())
            else:
                cam = methods[args.method](model=model,
                                           target_layers=target_layers,
                                           use_cuda=args.use_cuda,
                                           reshape_transform=reshape_transform)

            assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
            rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]  # BGR-RGB

            # rgb_img = cv2.resize(rgb_img, (224, 224))
            rgb_img = crop_center(rgb_img, 224, 224)
            rgb_img = np.float32(rgb_img) / 255


            # norm_mean = [0.014, 0.013, 0.005]
            # norm_std = [0.378, 0.422, 0.986]
            input_tensor = preprocess_image(rgb_img,)

            cam.batch_size = 32
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=None,
                                eigen_smooth=args.eigen_smooth,
                                aug_smooth=args.aug_smooth)
            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :]


            gray_img = rgb_img[:, :, 0]  # 通道顺序为RGB，R通道在索引0处
            # 创建一个包含R通道数据的三通道图像
            # T2WI-2    dwi-1    ror-0 进行可视化将只显示在T2WI
            gray_2_rgb = np.zeros_like(rgb_img)
            gray_2_rgb[:, :, 0] = gray_img
            gray_2_rgb[:, :, 1] = gray_img
            gray_2_rgb[:, :, 2] = gray_img

            cam_image = show_cam_on_image(gray_2_rgb, grayscale_cam)
            save_path = os.path.join(predict_vis,'result_' + image_file)
            cv2.imwrite(save_path, cam_image)



if __name__ == '__main__':
    main()