import argparse
import json
import os
import warnings

import numpy as np
from PIL import Image
from torchvision import transforms
from utils import GradCAM, show_cam_on_image
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
# from models.resnet import resnet50 as resnet50
from models.efficientNet import efficientnet_b3 as efficientnet
from models.efficientNet_orgin import efficientnet_b3 as efficientnet_orgin
from models.swin_transformer import swin_tiny_patch4_window7_224 as swin_tiny
from models.swin_transformer_orgin import swin_tiny_patch4_window7_224 as swin_tiny_orgin
warnings.filterwarnings("ignore")  # 忽略警告


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=
                        '../实验记录_swin/best/exp1_ce_clinic_SGD_datasets_30_suzhou_swin_tiny_bs16_epoch50_lr0.001_kflod5_seed113'
                        , help='实验保存的路径主目录')
    parser.add_argument('--best_fold', type=int, default=4, help='选取最佳结果的一折绘图')
    parser.add_argument('--model_name', type=str, default='eff', help='resnet50 和 eff ')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--show_plt', type=bool, default=False, help='是否展示出来每个图片的注意力图,false就是只保存下来,而不会展示')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--clinical', type=bool, default=False, help='是否使用临床信息')  # 不传入默认False 传入就是True
    parser.add_argument('--clinical_json_path', type=str, default='./datasets/bcr_clinlic.json', help='临床信息所在的位置')
    args = parser.parse_args()

    model_name = args.model_name
    device = args.device

    norm_mean = [0.014, 0.013, 0.005]
    norm_std = [0.378, 0.422, 0.986]
    data_transform = transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize(norm_mean, norm_std)])


    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)
    # create model
    if args.model_name == 'resnet50':
        pass

    elif args.model_name == 'swin_tiny':
        if args.clinical == True:
            model = swin_tiny(num_classes=args.num_classes).to(args.device)
        if args.clinical == False:
            model = swin_tiny_orgin(num_classes=args.num_classes).to(args.device)

    elif args.model_name == 'eff':
        if args.clinical == True:
            model = efficientnet(num_classes=args.num_classes).to(args.device)
        if args.clinical == False:
            model = efficientnet_orgin(num_classes=args.num_classes).to(args.device)

    # load model weights
    weights_path = os.path.join(args.exp_name, f"save_weights/best_model_fold{args.best_fold}.pth")
    # weights_path = "./best_model_fold6.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # 导入图片
    vis_folder = '../datasets/bcr_diease_vis'
    predict_vis = args.exp_name + '/vis'
    os.makedirs(predict_vis, exist_ok=True)
    for root, dirs, files in os.walk(vis_folder):
        image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        if not image_files:
            continue
        for image_file in image_files:
            img_path = os.path.join(root, image_file)
            # print(image_path)
            img = Image.open(img_path).convert('RGB')
            plt.imshow(img)
            # [N, C, H, W]
            img = data_transform(img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)
            # resnet
            # target_layers = [net.layer4[-1]]
            # print(target_layers)
            # efficientnet
            if model_name == 'resnet50':
                target_layers = [model.layer4[-1]]
            elif model_name == 'eff':
                target_layers = [model.features]
            # target_layers = [model.features]
            # print(target_layers)
            # prediction
            model.eval()
            with torch.no_grad():
                # predict class
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()

            print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                         predict[predict_cla].numpy())
            plt.title(print_res)
            for i in range(len(predict)):
                print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                          predict[i].numpy()))
            if args.show_plt:
                plt.show()
            else:
                plt.close()

            # image_size = 256#训练图像的尺寸，在你训练图像的时候图像尺寸是多少这里就填多少
            assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
            img = Image.open(img_path).convert('RGB')#将图片转成RGB格式的
            img = np.array(img, dtype=np.uint8) #转成np格式
            img2 = img[:,:,0:1] # 1是T2WI 2是DWI 3是ROI
            # img = center_crop_img(img, image_size) #将测试图像裁剪成跟训练图片尺寸相同大小的
            # [C, H, W]
            img_tensor = data_transform(img)#简单预处理将图片转化为张量
            # expand batch dimension
            # [C, H, W] -> [N, C, H, W]
            input_tensor = torch.unsqueeze(img_tensor, dim=0) #增加一个batch维度
            # print(input_tensor)
            # print(input_tensor.shape)

            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
            grayscale_cam = cam(input_tensor=input_tensor)
            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,     # 1是T2WI 2是DWI 3是ROI
                                              grayscale_cam,
                                              use_rgb=True)
            plt.imshow(visualization)
            save_path = os.path.join(predict_vis, 'result_' + image_file)
            # print(save_path)
            plt.savefig(save_path)#将热力图的结果保存到本地当前文件夹
            if args.show_plt:
                plt.show()
            else:
                plt.close()


if __name__ == '__main__':
    main()