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
# from models.resnet import resnet50 as create_model
from models.efficientNet import efficientnet_b3 as create_model

warnings.filterwarnings("ignore")  # 忽略警告
def main():
    net =create_model(num_classes=2)
    device = torch.device("cpu")
    # print(net)
    ### dwi_eff best_model_fold3.pth
    net.load_state_dict(torch.load("./best_model_fold3.pth", map_location=device))  # 载入训练模型权重，你将训练的模型权重放到当前文件夹下即可
    # target_layers = [net.layer4[-1]] #这里是 看你是想看那一层的输出，我这里是打印最后一层的输出，

    # resnet
    # target_layers = [net.layer4[-1]]
    # print(target_layers)
    # efficientnet
    target_layers = [net.features]
    # print(target_layers)
    data_transform = transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.048, 0.048, 0.048], [0.95, 0.95, 0.95])])
    # 导入图片
    # img_path = "N151847-1_12.png"  #这里是导入你需要测试图片
    # img_path = "./vis_t2wi/1/N153976-2_10.png"  #这里是导入你需要测试图片
    img_path = "./N150364-2_15.png"  #这里是导入你需要测试图片


    # prdict
    img = Image.open(img_path).convert('RGB')
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=2).to(device)
    # load model weights
    weights_path = "./best_model_fold3.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

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
    plt.show()





    image_size = 256#训练图像的尺寸，在你训练图像的时候图像尺寸是多少这里就填多少
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')#将图片转成RGB格式的
    img = np.array(img, dtype=np.uint8) #转成np格式
    # img = center_crop_img(img, image_size) #将测试图像裁剪成跟训练图片尺寸相同大小的
    # [C, H, W]
    img_tensor = data_transform(img)#简单预处理将图片转化为张量
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0) #增加一个batch维度
    print(input_tensor)
    print(input_tensor.shape)
    cam = GradCAM(model=net, target_layers=target_layers, use_cuda=False)
    grayscale_cam = cam(input_tensor=input_tensor)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    save_path ='./results_' + img_path.split('/')[1]
    # print(save_path)
    plt.savefig(save_path)#将热力图的结果保存到本地当前文件夹
    plt.show()
if __name__ == '__main__':
    main()