#!/usr/bin/env python
# coding=utf-8
import os
import json
import sys
# sys.path.append('../')
import matplotlib.pyplot as plt
import numpy as np
import warnings
import torch
# mpl.use('TkAgg')
from sklearn import metrics
from torchvision import transforms
from tqdm import tqdm
from models.efficientNet import efficientnet_b3 as efficientnet
from models.resnet import resnet50
from utils import ConfusionMatrix, Logger
from my_dataset import MyDataset_2D_2modal
warnings.filterwarnings("ignore")

def whole_data_path(data_fold, fold_path):
    """
    返回文件的完成路径
    """
    all_fold_data = []
    for i in data_fold:
        fold_name = i[:-2]  # dwi是-1 t2wi是-2
        label = i[-1:]  # 0，1
        fold_name = os.path.join(fold_path, fold_name)  # 文件夹的全路径
        single_fold_data = []
        for j in os.listdir(fold_name):
            single_fold_data.append(os.path.join(fold_name, j))
        single_fold_data = sorted(single_fold_data)
        single_fold_data.extend(label)
        # print('single_fold_data', single_fold_data)
        all_fold_data.append(single_fold_data)
        all_fold_data = sorted(all_fold_data)
    return all_fold_data

def predict_metric_slice(args, exp_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = args.model_name
    show_plt = args.show_plt
    print(device)
    if args.cross_val == True:
        for i in range(5):
            i += 1
            model_weight_path = os.path.join(exp_name, f"save_weights/best_model_fold{i}.pth")
            exp_name_save = os.path.join(exp_name, 'metrtic', 'slice')
            os.makedirs(exp_name_save, exist_ok=True)
            filename = os.path.join(exp_name_save, f'fold{i}_metric.txt')
            # logging.info('this should to write to the log file')
            # print(filename)
            sys.stdout = Logger(filename=filename, stream=sys.stdout)

            data_transform = transforms.Compose([
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.047,0.047,0.047],[0.948, 0.946,0.953])])


            test_folder = args.test_path
            fold_path = args.fold_path

            with open(test_folder, 'r') as file:
                test_data_list = file.read().splitlines()

            test_data = whole_data_path(test_data_list, fold_path)


            test_dataset = MyDataset_2D_2modal(test_data, transform=data_transform)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, num_workers=4)
            if model_name == 'resnet50':
                model = resnet50(num_classes=args.num_classes).to(args.device)
            elif model_name == 'eff':
                model = efficientnet(num_classes=args.num_classes).to(args.device)
            assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
            model.load_state_dict(torch.load(model_weight_path, map_location=device))
            model.to(device)



            # read class_indict
            json_label_path = './class_indices.json'
            assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
            json_file = open(json_label_path, 'r')
            class_indict = json.load(json_file)

            labels = [label for _, label in class_indict.items()]
            confusion = ConfusionMatrix(num_classes=2, labels=labels, save_path=exp_name_save + f'/fold{i}_metric.png', show_plt =show_plt)
            model.eval()

            predict_data = []
            y_true = []
            output_lable = []
            correct_test = 0
            with torch.no_grad():
                test_loader = tqdm(test_loader, file=sys.stdout)
                for step, image_label in enumerate(test_loader):
                    image_dwi = image_label[0].to(device)
                    image_t2 = image_label[1].to(device)
                    test_labels = image_label[2].to(device)

                    true_labels = test_labels.detach()
                    # true_labels = torch.squeeze(true_labels)
                    # 获取真实标签 以追加的方式
                    y_true.extend(true_labels.cpu().numpy())
                    # 获取模型输出结果
                    outputs = model(image_dwi, image_t2)
                    # 二分类用sigmod
                    # outputs = torch.sigmoid(outputs)
                    # 多分类用softmax
                    outputs = torch.softmax(outputs, dim=1)
                    # 降维
                    # outputs_data = torch.squeeze(outputs)  # 8 2 float
                    # print(outputs_data)
                    # 获取分类概率
                    predict_data.extend(outputs.cpu().numpy())  # array
                    # predict_data.extend(outputs_data.cpu().numpy().tolist())

                    # 获取分类标签结果
                    outputs = torch.argmax(outputs, dim=1)
                    output_lable.extend(outputs.cpu().numpy())

                    confusion.update(outputs.to("cpu").numpy(), test_labels.to("cpu").numpy())
            confusion.plot()
            confusion.summary()

            # print('预测标签', output_lable)
            # print('预测概率', predict_data)

            predict_probs = np.asarray(predict_data)
            # 获取正类的概率
            y_score = predict_probs[:, 1]
            # 真实标签
            y_true = np.asarray(y_true)

            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
            roc_auc = metrics.auc(fpr, tpr)  # 计算auc的值
            print('ROC ：', roc_auc)

            # 绘制roc曲线
            # plt.figure()
            lw = 2
            plt.figure(figsize=(10, 10))
            plt.plot(fpr, tpr, color='darkorange', lw=lw, label='LR ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(exp_name_save, f'fold{i}_auc.png'))
            if show_plt:
                plt.show()
            else:
                plt.close()
    elif args.cross_val ==False:
        model_weight_path = os.path.join(exp_name, f"save_weights/best_model.pth")
        exp_name_save = os.path.join(exp_name, 'metrtic', 'slice')
        os.makedirs(exp_name_save, exist_ok=True)
        filename = os.path.join(exp_name_save, f'metric.txt')
        sys.stdout = Logger(filename=filename, stream=sys.stdout)

        data_transform = transforms.Compose([
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.035, 0.041, 0.024], [0.855, 0.958, 0.807])])

        # file_path = os.path.join(exp_name, 'dataset', f"test.txt")
        # #
        # with open(file_path, 'r') as file:
        #     test_data = file.read().splitlines()

        # print(data_list)
        file_path = args.cv_test_folder
        fold_path = args.fold_path

        with open(file_path, 'r') as file:
            data_list = file.read().splitlines()
        test_data = whole_data_path(data_list, fold_path)

        test_dataset = MyDataset_gpt(test_data, transform=data_transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, num_workers=4)
        if model_name == 'resnet50':
            model = resnet50(num_classes=args.num_classes).to(args.device)
        elif model_name == 'eff':
            model = efficientnet(num_classes=args.num_classes).to(args.device)
        assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        model.to(device)

        # read class_indict
        json_label_path = './class_indices.json'
        assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
        json_file = open(json_label_path, 'r')
        class_indict = json.load(json_file)

        labels = [label for _, label in class_indict.items()]
        confusion = ConfusionMatrix(num_classes=2, labels=labels, save_path=exp_name_save + f'/metric.png',
                                    show_plt=show_plt)
        model.eval()

        predict_data = []
        y_true = []
        output_lable = []
        correct_test = 0
        with torch.no_grad():
            for test_data, labels in tqdm(test_loader):
                test_images, test_labels = test_data, labels
                true_labels = test_labels.detach()
                # true_labels = torch.squeeze(true_labels)
                # 获取真实标签 以追加的方式
                y_true.extend(true_labels.cpu().numpy())
                # 获取模型输出结果
                outputs = model(test_images.to(device))
                # 二分类用sigmod
                # outputs = torch.sigmoid(outputs)
                # 多分类用softmax
                outputs = torch.softmax(outputs, dim=1)

                # 降维
                # outputs_data = torch.squeeze(outputs)  # 8 2 float
                # print(outputs_data)
                # 获取分类概率
                predict_data.extend(outputs.cpu().numpy())  # array
                # predict_data.extend(outputs_data.cpu().numpy().tolist())

                # 获取分类标签结果
                outputs = torch.argmax(outputs, dim=1)
                output_lable.extend(outputs.cpu().numpy())

                confusion.update(outputs.to("cpu").numpy(), test_labels.to("cpu").numpy())
        confusion.plot()
        confusion.summary()

        # print('预测标签', output_lable)
        # print('预测概率', predict_data)

        predict_probs = np.asarray(predict_data)
        # 获取正类的概率
        y_score = predict_probs[:, 1]
        # 真实标签
        y_true = np.asarray(y_true)

        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        roc_auc = metrics.auc(fpr, tpr)  # 计算auc的值
        print('ROC ：', roc_auc)

        # 绘制roc曲线
        # plt.figure()
        lw = 2
        plt.figure(figsize=(10, 10))
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='LR ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(exp_name_save, f'roc.png'))
        if show_plt:
            plt.show()
        else:
            plt.close()



if __name__ == '__main__':
    predict_metric_slice()