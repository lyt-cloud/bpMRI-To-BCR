import argparse
import csv
import json
import logging
import random
import sys
import os

import pandas as pd
from sklearn import preprocessing
from torch.optim import lr_scheduler

sys.path.append("..")
import torch
import torch.optim as optim
from numpy import *
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter
from models.efficientNet1_modal import efficientnet_b3 as efficientnet1modal
from models.efficientNet import efficientnet_b3 as efficientnet2modal
from models.resnet import resnet50 as resnet50
from models.resnet import resnet34 as resnet34
from my_dataset import MyDataset_2D_1modal, MyDataset_2D_2modal, MyDataset_2D_3modal
from utils import train_one_epoch, evaluate, EarlyStopping, read_train_data, test_metric
import math

"""
训练脚本
"""


def train_val_split_data(data, train_ratio, val_ratio, test_ratio=0, no_test=True):
    """
    分割数据成训练集、验证集和测试集。

    参数：
    data (list): 包含数据的列表，每个元素是一个样本，最后一个元素是类别标签（'0'或'1'）。
    train_ratio (float): 训练集比例。
    val_ratio (float): 验证集比例。
    test_ratio (float): 测试集比例。

    返回：
    train_set (list): 训练集样本列表。
    val_set (list): 验证集样本列表。
    test_set (list): 测试集样本列表。
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "比例之和必须等于1.0"

    # 分离 '1' 和 '0' 的样本
    # samples_0 = [item for item in data if item[-1] == '0']
    # samples_1 = [item for item in data if item[-1] == '1']
    samples_0 = [item for item in data if item.endswith(',0')]
    samples_1 = [item for item in data if item.endswith(',1')]

    # 计算每个类别应该有多少个样本
    train_samples_0 = int(len(samples_0) * train_ratio)
    val_samples_0 = int(len(samples_0) * val_ratio)
    train_samples_1 = int(len(samples_1) * train_ratio)
    val_samples_1 = int(len(samples_1) * val_ratio)

    # 随机选择训练集、验证集和测试集
    random.shuffle(samples_0)
    random.shuffle(samples_1)

    train_set = samples_0[:train_samples_0] + samples_1[:train_samples_1]
    val_set = samples_0[train_samples_0:train_samples_0 + val_samples_0] + samples_1[
                                                                           train_samples_1:train_samples_1 + val_samples_1]

    if not no_test:
        # 随机选择测试集
        test_set = samples_0[train_samples_0 + val_samples_0:] + samples_1[train_samples_1 + val_samples_1:]
        return train_set, val_set, test_set
    else:
        return train_set, val_set


def whole_data_path1(data_path, fold_path):
    """
    返回文件的完成路径
    """
    data = []
    for i in data_path:
        fold_name = i[:-2]  # dwi是-1 t2wi是-2
        label = i[-1:]  # 0，1
        fold_name = os.path.join(fold_path, fold_name)

        data_l = [os.path.join(fold_name, j) + "," + label for j in os.listdir(fold_name)]
        for j in data_l:
            data.append(j)
    return data


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


def train_cross_val(args, exp_name, data_transform):
    # 创建交叉验证的txt文件路径
    os.makedirs(os.path.join(exp_name, "fold"), exist_ok=True)
    tb_writer = SummaryWriter(log_dir=os.path.join(exp_name, 'runs'))
    device = args.device
    # 读取数据
    file_path = args.file_path_cv
    data_folder = []
    label = []
    with open(file_path, 'r') as file:
        data_list = file.read().splitlines()
        for i in data_list:
            data_folder.append(i.split(',')[0])
            label.append(i.split(',')[1])
    k = args.k
    kfold = StratifiedKFold(n_splits=k, shuffle=False)
    for fold, (train_fold_indices, test_fold_indices) in enumerate(kfold.split(data_folder, label)):
        # train_fold_indices, val_fold_indices = train_val_spilt(train_val_fold_indices, ratio=0.25)
        train_val_fold_data = [data_list[i] for i in train_fold_indices]
        train_fold_data, val_fold_data = train_val_split_data(train_val_fold_data, train_ratio=0.75, val_ratio=0.25)
        test_fold_data = [data_list[i] for i in test_fold_indices]

        print('train_fold_data', train_fold_data)
        print('val_fold_data', val_fold_data)
        print('test_fold_data', test_fold_data)

        fold_path = args.fold_path
        # 获取全部路径
        train_data = whole_data_path(train_fold_data, fold_path)
        val_data = whole_data_path(val_fold_data, fold_path)
        test_data = whole_data_path(test_fold_data, fold_path)

        fold_file_path = os.path.join(exp_name, f'fold/fold_{fold + 1}_train_fold.txt')
        with open(fold_file_path, 'w') as fold_file:
            for data_path in train_fold_data:
                fold_file.write(data_path + '\n')
        fold_file_path = os.path.join(exp_name, f'fold/fold_{fold + 1}_val_fold.txt')
        with open(fold_file_path, 'w') as fold_file:
            for data_path in val_fold_data:
                fold_file.write(data_path + '\n')

        fold_file_path = os.path.join(exp_name, f'fold/fold_{fold + 1}_test_fold.txt')
        with open(fold_file_path, 'w') as fold_file:
            for data_path in test_fold_data:
                fold_file.write(data_path + '\n')

        # 打乱顺序
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)

        print("{} images for training.".format(len(train_data)))
        print("{} images for valing.".format(len(val_data)))
        print("{} images for testing.".format(len(test_data)))


        ## 以后修改,会输入一个modal_num
        if args.modal_num == 1:
            train_dataset = MyDataset_2D_1modal(train_data, transform=data_transform['train'])
            val_dataset = MyDataset_2D_1modal(val_data, transform=data_transform['val'])
            test_dataset = MyDataset_2D_1modal(test_data, transform=data_transform['val'])
        elif args.modal_num == 2:
            train_dataset = MyDataset_2D_2modal(train_data, transform=data_transform['train'])
            val_dataset = MyDataset_2D_2modal(val_data, transform=data_transform['val'])
            test_dataset = MyDataset_2D_2modal(test_data, transform=data_transform['val'])
        elif args.modal_num == 3:
            train_dataset = MyDataset_2D_3modal(train_data, transform=data_transform['train'])
            val_dataset = MyDataset_2D_3modal(val_data, transform=data_transform['val'])
            test_dataset = MyDataset_2D_3modal(test_data, transform=data_transform['val'])
        else:
            print("请根据上面的MyDataset_2D_1modal进行修改")

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.nw)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.nw)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.nw)

        # 临床特征加载
        if args.clinical == True:
            clinical_json_address = args.clinical_json_path
            with open(clinical_json_address, encoding='utf-8') as f:
                json_dict = json.load(f)
            peoples = [i for i in json_dict]
            features = np.array([json_dict[i] for i in json_dict], dtype=np.float32)
            # clinical_features = features
            # 特征归一化
            min_max_scaler = preprocessing.MinMaxScaler()
            clinical_features = min_max_scaler.fit_transform(features)

        if args.model_name == 'resnet50':
            model = resnet50(num_classes=args.num_classes, clinical=args.clinical).to(args.device)
        elif args.model_name == 'eff':
            if args.modal_num == 1:
                model = efficientnet1modal(num_classes=args.num_classes, modal_num=args.modal_num).to(args.device)
            elif args.modal_num == 2 and args.clinical == True:
                model = efficientnet2modal(num_classes=args.num_classes, modal_num=args.modal_num, clinical=args.clinical).to(args.device)

        if fold == 0:
            print(model)
            # pass
        if args.weights != "":
            if os.path.exists(args.weights):
                # 加载所有权重
                weights_dict = torch.load(args.weights, map_location=device)
                load_weights_dict = {k: v for k, v in weights_dict.items()
                                     if model.state_dict()[k].numel() == v.numel()}
                print(model.load_state_dict(load_weights_dict, strict=False))

                # 加载部分权重
            #     weights_dict = torch.load(args.weights, map_location=device)
            #     del_key = []
            #     for key, _ in weights_dict.items():  # 遍历预训练权重的有序字典
            #         del_key.append(key)
            #     n = len(del_key)
            #     bek = []
            #     for i in range(n):
            #         if "features.stem_conv" in del_key[i] or "features.1a" in del_key[i] or "features.1b" in del_key[i] \
            #                 or "features.2a" in del_key[i] or "features.2b" in del_key[i] or "features.2c" in del_key[
            #             i]:  # 需要保留的权重
            #             bek.append(i)
            #     del_key = [del_key[i] for i in range(n) if (i not in bek)]
            #     print("del_key", del_key)
            #     for key in del_key:  # 遍历要删除字段的list
            #         del weights_dict[key]  # 删除预训练权重的key和对应的value
            #     missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
            #     print('missing_keys', missing_keys)
            #     print('unexpected_keys', unexpected_keys)
            # else:
            #     raise FileNotFoundError("not found weights file: {}".format(args.weights))

        # 是否冻结权重
        if args.freeze_layers:
            for name, para in model.named_parameters():
                # 除最后一个卷积层和全连接层外，其他权重全部冻结 只训练以下这些
                if ("features.top" not in name) and ("classifier" not in name) \
                        and ("features.7a" not in name) and ("features.7b" not in name) \
                        and ("features.6a" not in name) and ("features.6b" not in name) and ("features.6c" not in name) \
                        and ("features.6d" not in name) and ("features.6e" not in name) and ("features.6f" not in name) \
                        and ("features.5a" not in name) and ("features.5b" not in name) and ("features.5c" not in name) \
                        and ("features.5d" not in name) and ("features.5e" not in name) and ("features.5f" not in name) \
                        and ("features.4a" not in name) and ("features.4b" not in name) and ("features.4c" not in name) \
                        and ("features.4d" not in name) and ("features.4e" not in name) and ("features.4f" not in name):
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))

        pg = [p for p in model.parameters() if p.requires_grad]
        # optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-3)
        optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=1E-3)
        # optimizer = optim.Adam(pg, lr=args.lr, weight_decay=1E-3)
        lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)

        best_acc = 0.
        low_loss = 10.
        val_acc_max_l = []
        train_acc_max_l = []
        train_loss_min_l = []
        val_loss_min_l = []
        early_stopping = EarlyStopping(patience=50, verbose=True, monitor='val_ap', op_type='max')
        for epoch in range(args.epochs):
            # train
            train_loss, train_acc = train_one_epoch(model=model,
                                                    optimizer=optimizer,
                                                    data_loader=train_loader,
                                                    device=device,
                                                    epoch=epoch,
                                                    modal_num=args.modal_num,
                                                    peoples=peoples,
                                                    clinical_feat=clinical_features,
                                                    kfold=fold + 1,
                                                    )

            scheduler.step()
            # validate
            val_loss, val_acc = evaluate(model=model,
                                         data_loader=val_loader,
                                         device=device,
                                         epoch=epoch,
                                         modal_num=args.modal_num,
                                         peoples=peoples,
                                         clinical_feat=clinical_features,
                                         kfold=fold + 1)
            val_ap = val_acc
            tags = [f'train_loss_fold{fold + 1}', f"train_acc_fold{fold + 1}", f"val_loss_fold{fold + 1}",
                    f"val_acc_fold{fold + 1}", f"learning_rate_fold{fold + 1}"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_acc, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

            logging.info(
                'fold %d :  epoch : %d, train_loss : %f,  train_acc: %f val_loss : %f,  val_acc: %f  learning_rate :%f' %
                (fold + 1, epoch, train_loss, train_acc, val_loss, val_acc, optimizer.param_groups[0]["lr"]))

            if epoch > 3 and args.save_best is True:
                early_stopping(val_ap)
                if val_ap == 1:
                    break
                # # early stopping
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                # if args.save_best is True:
                if best_acc <= val_acc:
                    best_acc = val_acc
                    os.makedirs(os.path.join(exp_name, "save_weights/model_fold{}".format(fold + 1)), exist_ok=True)
                    torch.save(model.state_dict(),
                               os.path.join(exp_name, "save_weights/best_model_fold{}.pth".format(fold + 1)))
                    torch.save(model.state_dict(),
                               os.path.join(exp_name,
                                            "save_weights/model_fold{}/best_model_fold{}_valacc{:.3f}.pth".format(
                                                fold + 1, epoch, best_acc)))
                else:
                    continue
                if low_loss >= val_loss:
                    low_loss = val_loss
                    torch.save(model.state_dict(),
                               os.path.join(exp_name, "save_weights/low_loss_model_fold{}.pth".format(fold + 1)))
                    torch.save(model.state_dict(),
                               os.path.join(exp_name,
                                            "save_weights/model_fold{}/low_loss_model_fold{}_low_loss{:.3f}_valacc{}.pth".format(
                                                fold + 1, epoch, low_loss, val_acc)))

            # 将所有的验证结果保存到一个列表 方便提取最好的结果
            train_acc_max_l.append(train_acc)
            train_loss_min_l.append(train_loss)
            val_acc_max_l.append(val_acc)
            val_loss_min_l.append(val_loss)
        # 打印训练结果并保存
        index_max = val_acc_max_l.index(max(val_acc_max_l))
        f = open(os.path.join(exp_name, "train_acc.txt"), "a")
        if fold == 0:
            f.write("fold" + "\t" + "train_loss" + "\t" + "train_acc" + "\t" + "val_acc" +
                    "\t"+ "val_loss")
        f.write('\n' + "fold" + str(fold + 1) + ":\t" + str('%.3f' % train_loss_min_l[index_max]) + " ;\t" + str(
            '%.3f' % train_acc_max_l[index_max]) + " ;\t" + str('%.3f' % val_acc_max_l[index_max])
                + " ;\t" + str('%.3f' % val_loss_min_l[index_max]))
        f.close()

        model_weight = os.path.join(exp_name, "save_weights/best_model_fold{}.pth".format(fold + 1))
        metric = test_metric(args, exp_name=exp_name, i=fold + 1, model=model, model_weight=model_weight,
                             peoples=peoples,clinical_feat=clinical_features,
                             data_loader=test_loader, device=device)
        print(metric)
        test_metric_save = os.path.join(exp_name, 'metrtic', 'metric.csv')
        field_names = ["fold", "acc", "precision", "recall", "specificity", "F1_score", "roc_auc"]
        with open(test_metric_save, mode='a', newline='') as csv_file:
            if fold + 1 == 1:
                writer = csv.DictWriter(csv_file, fieldnames=field_names)
                writer.writeheader()
                writer2 = csv.writer(csv_file)
                writer2.writerow(metric)
            else:
                writer = csv.writer(csv_file)
                writer.writerow(metric)
        # 每一折之后清除显存
        torch.cuda.empty_cache()

    df = pd.read_csv(test_metric_save)

    # 计算每列的平均值
    average_values = df[["acc", "precision", "recall", "specificity", "F1_score", "roc_auc"]].mean().round(3)
    # 转换平均值为一个字典
    average_dict = average_values.to_dict()

    # 打开CSV文件以追加数据
    with open(test_metric_save, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=df.columns)
        # 写入平均值数据
        writer.writerow({"fold": "Average", **average_dict})


def train(args, exp_name, data_transform):
    # def train(args, data_transform):
    # 创建交叉验证的txt文件路径
    # exp_name = args.exp_name
    os.makedirs(exp_name, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=os.path.join(exp_name, 'runs'))
    device = args.device
    # 读取数据
    train_data = args.train_path
    val_data = args.val_path
    test_data = args.test_path

    # 获取文件夹名及标签
    with open(train_data, 'r') as file:
        train_data_list = file.read().splitlines()
    with open(val_data, 'r') as file:
        val_data_list = file.read().splitlines()
    with open(test_data, 'r') as file:
        test_data_list = file.read().splitlines()

    fold_path = args.fold_path
    train_data = whole_data_path(train_data_list, fold_path)
    val_data = whole_data_path(val_data_list, fold_path)
    test_data = whole_data_path(test_data_list, fold_path)
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    print("{} images for training.".format(len(train_data)))
    print("{} images for valing.".format(len(val_data)))
    print("{} images for testing.".format(len(test_data)))
    # 保存训练验证测试集的信息
    dataset_name = os.path.join(exp_name, 'dataset')
    os.makedirs(dataset_name, exist_ok=True)
    fold_file_path = os.path.join(dataset_name, f'train.txt')
    # print(train_data)
    with open(fold_file_path, 'w') as fold_file:
        for data_path in train_data:
            fold_file.write(data_path + '\n')
    fold_file_path = os.path.join(dataset_name, f'val.txt')
    with open(fold_file_path, 'w') as fold_file:
        for data_path in val_data:
            fold_file.write(data_path + '\n')
    fold_file_path = os.path.join(dataset_name, f'test.txt')
    with open(fold_file_path, 'w') as fold_file:
        for data_path in test_data:
            fold_file.write(data_path + '\n')

    ## 创建dataloader
    train_dataset = MyDataset_2D_2modal(train_data, transform=data_transform['train'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.nw)
    val_dataset = MyDataset_2D_2modal(val_data, transform=data_transform['val'])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.nw)

    # 选择模型
    if args.model_name == 'resnet50':
        model = resnet50(num_classes=args.num_classes).to(args.device)
    elif args.model_name == 'eff':
        model = efficientnet(num_classes=args.num_classes).to(args.device)
    # 打印模型
    print(model)
    # 载入预训练权重
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # 是否冻结权重
    if args.freeze_layers:
        weights_dict = torch.load(args.weights, map_location=device)
        del_key = []
        for key, _ in weights_dict.items():  # 遍历预训练权重的有序字典
            del_key.append(key)
        n = len(del_key)
        bek = []
        for i in range(n):
            if "features.stem_conv" in del_key[i] or "features.1a" in del_key[i] or "features.1b" in del_key[i] \
                    or "features.2a" in del_key[i] or "features.2b" in del_key[i] or "features.2c" in del_key[
                i]:  # 需要保留的权重
                bek.append(i)
        del_key = [del_key[i] for i in range(n) if (i not in bek)]
        print("del_key", del_key)
        for key in del_key:  # 遍历要删除字段的list
            del weights_dict[key]  # 删除预训练权重的key和对应的value
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        print('missing_keys', missing_keys)
        print('unexpected_keys', unexpected_keys)

    # 网络优化器件和学习率衰减
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-2)
    # optimizer = optim.AdamW(pg, lr=args.lr, weight_decay= 0.1)

    # # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)

    best_acc = 0.
    early_stopping = EarlyStopping(patience=50, verbose=True, monitor='val_ap', op_type='max')
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                kfold=None)

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch,
                                     kfold=None)
        val_ap = val_acc
        tags = [f'train_loss', f"train_acc", f"val_loss",
                f"val_acc", f"learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        logging.info(
            'epoch : %d, train_loss : %f,  train_acc: %f val_loss : %f,  val_acc: %f  learning_rate :%f' %
            (epoch, train_loss, train_acc, val_loss, val_acc, optimizer.param_groups[0]["lr"]))
        # 早停
        # early_stopping(val_ap)
        # if val_ap == 1:
        #     break
        # # early stopping
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break
        # 前几个epoch的权重不能要
        # if epoch > 2 and args.save_best is True:
        if args.save_best is True:
            early_stopping(val_ap)
            if val_ap == 1:
                break
            # early stopping
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # if args.save_best is True:
            if best_acc <= val_acc:
                best_acc = val_acc
                os.makedirs(os.path.join(exp_name, "save_weights"), exist_ok=True)
                torch.save(model.state_dict(),
                           os.path.join(exp_name, "save_weights/best_model.pth".format(epoch)))
                torch.save(model.state_dict(),
                           os.path.join(exp_name, "save_weights/best_model_epoch{}_valacc{:.3f}.pth".format(
                               epoch, best_acc)))
            else:
                continue

        f = open(os.path.join(exp_name, "results.txt"), "w")
        f.write("train_acc :" + str('%.3f' % train_acc) + "\n"
                + "train_loss :" + str('%.3f' % train_loss) + "\n"
                + "val_acc : " + str('%.3f' % val_acc) + "\n"
                + "val_loss : " + str('%.3f' % val_loss))
        f.close()
    torch.cuda.empty_cache()  # 每个epoch结束释放显存


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--cross_val', type=bool, default=True, help='是否使用交叉验证')
    parser.add_argument('--exp_name', type=str, default='./exp1', help='实验保存的路径主目录')
    parser.add_argument('--model_name', type=str, default='resnet50', help='resnet50 和 eff ')
    parser.add_argument('--weights', type=str, default='./preweight/resnet50.pth',
                        help='initial weights path')
    parser.add_argument('--file_path_cv', type=str, default='../dataset/train_fold.txt', help='交叉验证训练txt文件路径')
    parser.add_argument('--fold_path', type=str,
                        default='../dataset/2modal_d_t',
                        help='目录对应的文件夹所在的位置')
    parser.add_argument('--k', type=int, default=5, help='k fold')
    parser.add_argument('--show_plt', type=bool, default=False, help='是否展示出来混淆矩阵和roc曲线,false就是只保存下来,而不会展示')
    parser.add_argument('--nw', type=int, default=4, help='num worker')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lrf', type=float, default=0.0001)
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--save_best', default=True, type=bool, help='only save best dice weights')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    opt = parser.parse_args()

    train(opt)
