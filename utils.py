import csv
import json
import os
import pickle
import random
import sys
import matplotlib.pyplot as plt
import torch
from prettytable import PrettyTable
from sklearn.metrics import roc_auc_score
from torch import nn
from tqdm import tqdm
from sklearn import metrics
# from train_utils.focalloss import FocalLoss
from my_dataset import MyDataset_2dimage_clin, MyDataset_2dimage, MyDataset_3modal_roi_multi
from train_utils.multi_focalloss import FocalLoss
# from multi_focalloss import FocalLoss
import warnings
from scipy import interp

warnings.filterwarnings('ignore')


def read_train_data(root: str, val_rate: float = 0.0):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    # print("{} images for validation.".format(len(val_images_path)))
    # assert len(train_images_path) > 0, "number of training images must greater than 0."
    # assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label


def read_val_data(root: str, val_rate: float = 0.0):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images for valing.".format(len(train_images_path)))
    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


class Focal_ce_loss(nn.Module):
    def __init__(self):
        super(Focal_ce_loss, self).__init__()

    def forward(self, pred, labels):
        alpha = 0.25
        loss_1 = FocalLoss(alpha=alpha)
        loss1 = loss_1(pred, labels)
        loss_2 = torch.nn.CrossEntropyLoss()
        loss2 = loss_2(pred, labels)
        loss = 0.6 * loss2 + 0.4 * loss1
        return loss


def bootstrap_auc(y_true, y_pred, n_bootstraps=2000, rng_seed=42):
    n_bootstraps = n_bootstraps
    rng_seed = rng_seed
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        indices = rng.randint(len(y_pred), size=len(y_pred))
        try:
            score = roc_auc_score(y_true[indices], y_pred[indices])
        except ValueError:
            pass
        bootstrapped_scores.append(score)
        # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
    bootstrapped_scores = np.array(bootstrapped_scores)

    print("AUC: {:0.3f}".format(roc_auc_score(y_true, y_pred)))
    print("Confidence interval for the AUC score: [{:0.3f} - {:0.3}]".format(
        np.percentile(bootstrapped_scores, (2.5, 97.5))[0], np.percentile(bootstrapped_scores, (2.5, 97.5))[1]))

    return roc_auc_score(y_true, y_pred), np.percentile(bootstrapped_scores, (2.5, 97.5))


# def focal_ce_loss(pred, labels):
#     alpha = 0.25
#     loss_1 = FocalLoss(alpha=alpha)
#     loss1 = loss_1(pred, labels)
#     loss_2 = torch.nn.CrossEntropyLoss()
#     loss2 = loss_2(pred, labels)
#     loss = 0.7 * loss2 + 0.3 * loss1
#     return loss
# import torch.nn.functional as F
#
# class LabelSmoothingCrossEntropy(nn.Module):
#     def __init__(self, eps=0.1, reduction='mean'):
#         super(LabelSmoothingCrossEntropy, self).__init__()
#         self.eps = eps
#         self.reduction = reduction
#
#     def forward(self, output, target):
#         # CrossEntropyLoss = Softmax–Log–NLLLoss
#         c = output.size()[-1]
#         log_preds = F.log_softmax(output, dim=-1)
#         if self.reduction=='sum':
#             loss = -log_preds.sum()
#         else:
#             loss = -log_preds.sum(dim=-1)
#             if self.reduction=='mean':
#                 loss = loss.mean()
#         return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert 0.0 < smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        """
        写法1
        """
        # logprobs = F.log_softmax(x, dim=-1)
        # nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        # nll_loss = nll_loss.squeeze(1)  # 得到交叉熵损失
        # # 注意这里要结合公式来理解，同时留意预测正确的那个类，也有a/K，其中a为平滑因子，K为类别数
        # smooth_loss = -logprobs.mean(dim=1)
        # loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        """
        写法2
        """
        y_hat = torch.softmax(x, dim=1)
        # 这里cross_loss和nll_loss等价
        cross_loss = self.cross_entropy(y_hat, target)
        smooth_loss = -torch.log(y_hat).mean(dim=1)
        # smooth_loss也可以用下面的方法计算,注意loga + logb = log(ab)
        # smooth_loss = -torch.log(torch.prod(y_hat, dim=1)) / y_hat.shape[1]
        loss = self.confidence * cross_loss + self.smoothing * smooth_loss
        return loss.mean()

    def cross_entropy(self, y_hat, y):
        return - torch.log(y_hat[range(len(y_hat)), y])


def train_one_epoch(args, model, optimizer, data_loader, device, epoch,  modal_num, peoples=None, clinical_feat=None,
                    kfold=None):
    model.train()
    # w = torch.tensor([0.65, 0.35]).to(device)
    from train_utils.loss import GeneralizedCrossEntropy,SCELoss,ReverseCrossEntropy
    if args.loss == 'ce':
        w = torch.tensor([0.25, 0.75]).to(device)
        loss_function = torch.nn.CrossEntropyLoss(weight=w)
    elif args.loss == 'gce':
        loss_function = GeneralizedCrossEntropy(num_classes=2)
    elif args.loss == 'sce':
        loss_function = SCELoss(num_classes=2)
    elif args.loss == 'rce':
        loss_function = ReverseCrossEntropy(num_classes=2)
    elif args.loss == 'nor_ce':
        loss_function = torch.nn.CrossEntropyLoss()
    elif args.loss == '37_ce':
        w = torch.tensor([0.3, 0.7]).to(device)
        loss_function = torch.nn.CrossEntropyLoss(weight=w)


    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()
    sample_num = 0
    # print(data_loader)
    data_loader = tqdm(data_loader, file=sys.stdout)
    if modal_num == 2:
        for step, data in enumerate(data_loader):
            images, labels, name = data
            if args.clinical == True:
                # images, labels, name = data
                clinical_features = [clinical_feat[peoples.index(i)] for i in name]
                pred = model(images.to(device),
                             torch.from_numpy(np.array(clinical_features, dtype=np.float32)).to(device))
                # pred = model(torch.from_numpy(np.array(clinical_features, dtype=np.float32)).to(device))
            if args.clinical == False:
                # images, labels = data
                pred = model(images.to(device))

            sample_num += images.shape[0]
            # sample_num += len(images)

            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels.to(device)).sum()
            # loss = focal_ce_loss(pred, labels.to(device))
            loss = loss_function(pred, labels.to(device))
            loss.backward()
            accu_loss += loss.detach()
            if kfold != None:
                data_loader.desc = "[fold {}][train epoch {}] loss: {:.3f}, acc: {:.3f}".format(kfold, epoch,
                                                                                                accu_loss.item() / (
                                                                                                        step + 1),

                                                                                                accu_num.item() / sample_num)
            else:
                data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                       accu_loss.item() / (step + 1),
                                                                                       accu_num.item() / sample_num)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)

            optimizer.step()
            optimizer.zero_grad()

        return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(args,model, data_loader, device, epoch, modal_num,  peoples=None, clinical_feat=None, kfold=None):
    # w = torch.tensor([0.4, 0.6]).to(device)
    # w = torch.tensor([0.65, 0.35]).to(device)
    from train_utils.loss import GeneralizedCrossEntropy,SCELoss,ReverseCrossEntropy
    if args.loss == 'ce':
        w = torch.tensor([0.3, 0.7]).to(device)
        loss_function = torch.nn.CrossEntropyLoss(weight=w)
    elif args.loss == 'gce':
        loss_function = GeneralizedCrossEntropy(num_classes=2)
    elif args.loss == 'sce':
        loss_function = SCELoss(num_classes=2)
    elif args.loss == 'rce':
        loss_function = ReverseCrossEntropy(num_classes=2)
    # loss_function = FocalLoss(alpha=alpha)
    # loss_function = Focal_ce_loss()
    model.eval()
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0

    if modal_num == 2:
        data_loader = tqdm(data_loader, file=sys.stdout)
        for step, data in enumerate(data_loader):
            images, labels, name = data
            if args.clinical == True:
                # images, labels, name = data
                clinical_features = [clinical_feat[peoples.index(i)] for i in name]
                pred = model(images.to(device),
                             torch.from_numpy(np.array(clinical_features, dtype=np.float32)).to(device))
                # pred = model(torch.from_numpy(np.array(clinical_features, dtype=np.float32)).to(device))
            if args.clinical == False:
                # images, labels = data
                pred = model(images.to(device))

            sample_num += images.shape[0]
            # sample_num += len(images)
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels.to(device)).sum()
            loss = loss_function(pred, labels.to(device))
            # loss = focal_ce_loss(pred, labels.to(device))
            accu_loss += loss
            if kfold != None:
                data_loader.desc = "[fold {}][valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(kfold, epoch,
                                                                                                accu_loss.item() / (
                                                                                                            step + 1),
                                                                                                accu_num.item() / sample_num)
            else:
                data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                       accu_loss.item() / (step + 1),
                                                                                       accu_num.item() / sample_num)

        return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def metric_test_slice(args, exp_name, i, model, model_weight, clinical, data_loader, device, tprs, aucs,metric_output,metric_labels,peoples=None,
                      clinical_feat=None, external=None, ):
    # read class_indict

    if external == True:
        exp_name_save = os.path.join(exp_name, 'external_metrtic', 'slice')
    else:
        exp_name_save = os.path.join(exp_name, 'metrtic', 'slice')
    os.makedirs(exp_name_save, exist_ok=True)
    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=args.num_classes, labels=labels,
                                save_path=os.path.join(exp_name_save, f'fold{i}_metric.png'),
                                show_plt=args.show_plt)

    model.load_state_dict(torch.load(model_weight, map_location=device))
    model.to(device)
    model.eval()
    predict_data = []
    y_true = []
    output_lable = []
    mean_fpr = np.linspace(0, 1, 100)
    # fig, ax = plt.subplots()
    with torch.no_grad():
        test_loader = tqdm(data_loader, file=sys.stdout)

        for step, data in enumerate(test_loader):
            images, labels, name = data
            if clinical == True:
                # images, labels, name = data
                clinical_features = [clinical_feat[peoples.index(i)] for i in name]
                outputs = model(images.to(device),
                                torch.from_numpy(np.array(clinical_features, dtype=np.float32)).to(device))
                # outputs = model(torch.from_numpy(np.array(clinical_features, dtype=np.float32)).to(device))
            if clinical == False:
                # images, labels = data
                outputs = model(images.to(device))
            true_labels = labels.detach()
            # true_labels = torch.squeeze(true_labels)
            # 获取真实标签 以追加的方式
            y_true.extend(true_labels.cpu().numpy())
            # 获取模型输出结果
            # outputs = model(image_dwi, image_t2)
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
            # outputs = torch.argmax(outputs, dim=1)
            outputs_1 = outputs[:, 1]
            outputs = (outputs_1 > 0.323).long()

            output_lable.extend(outputs.cpu().numpy())
            # a = outputs.to("cpu").numpy()
            # b = labels.to("cpu").numpy()
            # print('a')
            metric_output = np.concatenate((metric_output, outputs.to("cpu").numpy()))
            metric_labels = np.concatenate((metric_labels, labels.to("cpu").numpy()))
            confusion.update(outputs.to("cpu").numpy(), labels.to("cpu").numpy())

    confusion.plot()
    acc, Precision, recall, specificity, F1_score = confusion.summary()
    # print('预测标签', output_lable)
    # print('预测概率', predict_data)

    predict_probs = np.asarray(predict_data)
    # 获取正类的概率
    y_score = predict_probs[:, 1]
    # 真实标签
    y_true = np.asarray(y_true)

    roc_auc, ci = bootstrap_auc(np.array(y_true), np.array(y_score))

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    # roc_auc = metrics.auc(fpr, tpr)  # 计算auc的值

    print('ROC ：', roc_auc, "CI : ", ci)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    aucs.append(roc_auc)
    # 绘制roc曲线
    # plt.figure()
    lw = 2
    # plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='b', lw=lw, label='LR ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='r', lw=lw, linestyle='--', alpha=.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(exp_name_save, f'fold{i}_auc.png'))
    if args.show_plt:
        plt.show()
    else:
        plt.close()
    metric = [i, acc, Precision, recall, specificity, F1_score, roc_auc, ci[0], ci[1]]
    metric = [round(x, 3) for x in metric]
    metric_output = metric_output.astype(np.int64)
    metric_labels = metric_labels.astype(np.int64)
    # metric.append(ci)
    return metric, mean_fpr, tprs, aucs, metric_output, metric_labels


@torch.no_grad()
def metric_test_case(args, exp_name, external_datasets_name, i, model, data_fold, data_transform, model_weight, device, tprs, aucs,metric_output,metric_labels,best_cutoff,
                     peoples=None, clinical_feat=None, external=None):
    # read class_indict
    mean_fpr = np.linspace(0, 1, 100)
    json_label_path = './class_indices.json'
    if external == True:
        exp_name_save = os.path.join(exp_name, 'external_metrtic_'+str(external_datasets_name), 'case')
    if external == False:
        exp_name_save = os.path.join(exp_name, 'metrtic', 'case')
    os.makedirs(exp_name_save, exist_ok=True)
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)
    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=args.num_classes, labels=labels,
                                save_path=os.path.join(exp_name_save, f'fold{i}_metric.png'),
                                show_plt=args.show_plt)
    # 加载模型
    model.load_state_dict(torch.load(model_weight, map_location=device))
    model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    for fold_index, k in enumerate(data_fold):  # data_fold是一个列表 ['./datasets_30_suzhou/diease_folder/N152845,0',  ...]
        fold = k[:-2]
        label = k[-1:]
        datalist = [os.path.join(fold, data) + ',' + label for data in os.listdir(fold)]  # 一个folder里的所有文件
        # print(datalist)
        # if args.clinical == True:
        # test_dataset = MyDataset_2dimage_clin(datalist, transform=data_transform['val'])
        test_dataset = MyDataset_3modal_roi_multi(datalist, transform=data_transform['val'])

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(datalist),
                                                  num_workers=args.nw)  # 加载一个文件夹爱里所有的文件\
        # if args.clinical == False:
        #     test_dataset = MyDataset_2dimage(datalist, transform=data_transform['val'])
        #     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(datalist),
        #                                               num_workers=args.nw)  # 加载一个文件夹爱里所有的文件\

        predict_data_slice = []
        y_true_slice = []

        with torch.no_grad():
            # test_loader = tqdm(test_loader, file=sys.stdout)
            for step, data in enumerate(test_loader):
                images, labels, name = data
                if args.clinical == True:
                    # images, labels, name = data
                    clinical_features = [clinical_feat[peoples.index(i)] for i in name]
                    y_true_slice.append(labels.cpu().numpy())
                    outputs = model(images.to(device),
                                    torch.from_numpy(np.array(clinical_features, dtype=np.float32)).to(device))
                if args.clinical == False:
                    # images, labels = data
                    y_true_slice.append(labels.cpu().numpy())
                    outputs = model(images.to(device))
                outputs = torch.softmax(outputs, dim=1)
                output = outputs.float().squeeze().cpu().numpy()  #
                predict_data_slice.append(output)

            predict_data_slice = np.mean(np.asarray(predict_data_slice), axis=0)
            # print(predict_data_slice.shape)
            # print(len(predict_data_slice))

            # ### 取top3概率求平均
            if predict_data_slice.shape == (2,):
                sorted_data = list(predict_data_slice)
                predict_top_three_rows = sorted_data
                predict_case_prob = predict_top_three_rows
                predict_case_class = np.argmax(predict_case_prob)
                predict_case_class = 1 if predict_case_prob[1] >= best_cutoff else 0
                y_pred.append(predict_case_prob)


            elif len(predict_data_slice) > 2:
                sorted_data = sorted(predict_data_slice, key=lambda row: max(row), reverse=True)
                predict_top_three_rows = sorted_data[:3]
                predict_case_prob = np.mean(predict_top_three_rows, axis=0)
                predict_case_class = 1 if predict_case_prob[1] >= best_cutoff else 0
                y_pred.append(predict_case_prob.tolist())

            else:
                sorted_data = list(predict_data_slice)
                predict_top_three_rows = sorted_data
                predict_case_prob = np.mean(predict_top_three_rows, axis=0)
                predict_case_class = 1 if predict_case_prob[1] >= best_cutoff else 0
                y_pred.append(predict_case_prob.tolist())


            y_true_case = y_true_slice[0][0]

            y_true.append(y_true_case)
            # print(y_true_case)
            # 病例级别的预测概率  预测类别 真实类别
            # print(predict_case_prob, predict_case_class, y_true_case)
            prdeict_case = np.array(predict_case_class).reshape((1,))

            true_case = np.array(y_true_case).reshape((1,))
            metric_output = np.concatenate((metric_output, prdeict_case))
            metric_labels = np.concatenate((metric_labels, true_case))
            confusion.update(prdeict_case, true_case)
        # name是一个里面元素都一样的tuple 取其中的一个当作name
        # if len(predict_data_slice) > 1:
        # print(len(name))
        if len(name) != 1:
            predict_name = [name[0], y_true_case, predict_case_class, predict_case_prob, round(predict_case_prob[0], 3),
                            round(predict_case_prob[1], 3)]
        else:
            print('a')
            predict_name = [name, y_true_case, predict_case_class, predict_case_prob, round(predict_case_prob[0], 3),
                        round(predict_case_prob[1], 3)]


        predict_case_name = os.path.join(exp_name_save, 'predict_test_name.csv')
        head_names = ["name", "true_label", "predict_label", "predict_prob", "predict_0_prob", "predict_1_prob"]
        with open(predict_case_name, mode='a', newline='') as csv_file:
            if i==1 and fold_index + 1 == 1:
                writer = csv.DictWriter(csv_file, fieldnames=head_names)
                writer.writeheader()
                writer2 = csv.writer(csv_file)
                writer2.writerow(predict_name)
            else:
                writer = csv.writer(csv_file)
                writer.writerow(predict_name)
    ## 绘制混淆矩阵
    confusion.plot()
    acc, Precision, recall, specificity, F1_score = confusion.summary()
    # print('a')
    y_pred = np.array(y_pred)
    # 获取正类的概率
    y_score = y_pred[:, 1]

    roc_auc, ci = bootstrap_auc(np.array(y_true), np.array(y_score))
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)  # 输入ndarray

    # roc_auc = metrics.auc(fpr, tpr)  # 计算auc的值
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    aucs.append(roc_auc)
    print('AUC ：', roc_auc, 'CI:', ci)
    # 绘制roc曲线
    # plt.figure()
    lw = 2
    # plt.figure(figsize=(10, 10))
    # plt.plot(fpr, tpr, color='b', lw=lw, label='LR ROC curve (area = %0.3f)' % roc_auc)
    plt.plot(fpr, tpr, color='b', lw=lw, label='Fusion Model(AUC = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='k', lw=lw, linestyle='--', alpha=.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(exp_name_save, f'fold{i}_auc.png'))
    plt.close()

    metric = [i, acc, Precision, recall, specificity, F1_score, roc_auc, ci[0], ci[1]]
    metric = [round(x, 3) for x in metric]

    metric_output = metric_output.astype(np.int64)
    metric_labels = metric_labels.astype(np.int64)
    return metric, mean_fpr, tprs, aucs, metric_output, metric_labels


# 绘制gradcam图
import cv2
import numpy as np


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compatibility with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


class GradCAM:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 use_cuda=False):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=(2, 3), keepdims=True)

    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)
        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)
        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])
        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)
        return result

    def __call__(self, input_tensor, target_category=None):
        if self.cuda:
            input_tensor = input_tensor.cuda()
        # 正向传播得到网络输出logits(未经过softmax)
        output = self.activations_and_grads(input_tensor)
        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)
        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"category id: {target_category}")
        else:
            assert (len(target_category) == input_tensor.size(0))
        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)
        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")
    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def center_crop_img(img: np.ndarray, size: int):
    h, w, c = img.shape
    if w == h == size:
        return img
    if w < h:
        ratio = size / w
        new_w = size
        new_h = int(h * ratio)
    else:
        ratio = size / h
        new_h = size
        new_w = int(w * ratio)
    img = cv2.resize(img, dsize=(new_w, new_h))
    if new_w == size:
        h = (new_h - size) // 2
        img = img[h: h + size]
    else:
        w = (new_w - size) // 2
        img = img[:, w: w + size]
    return img


# 混淆矩阵绘制
class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """

    def __init__(self, num_classes: int, labels: list, save_path: str, show_plt: bool):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels
        self.save_path = save_path
        self.show_plt = show_plt

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "specificity", "F1_score"]
        i = 1
        TP = self.matrix[i, i]
        FP = np.sum(self.matrix[i, :]) - TP
        FN = np.sum(self.matrix[:, i]) - TP
        TN = np.sum(self.matrix) - TP - FP - FN
        Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
        sensitivity = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
        # Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
        specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
        F1_score = 2 * (Precision * sensitivity) / (Precision + sensitivity)
        table.add_row([self.labels[i], Precision, sensitivity, specificity, F1_score])
        print(table)
        return acc, Precision, sensitivity, specificity, F1_score

    def plot(self):
        matrix = self.matrix
        save_path = self.save_path
        show_plt = self.show_plt
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max()
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                # print(info, thresh)
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > 0.8 * thresh else "black")
        plt.tight_layout()
        plt.savefig(save_path)
        if show_plt:
            plt.show()
        else:
            plt.close()
        # plt.show()


class Logger(object):
    def __init__(self, filename=None, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class Logger2(object):
    def __init__(self, filename=None, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def __exit__(self, *args):
        self.close()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        if self.terminal != None:
            sys.stdout = self.terminal
            self.terminal = None

        if self.log != None:
            self.log.close()
            self.log = None


# import sys, os
#
# class Logger(object):
#     "Lumberjack class - duplicates sys.stdout to a log file and it's okay"
#     #source: https://stackoverflow.com/q/616645
#     def __init__(self, filename='', mode="w", buff=0):
#         self.stdout = sys.stdout
#         self.file = open(filename, mode, buff)
#         sys.stdout = self
#
#     def __del__(self):
#         self.close()
#
#     def __enter__(self):
#         pass
#
#     def __exit__(self, *args):
#         self.close()
#
#     def write(self, message):
#         self.stdout.write(message)
#         self.file.write(message)
#
#     def flush(self):
#         self.stdout.flush()
#         self.file.flush()
#         os.fsync(self.file.fileno())
#
#     def close(self):
#         if self.stdout != None:
#             sys.stdout = self.stdout
#             self.stdout = None
#
#         if self.file != None:
#             self.file.close()
#             self.file = None


# 早听
class EarlyStopping(object):
    """Early stops the training if performance doesn't improve after a given patience."""

    def __init__(self, patience=10, verbose=True, delta=0, monitor='val_loss', op_type='min'):
        """
        Args:
            patience (int): How long to wait after last time performance improved.
                            Default: 10
            verbose (bool): If True, prints a message for each performance improvement.
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            monitor (str): Monitored variable.
                            Default: 'val_loss'
            op_type (str): 'min' or 'max'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.monitor = monitor
        self.op_type = op_type

        if self.op_type == 'min':
            self.val_score_min = np.Inf
        else:
            self.val_score_min = 0

    def __call__(self, val_score):

        score = -val_score if self.op_type == 'min' else val_score

        if self.best_score is None:
            self.best_score = score
            self.print_and_update(val_score)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.print_and_update(val_score)
            self.counter = 0

    def print_and_update(self, val_score):
        '''print_message when validation score decrease.'''
        if self.verbose:
            print(self.monitor, f'optimized ({self.val_score_min:.6f} --> {val_score:.6f}).  Saving model ...')
        self.val_score_min = val_score


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label
