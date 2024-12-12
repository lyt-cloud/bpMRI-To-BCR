import os
import random


def train_val_split_data(data, train_ratio, val_ratio, test_ratio=0, no_test=True, random_seed=None):
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
    # 设置随机种子
    if random_seed is not None:
        random.seed(random_seed)
    # 分离 '1' 和 '0' 的样本
    # samples_0 = [item for item in data if item[-1] == '0']
    # samples_1 = [item for item in data if item[-1] == '1']
    samples_0 = [item for item in data if item.endswith(',0')]
    samples_1 = [item for item in data if item.endswith(',1')]

    # 计算每个类别应该有多少个样本
    train_samples_0 = int(len(samples_0) * train_ratio)
    val_samples_0 = int(len(samples_0) - train_samples_0)
    # val_samples_0 = int(len(samples_0) * val_ratio)
    train_samples_1 = int(len(samples_1) * train_ratio)
    val_samples_1 = int(len(samples_1) - train_samples_1)
    # val_samples_1 = int(len(samples_1) * val_ratio)

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

def whole_data_path2(data_path, fold_path):
    """
    返回文件夹路径
    """
    data = []
    for i in data_path:
        fold_name = i[:-2]  # dwi是-1 t2wi是-2
        label = i[-1:]  # 0，1

        fold_name = os.path.join(fold_path, fold_name)
        data.append(fold_name + "," + label)

        # data_l = [os.path.join(fold_name, j) + "," + label for j in os.listdir(fold_name)]
        # for j in data_l:
        #     data.append(j)
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

