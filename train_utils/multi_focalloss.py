import torch
import torch.nn as nn


class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha=[0.3, 0.15, 0.15, 0.4], gamma=2, reduction='mean'):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1)  # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  # 对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss


# chatgpt 多分类focal loss
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=None, gamma=2, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#
#     def forward(self, inputs, targets):
#         inputs = inputs.view(-1, inputs.size(-1))
#         targets = targets.view(-1, 1)
#
#         log_probs = F.log_softmax(inputs, dim=1)
#         probs = torch.exp(log_probs)
#
#         pt = probs.gather(1, targets)
#
#         focal_weights = torch.pow(1 - pt, self.gamma)
#
#         if self.alpha is not None:
#             alpha_weights = torch.gather(self.alpha, targets)
#             focal_weights = focal_weights * alpha_weights
#
#         focal_loss = -focal_weights * log_probs.gather(1, targets)
#
#         if self.reduction == 'mean':
#             focal_loss = focal_loss.mean()
#         elif self.reduction == 'sum':
#             focal_loss = focal_loss.sum()
#
#         return focal_loss
import torch.nn.functional as F
class FocalLoss(nn.Module):
    """
    Softmax and sigmoid focal loss
    """

    def __init__(self,
                 num_labels=2,
                 gamma=2.0,
                 alpha=0.25,
                 epsilon=1.e-9,
                 reduction='mean',
                 activation_type='softmax',
                 ):

        super(FocalLoss, self).__init__()
        self.num_labels = num_labels
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.activation_type = activation_type
        self.reduction = reduction
    def forward(self, preds, target):
        """
        Args:
            logits: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        if self.activation_type == 'softmax':
            idx = target.view(-1, 1).long()
            one_hot_key = torch.zeros(idx.size(0), self.num_labels, dtype=torch.float32, device=idx.device)
            one_hot_key = one_hot_key.scatter_(1, idx, 1)
            logits = F.softmax(preds, dim=-1)
            loss = -self.alpha * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss = loss.sum(1)
        elif self.activation_type == 'sigmoid':
            multi_hot_key = target
            logits = F.sigmoid(preds)
            zero_hot_key = 1 - multi_hot_key
            loss = -self.alpha * multi_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss += -(1 - self.alpha) * zero_hot_key * torch.pow(logits, self.gamma) * (1 - logits + self.epsilon).log()
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
# class FocalLoss(nn.Module):
#     """
#     参考 https://github.com/lonePatient/TorchBlocks
#     """
#
#     def __init__(self, gamma=2.0, alpha=0.25, epsilon=1.e-9, device=None):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         if isinstance(alpha, list):
#             self.alpha = torch.Tensor(alpha).to(device)
#             # self.alpha = torch.Tensor(alpha)
#         else:
#             self.alpha = alpha
#         self.epsilon = epsilon
#
#     def forward(self, input, target):
#         """
#         Args:
#             input: model's output, shape of [batch_size, num_cls]
#             target: ground truth labels, shape of [batch_size]
#         Returns:
#             shape of [batch_size]
#         """
#         num_labels = input.size(-1)
#         idx = target.view(-1, 1).long()
#         one_hot_key = torch.zeros(idx.size(0), num_labels, dtype=torch.float32, device=idx.device)
#         one_hot_key = one_hot_key.scatter_(1, idx, 1)
#         logits = torch.softmax(input, dim=-1)
#         # a1 = -self.alpha
#         # print(a1)
#         # print("a")
#         loss = -self.alpha * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
#         loss = loss.sum(1)
#         return loss.mean()


if __name__ == '__main__':
    alpha = [0.1, 0.2, 0.3, 0.15, 0.25]
    loss = FocalLoss(alpha=alpha)
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    output = loss(input, target)
    print(output)
    output.backward()
