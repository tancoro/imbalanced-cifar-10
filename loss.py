import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def print_x(pref, x):
    print(pref)
    n = x.detach().cpu().numpy()
    print(n)
    print(n.shape)


class BasicCrossEntropyLoss():
    def __init__(self):
        pass

    def __call__(self, x, labels):
        # presetのcross_entropyloss
        # x = x.log()
        # x = F.nll_loss(x, labels, reduction='mean')

        # softmaxの結果Pのlogを取得
        x = x.log()
        # labelのインデックスに対応する値を取得
        labels_view = labels.view(-1, 1)
        x = x.gather(1, labels_view).view(-1)
        # -log(P)
        x = (-1) * x.view(-1)
        # 平均
        x = x.sum() / len(x)

        return x

class _InverseClassFrequency():
    def __init__(self, data_count_map, device):
        self._prepared_loss_weight(data_count_map, device)

    def _prepared_loss_weight(self, data_count_map, device):
        data_list = np.array([k[1] for k in sorted(data_count_map.items(), key=lambda x: x[0])])
        print(data_list)
        data_list = 1.0 / data_list # 0 があるとエラー
        print(data_list)
        self.loss_weight = torch.from_numpy(data_list).float().to(device)
        print(self.loss_weight)

class _ClassBalanced():
    def __init__(self, data_count_map, device, beta=0.99):
        self._prepared_loss_weight(data_count_map, device, beta)

    def _prepared_loss_weight(self, data_count_map, device, beta):
        data_list = np.array([k[1] for k in sorted(data_count_map.items(), key=lambda x: x[0])])
        print(data_list)
        data_list = (1 - beta)/(1.0 - (beta ** data_list))
        print(data_list)
        self.loss_weight = torch.from_numpy(data_list).float().to(device)
        print(self.loss_weight)


class CrossEntropyLoss():
    def __init__(self, data_count_map, device):
        print('len {}'.format(len(data_count_map)))
        self.loss_weight = torch.ones(len(data_count_map)).to(device)

    def __call__(self, x, labels):
        # softmaxの結果Pのlogを取得
        x = x.log()
        # labelのインデックスに対応する値を取得
        labels_view = labels.view(-1, 1)
        x = x.gather(1, labels_view).view(-1)
        # log(P)と加重平均をとるための重みベクトル
        # print_x('labels_view.view(-1)', labels_view.view(-1))
        weighted_vec = self.loss_weight.gather(0, labels_view.view(-1))
        # 総和を1にしておく
        weighted_vec = weighted_vec / weighted_vec.sum()
        # print_x('weighted_vec', weighted_vec)
        # weighted_vec = torch.ones(len(labels)).to(device)
        # -log(P) と weighted_vec の内積を取る
        x = ((-1) * x.view(-1) * weighted_vec).sum()

        return x

class FocalLoss():
    def __init__(self, data_count_map, device, gamma=2.0):
        print('len {}'.format(len(data_count_map)))
        self.loss_weight = torch.ones(len(data_count_map)).to(device)
        self.gamma = gamma

    def __call__(self, x, labels):
        if self.gamma < 1:
            x = x.clamp(max=1. -1e-7)
        # labelのインデックスに対応する値を取得
        labels_view = labels.view(-1, 1)
        x = x.gather(1, labels_view).view(-1)
        # log(P)と加重平均をとるための重みベクトル
        # print_x('labels_view.view(-1)', labels_view.view(-1))
        weighted_vec = self.loss_weight.gather(0, labels_view.view(-1))
        # 総和を1にしておく
        weighted_vec = weighted_vec / weighted_vec.sum()
        # print_x('weighted_vec', weighted_vec)
        # weighted_vec = torch.ones(len(labels)).to(device)
        # -log(P) と weighted_vec の内積を取る
        f = torch.pow((1 - x), self.gamma)
        # if torch.isnan(f).sum() > 0:
        #    exit()
        x = ((-1) * weighted_vec * f * x.log()).sum()

        return x

class InverseClassFrequencyCrossEntropyLoss(_InverseClassFrequency, CrossEntropyLoss):
    pass

class InverseClassFrequencyFocalLoss(_InverseClassFrequency, FocalLoss):
    def __init__(self, data_count_map, device, gamma=2.0):
        super().__init__(data_count_map, device)
        self.gamma = gamma

class ClassBalancedCrossEntropyLoss(_ClassBalanced, CrossEntropyLoss):
    pass

class ClassBalancedFocalLoss(_ClassBalanced, FocalLoss):
    def __init__(self, data_count_map, device, beta=0.99, gamma=2.0):
        super().__init__(data_count_map, device, beta)
        self.gamma = gamma


if __name__ == '__main__':
    icf_fl = InverseClassFrequencyFocalLoss({1: 30, 2: 40}, torch.device("cpu"), gamma=3.0)
    print(icf_fl.gamma)
