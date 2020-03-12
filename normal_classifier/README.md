# 通常のCIFAR10の学習

不均衡にしたCIFAR10をResNet18で学習

### Usage

```
$ python train.py
```

### Accuracy
CIFAR10の訓練用画像 50000枚（各クラス5000枚）を使って学習し、10000枚（各クラス1000枚）のテスト画像を使ってAccuracyの測定をしています。Optimizerには Momentum SGD (momentum=0.9, weight_decay=0.0005) を使用しています。


| Epochs            | BatchSize   |  Accuracy   |
| ----------------- | ----------- | ----------- |
| 50                | 128         |  92.13%     |

LearningRateは以下のように減衰

```
lr = 0.1 if epoch < 20
lr = 0.01 if 20 <= epoch < 30
lr = 0.001 if 30 <= epoch < 50
```

