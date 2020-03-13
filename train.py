# import comet_ml in the top of your file
from comet_ml import Experiment

import numpy as np

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data.sampler import Sampler

from resnet import ResNet18

from loss import print_x
from loss import BasicCrossEntropyLoss
from loss import CrossEntropyLoss
from loss import FocalLoss
from loss import InverseClassFrequencyCrossEntropyLoss
from loss import InverseClassFrequencyFocalLoss
from loss import ClassBalancedCrossEntropyLoss
from loss import ClassBalancedFocalLoss

## (for Mac) OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

class ReductionSampler(Sampler):
    def __init__(self, data_source, sampling_rate={}):
        self.data_source = data_source

        label_hash = {}
        for idx, (img, label) in enumerate(self.data_source):
            if not label in label_hash:
                label_hash[label] = []
            label_hash[label].append(idx)

        self.data_count_map = {}
        self.indices = []
        for k in label_hash.keys():
            if k in sampling_rate:
                label_size = len(label_hash[k])
                sampling_count = int(label_size * sampling_rate[k])
                rand_idx = torch.randint(high=label_size - 1, size=(sampling_count,), dtype=torch.int64).numpy()
                self.indices.extend(np.array(label_hash[k])[rand_idx].tolist())
                self.data_count_map[k] = len(rand_idx)
            else:
                self.indices.extend(label_hash[k])
                self.data_count_map[k] = len(label_hash[k])

    def get_data_count_map(self):
        return self.data_count_map

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(args, model, device, train_loader, data_count, optimizer, epoch, experiment, lossfunc=BasicCrossEntropyLoss()):
    model.train()
    correct = 0
    total = 0
    iter_num = len(train_loader)
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        # forward (softmaxまで)
        output = model(data)
        # cal Loss
        loss = lossfunc(output, labels)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += len(data)
        g_step = iter_num * (epoch - 1) + batch_idx
        experiment.log_metric("loss", loss.item(), step=g_step)
        experiment.log_metric("accuracy", correct / total, step=g_step)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), data_count,
                100. * batch_idx / len(train_loader), loss.item()))

    experiment.log_metric("lr", get_lr(optimizer), step=epoch)

def test(args, model, device, test_loader, data_count, epoch, experiment, pref='', lossfunc=BasicCrossEntropyLoss()):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)

            # forward (softmaxまで)
            output = model(data)

            # nomal
            # test_loss += F.nll_loss(output.log(), labels, reduction='sum').item() # sum up batch loss

            # cal Loss
            test_loss += (lossfunc(output, labels) * len(labels)).item()
            print('test_loss: {}'.format(test_loss))

            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= data_count

    experiment.log_metric(pref + "_loss", test_loss, step=(epoch-1))
    experiment.log_metric(pref + "_accuracy", correct / data_count, step=(epoch-1))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, data_count, 100. * correct / data_count))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Cifar10 Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=25, metavar='N', help='number of epochs to train (default: 25)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--model-path', type=str, default='', metavar='M', help='model param path')
    parser.add_argument('--loss-type', type=str, default='CE', metavar='L', help='B or CE or F or ICF_CE or ICF_F or CB_CE or CB_F')
    parser.add_argument('--beta', type=float, default=0.999, metavar='B', help='Beta for ClassBalancedLoss')
    parser.add_argument('--gamma', type=float, default=2.0, metavar='G', help='Gamma for FocalLoss')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--balanced-data', action='store_true', default=False, help='For sampling rate. Default is Imbalanced-data.')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    args = parser.parse_args()

    # Add the following code anywhere in your machine learning file
    experiment = Experiment(api_key="5Yl3Rxz9S3E0PUKQTBpA0QJPi", project_name="imbalanced-cifar-10", workspace="tancoro")

    # ブラウザの実験ページを開く
    # experiment.display(clear=True, wait=True, new=0, autoraise=True)
    # 実験キー(実験を一意に特定するためのキー)の取得
    exp_key = experiment.get_key()
    print('KEY: ' + exp_key)
    # HyperParamの記録
    hyper_params = {
        'batch_size': args.batch_size,
        'epoch': args.epochs,
        'learning_rate': args.lr,
        'sgd_momentum' : args.momentum,
        'model_path' : args.model_path,
        'loss_type' : args.loss_type,
        'beta' : args.beta,
        'gamma' : args.gamma,
        'torch_manual_seed': args.seed,
        'balanced_data' : args.balanced_data
    }
    experiment.log_parameters(hyper_params)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print('use_cuda {}'.format(use_cuda))

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # train dataset
    cifar10_train_dataset = datasets.CIFAR10('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.RandomCrop(32, padding=4),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ]))
    # train sampling rate
    sampling_rate = {}
    if not args.balanced_data:
        sampling_rate = {1:0.05, 4:0.05, 6:0.05}
    print(sampling_rate)
    # train Sampler
    train_sampler = ReductionSampler(cifar10_train_dataset, sampling_rate=sampling_rate)
    # train loader
    train_loader = torch.utils.data.DataLoader(cifar10_train_dataset,
        batch_size=args.batch_size, sampler=train_sampler, **kwargs)
    # test dataset
    cifar10_test_dataset = datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ]))
    # test majority loader
    test_majority_sampler = ReductionSampler(cifar10_test_dataset, sampling_rate={1:0, 4:0, 6:0})
    test_majority_loader = torch.utils.data.DataLoader(cifar10_test_dataset,
        batch_size=args.test_batch_size, sampler=test_majority_sampler, **kwargs)
    # test minority loader
    test_minority_sampler = ReductionSampler(cifar10_test_dataset, sampling_rate={0:0, 2:0, 3:0, 5:0, 7:0, 8:0, 9:0})
    test_minority_loader = torch.utils.data.DataLoader(cifar10_test_dataset,
            batch_size=args.test_batch_size, sampler=test_minority_sampler, **kwargs)
    # test alldata loader
    test_alldata_loader = torch.utils.data.DataLoader(cifar10_test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = ResNet18().to(device)
    # train loss
    train_loss = BasicCrossEntropyLoss()
    if args.loss_type == 'CE':
        train_loss = CrossEntropyLoss(train_sampler.get_data_count_map(), device)
    elif args.loss_type == 'F':
        train_loss = FocalLoss(train_sampler.get_data_count_map(), device, gamma=args.gamma)
    elif args.loss_type == 'ICF_CE':
        train_loss = InverseClassFrequencyCrossEntropyLoss(train_sampler.get_data_count_map(), device)
    elif args.loss_type == 'ICF_F':
        train_loss = InverseClassFrequencyFocalLoss(train_sampler.get_data_count_map(), device, gamma=args.gamma)
    elif args.loss_type == 'CB_CE':
        train_loss = ClassBalancedCrossEntropyLoss(train_sampler.get_data_count_map(), device, beta=args.beta)
    elif args.loss_type == 'CB_F':
        train_loss = ClassBalancedFocalLoss(train_sampler.get_data_count_map(), device, beta=args.beta, gamma=args.gamma)
    print('Train Loss Type: {}'.format(type(train_loss)))

    # load param
    if len(args.model_path) > 0:
        model.load_state_dict(torch.load(args.model_path))

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    # lr = 0.1 if epoch < 15
    # lr = 0.01 if 15 <= epoch < 20
    # lr = 0.001 if 20 <= epoch < 25
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,20], gamma=0.1)

    for epoch in range(1, args.epochs + 1):
        with experiment.train():
            experiment.log_current_epoch(epoch)
            train(args, model, device, train_loader, len(train_sampler), optimizer, epoch, experiment, lossfunc=train_loss)
        with experiment.test():
            test(args, model, device, test_minority_loader, len(test_minority_sampler), epoch, experiment, pref='minority')
            test(args, model, device, test_majority_loader, len(test_majority_sampler), epoch, experiment, pref='majority')
            test(args, model, device, test_alldata_loader, len(test_alldata_loader.dataset), epoch, experiment, pref='all')
        if (args.save_model) and (epoch % 10 == 0):
            print('saving model to ./model/cifar10_{0}_{1:04d}.pt'.format(exp_key, epoch))
            torch.save(model.state_dict(), "./model/cifar10_{0}_{1:04d}.pt".format(exp_key, epoch))
        scheduler.step()

if __name__ == '__main__':
    main()
