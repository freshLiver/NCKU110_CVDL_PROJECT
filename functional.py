import os
import os.path
import time
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset


def default_loader(path):
    img = Image.open(path)
    return img


def default_list_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, label = line.strip().split(' ')
            imgList.append((imgPath, int(label)))
    return imgList


class ImageList(Dataset):

    def __init__(self,
                 root,
                 fileList,
                 transform=None,
                 list_reader=default_list_reader,
                 loader=default_loader
                 ) -> None:

        self.root = root
        self.imgList = list_reader(fileList)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        imgPath, target = self.imgList[index]
        img = self.loader(os.path.join(self.root, imgPath))

        if self.transform is not None:
            img = self.transform(img)

        print(np.array(img).shape)
        return img, target

    def __len__(self):
        return len(self.imgList)


class TrainingHelper:

    class AverageMeter(object):
        """
        Computes and stores the average and current value
        """

        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            # 在 train, valid 呼叫 update 時設定 val 以供 loss, acc log 使用
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

    # Init Util
    def __init__(self,
                 train_dataloader: DataLoader,
                 valid_dataloader: DataLoader,
                 epochs: int,
                 batch_size: int,
                 learning_rate: float,
                 print_frequency: int,
                 model: torch.nn.Module,
                 criterion,
                 optimizer
                 ) -> None:

        # train and valid data
        self.TRAIN_DATALOADER = train_dataloader
        self.VALID_DATALOADER = valid_dataloader

        # assign hyper-parameteres
        self.EPOCHS = epochs
        self.BS = batch_size
        self.LR = learning_rate

        self.MODEL = model
        self.CRITERION = criterion
        self.OPTIMIZER = optimizer

        self.PRINT_FREQUENCY = print_frequency

    @staticmethod
    def accuracy(output, target, topk=(1,)):
        """
        Computes the precision@k for the specified values of k
        """

        # size of this batch
        batch_size = target.size(0)

        # get pred results of each batch (dim 1)
        pred = torch.argmax(output, 1)

        # compare results and targets to calc num of correct
        correct = pred.eq(target).float()
        acc = correct.sum(0) / batch_size

        return [acc]

    def adjust_learning_rate(self, epoch):
        """
        Adjust Learning Rate while Training
        """
        scale = 0.457305051927326
        step = 10

        lr = self.LR * (scale ** (epoch // step))
        print('lr: {}'.format(lr))
        if (epoch != 0) & (epoch % step == 0):
            print('Change lr')
            for param_group in self.OPTIMIZER.param_groups:
                param_group['lr'] = param_group['lr'] * scale

    def train(self, epoch: int):

        batch_time = TrainingHelper.AverageMeter()
        data_time = TrainingHelper.AverageMeter()
        losses = TrainingHelper.AverageMeter()
        top1 = TrainingHelper.AverageMeter()
        top5 = TrainingHelper.AverageMeter()

        self.MODEL.train()

        end = time.time()
        for i, (input, target) in enumerate(self.TRAIN_DATALOADER):
            data_time.update(time.time() - end)

            # if using cuda
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output, _ = self.MODEL(input_var)
            loss = self.CRITERION(output, target_var)

            # measure accuracy and record loss
            prec1 = self.accuracy(output.data, target)
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))

            # compute gradient and do SGD step
            self.OPTIMIZER.zero_grad()
            loss.backward()
            self.OPTIMIZER.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print current training info
            if i % self.PRINT_FREQUENCY == 0:
                print(
                    f'Epoch: [{epoch}][{i}/{len(self.TRAIN_DATALOADER)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                    f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    f'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                )

    def validate(self):

        losses = TrainingHelper.AverageMeter()
        top1 = TrainingHelper.AverageMeter()
        top5 = TrainingHelper.AverageMeter()

        # switch to evaluate mode
        self.MODEL.eval()

        with torch.no_grad():
            for input, target in self.VALID_DATALOADER:

                # if using cuda
                if torch.cuda.is_available():
                    input = input.cuda()
                    target = target.cuda()

                input_var = torch.autograd.Variable(input)
                target_var = torch.autograd.Variable(target)

                # compute output
                output = self.MODEL(input_var)
                loss = self.CRITERION(output, target_var)

                # measure accuracy and record loss
                prec1 = self.accuracy(output.data, target)
                losses.update(loss.item(), input.size(0))
                top1.update(prec1[0], input.size(0))

        print(f'Test set: AvgLoss: {losses.avg}, Accuracy: ({top1.avg})')

        return top1.avg
