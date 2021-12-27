from typing import *

import cv2
import json

import numpy as np
from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset


class ImageList(Dataset):

    def __init__(self,
                 root: Path,
                 fileList: Path,
                 transform=None
                 ) -> None:

        self.root: Path = root
        self.imgList: Path = self.read_list(fileList)
        self.transform = transform

    @staticmethod
    def read_list(fileList):
        imgList = []
        with open(fileList, 'r') as file:
            for line in file.readlines():
                imgPath, label = line.strip().split(' ')
                imgList.append((imgPath, int(label)))
        return imgList

    def __getitem__(self, index):

        # get target image and its label
        imgPath, target = self.imgList[index]

        # read image
        img = Image.open(str(self.root.joinpath(imgPath))).convert("L")
        img = self.transform(img) if self.transform is not None else img

        # return result
        return img, target

    def __len__(self):
        return len(self.imgList)


class TrainingHelper:

    class Logger():

        def __init__(self) -> None:
            self.epochs = 0
            self.train_losses: List = []
            self.train_accuracies: List = []
            self.valid_losses: List = []
            self.valid_accuracies: List = []

        def push(self, train_loss: float, train_acc: float, valid_loss: float, valid_acc: float) -> None:
            self.epochs += 1
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.valid_losses.append(valid_loss)
            self.valid_accuracies.append(valid_acc)

        def visualize(self,
                      init_epoch=0,
                      loss_dst: str = None,        # save loss figure to this path
                      accuracy_dst: str = None     # save accuracy figure to this path
                      ) -> None:

            # set x points
            pts = list(range(init_epoch, self.epochs))

            # draw loss graph
            loss_fig = plt.figure(1)
            plt.plot(pts, self.train_losses, pts, self.valid_losses)
            plt.title('Training & Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(['Training', 'Validation'])
            plt.show()

            if loss_dst is not None:
                loss_fig.savefig(loss_dst)

            # draw accuracy graph
            acc_fig = plt.figure(2)
            plt.plot(pts, self.train_accuracies, pts, self.valid_accuracies)
            plt.title('Training & Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend(['Training', 'Validation'])
            plt.show()

            if accuracy_dst is not None:
                acc_fig.savefig(accuracy_dst)

        def save(self, dst: str):
            with open(dst, 'w') as out:
                train_log = {
                    'train_losses': self.train_losses,
                    "train_accuracies": self.train_accuracies,
                    "valid_losses": self.valid_losses,
                    "valid_accuracies": self.valid_accuracies
                }
                json.dump(train_log, out, ensure_ascii=False, indent=2)

    class AverageMeter(object):
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
                 optimizer,

                 class_mapping: dict = None,
                 ) -> None:

        # train and valid data
        self.TRAIN_DATALOADER = train_dataloader
        self.VALID_DATALOADER = valid_dataloader

        # assign hyper-parameteres
        self.EPOCHS = epochs
        self.BS = batch_size
        self.LR = learning_rate

        self.MODEL = model.cuda() if torch.cuda.is_available() else model
        self.CRITERION = criterion
        self.OPTIMIZER = optimizer

        self.PRINT_FREQUENCY = print_frequency

        # optional
        # class label mapping
        self.CLASS_MAPPING = class_mapping

        self.LOGGER = TrainingHelper.Logger()

    @staticmethod
    def accuracy(model_output, labels):
        # get pred results of each batch (dim 1)
        pred = torch.argmax(model_output, 1)

        # compare results and targets to calc num of correct
        accuracy = pred.eq(labels).float().sum(0) / labels.size(0)
        return accuracy

    @staticmethod
    def progress(iEpoch: int,
                 iBatch: int,
                 nBatch: int,
                 losses: AverageMeter,
                 accuracies: AverageMeter,
                 mode: str
                 ) -> None:

        # epoch header
        if mode == 'train' and iBatch == 0:
            print(f'Epoch [{iEpoch}]')
            print(f'  ├ Training')

        # epoch body
        if mode == 'train':
            print(f'  │    [{iBatch}/{nBatch}]')
            print(f'  │    ├ Loss {losses.val:.3f} ({losses.avg:.4f})')
            print(f'  │    └ Acc  {accuracies.val:.3f} ({accuracies.avg:.4f})')

        # epoch tail
        elif mode == 'valid':
            print(f'  └ Validation')
            print(f'       ├ Avg Loss {losses.avg:.4f}')
            print(f'       └ Avg Acc  {accuracies.avg:.4f}')

        # test
        elif mode == 'test':
            print(f'Testing')
            print(f'   ├ Avg Loss {losses.avg:.4f}')
            print(f'   └ Avg Acc  {accuracies.avg:.4f}')

        else:
            raise RuntimeError("Wrong Mode")

    def train(self, iEpoch: int):
        """
        """

        losses = TrainingHelper.AverageMeter()
        accuracies = TrainingHelper.AverageMeter()

        self.MODEL.train()
        for iBatch, (image, target) in enumerate(self.TRAIN_DATALOADER):

            # if using cuda
            if torch.cuda.is_available():
                image = image.cuda()
                target = target.cuda()

            # compute output (return fc2, fc1 result)
            output, _ = self.MODEL(image)
            loss = self.CRITERION(output, target)

            # measure accuracy and record loss
            acc = self.accuracy(output.data, target)

            losses.update(loss.item(), image.size(0))
            accuracies.update(acc.item(), image.size(0))

            # compute gradient and do optimize step
            self.OPTIMIZER.zero_grad()
            loss.backward()
            self.OPTIMIZER.step()

            # print current training info
            if iBatch % self.PRINT_FREQUENCY == 0:
                self.progress(
                    iEpoch=iEpoch,
                    iBatch=iBatch,
                    nBatch=len(self.TRAIN_DATALOADER),
                    losses=losses,
                    accuracies=accuracies,
                    mode='train'
                )

        return losses.avg, accuracies.avg

    def validate(self, iEpoch: int, mode='valid'):
        """

        """
        losses = TrainingHelper.AverageMeter()
        accuracies = TrainingHelper.AverageMeter()

        # switch to evaluate mode
        self.MODEL.eval()
        with torch.no_grad():
            for image, target in self.VALID_DATALOADER:

                # if using cuda
                if torch.cuda.is_available():
                    image = image.cuda()
                    target = target.cuda()

                # compute output (output fc2, fc1 result)
                output, _ = self.MODEL(image)
                loss = self.CRITERION(output, target)

                # measure accuracy and record loss
                acc = self.accuracy(output.data, target)

                losses.update(loss.item(), image.size(0))
                accuracies.update(acc.item(), image.size(0))

        self.progress(
            iEpoch=iEpoch,
            iBatch=0,
            nBatch=len(self.VALID_DATALOADER),
            losses=losses,
            accuracies=accuracies,
            mode=mode,
        )

        return losses.avg, accuracies.avg
