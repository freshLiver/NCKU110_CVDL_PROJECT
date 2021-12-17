import random
import math
from os import system
from pathlib import Path

from numpy import true_divide

from light_cnn import LightCNN_9Layers
from functional import TrainingHelper, ImageList

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# training
EPOCHS = 5
BS = 128             # batch size
LR = 0.0001          # learning rate
NUM_WORKER = 0
PRINT_FREQUENCY = 10
VALID_RATIO = 0.2


# give me abs path
ROOT = Path("/tmp")
TRAIN_LIST = Path.joinpath(ROOT, "train_list.txt")
VALID_LIST = Path.joinpath(ROOT, "valid_list.txt")


def get_dataset(tick_size: int = 100) -> int:
    # download dataset
    system(f"wget -cP {ROOT} http://vis-www.cs.umass.edu/lfw/lfw.tgz")

    # extract if dir not exists
    dataset_dir = Path(f'{ROOT}/lfw')
    if not dataset_dir.exists():
        system(f"tar -xf {ROOT}/lfw.tgz --directory {ROOT}")

    # load data into dict
    dataset = []

    NUM_CLASSES = 0
    for index, catdir in enumerate(dataset_dir.glob("*")):
        # iterate each file in this dir
        for pth in catdir.glob("*"):
            dataset.append((pth, index))
        NUM_CLASSES += 1

        # TODO trick
        if NUM_CLASSES == tick_size:
            break

    random.shuffle(dataset)

    # split dataset
    train_size = math.floor(len(dataset) * (1-VALID_RATIO))
    valid_size = len(dataset) - train_size

    # save to file
    with open(TRAIN_LIST, 'w') as f:
        for img, cat in dataset[:train_size]:
            f.write(f'{img} {cat}\n')

    with open(VALID_LIST, 'w') as f:
        for img, cat in dataset[train_size:]:
            f.write(f'{img} {cat}\n')

    return NUM_CLASSES


if __name__ == "__main__":

    NUM_CLASSES = get_dataset(100)

    from torchvision.models.vgg import vgg16
    model = vgg16(pretrained=False, num_classes=NUM_CLASSES)

    # large lr for last fc parameters
    if torch.cuda.is_available():
        model = model.cuda()

    # define transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # create data loaders
    train_loader = DataLoader(
        ImageList(root=ROOT, fileList=TRAIN_LIST, transform=train_transform),
        batch_size=BS,
        shuffle=True,
        num_workers=NUM_WORKER,
        pin_memory=True
    )

    valid_loader = DataLoader(
        ImageList(root=ROOT, fileList=VALID_LIST, transform=valid_transform),
        batch_size=BS,
        shuffle=False,
        num_workers=NUM_WORKER,
        pin_memory=True
    )

    helper = TrainingHelper(
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        epochs=EPOCHS,
        batch_size=BS,
        learning_rate=LR,
        print_frequency=PRINT_FREQUENCY,
        model=model,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(params=model.parameters(), lr=LR)
    )

    cudnn.benchmark = True

    helper.validate()
    for epoch in range(0, EPOCHS):

        helper.adjust_learning_rate(epoch)

        # train for one epoch
        helper.train(epoch)

        # evaluate on validation set
        prec1 = helper.validate()
