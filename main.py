import gdown
import math
import random
from os import system
from pathlib import Path


import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from functional import TrainingHelper, ImageList
from light_cnn import LightCNN_9Layers as LightCNN


# ----------------------------------------------------------------

# training
EPOCHS = 5
BS = 128            # batch size
LR = 0.001          # learning rate
NUM_WORKER = 0
PRINT_FREQUENCY = 10
VALID_RATIO = 0.2

LOGGER = SummaryWriter()

# give me abs path as ROOT(work dir)
ROOT = Path("/home/freshliver/Downloads")

# dataset paths
DATA_SRC = "https://drive.google.com/uc?id=11KFpKd8i8r1nES1AmSminorvRivB2M8_"
DATA_DST = ROOT.joinpath("vggface2-test.zip")
DATA_DIR = ROOT.joinpath("vggface2")


DATA_LIST = ROOT.joinpath("list.txt")
TRAIN_LIST = ROOT.joinpath("train_list.txt")
VALID_LIST = ROOT.joinpath("valid_list.txt")

# ! DOWNLOAD DATASET

# # download dataset
# gdown.download(str(DATA_SRC), str(DATA_DST), False)

# # extract if dir not exists
# if not DATA_DIR.exists():
#     system(f"unzip -d {DATA_DIR} {DATA_DST}")

# ----------------------------------------------------------------

# load data into dict
image_list = []

NUM_CLASSES = 0
for index, subdir in enumerate(DATA_DIR.glob("*")):
    # iterate each file in this dir
    for imgPath in subdir.glob("*"):
        image_list.append((Path.relative_to(imgPath, ROOT), index))
    NUM_CLASSES += 1

random.shuffle(image_list)

# split dataset into train and validation dateset
train_size = math.floor(len(image_list) * (1-VALID_RATIO))
valid_size = len(image_list) - train_size


# ----------------------------------------------------------------

# save to file
with open(TRAIN_LIST, 'w') as f:
    for img, cat in image_list[:train_size]:
        f.write(f'{img} {cat}\n')

with open(VALID_LIST, 'w') as f:
    for img, cat in image_list[train_size:]:
        f.write(f'{img} {cat}\n')


# ----------------------------------------------------------------

if __name__=='__main__':

    # ----------------------------------------------------------------

    model = LightCNN(num_classes=NUM_CLASSES)

    # large lr for last fc parameters
    if torch.cuda.is_available():
        model = model.cuda()

    print(model)

    # ----------------------------------------------------------------

    # define transforms
    train_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
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

    # ----------------------------------------------------------------

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

    for iEpoch in range(0, EPOCHS):

        # train for one epoch
        train_loss, train_acc = helper.train(iEpoch)

        # evaluate on validation set
        valid_loss, valid_acc = helper.validate(iEpoch)

        # log tensorboard
        LOGGER.add_scalar('train/loss', train_loss, iEpoch)
        LOGGER.add_scalar('train/accuracy', train_acc, iEpoch)
        LOGGER.add_scalar('valid/loss', valid_loss, iEpoch)
        LOGGER.add_scalar('valid/accuracy', valid_acc, iEpoch)

    # Testing
    helper.VALID_DATALOADER = valid_loader
    helper.validate(iEpoch, mode='test')

    
    # ----------------------------------------------------------------

    # TODO

    