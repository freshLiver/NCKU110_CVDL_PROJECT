from light_cnn import LightCNN_9Layers
from functional import TrainingHelper, ImageList

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# hyper-parameters
TEST_RATIO = 0.2

# training
EPOCHS = 5
BS = 128             # batch size
LR = 0.0001         # learning rate

ROOT = "./"
TRAIN_LIST = "./train_list.txt"
VALID_LIST = "./valid_list.txt"
NUM_CLASSES = 5749
NUM_WORKER = 2
PRINT_FREQUENCY = 10


if __name__ == "__main__":

    model = LightCNN_9Layers()

    # large lr for last fc parameters
    if torch.cuda.is_available():
        model = model.cuda()

    print(model)

    # load image
    train_loader = DataLoader(
        ImageList(
            root=ROOT,
            fileList=TRAIN_LIST,
            transform=transforms.Compose([
                transforms.RandomCrop(128),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        ),
        batch_size=BS,
        shuffle=True,
        num_workers=NUM_WORKER,
        pin_memory=True
    )

    valid_loader = DataLoader(
        ImageList(
            root=ROOT,
            fileList=VALID_LIST,
            transform=transforms.Compose([
                transforms.CenterCrop(128),
                transforms.ToTensor(),
            ])
        ),
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
