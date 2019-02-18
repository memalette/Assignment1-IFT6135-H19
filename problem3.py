__author__ = 'Cedric'
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms
import matplotlib.pyplot as plt

image_train_transforms = torchvision.transforms.Compose([torchvision.transforms.RandomAffine(degrees=10, shear=10,
                                                                                             scale=(0.9, 1.1)),
                                                        torchvision.transforms.RandomHorizontalFlip(),
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                                         [0.229, 0.224, 0.225])])
image_valid_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                                          [0.229, 0.224, 0.225])])

image_train = torchvision.datasets.ImageFolder(root='./data_dogs_cats/trainset', transform=image_train_transforms)
image_valid = torchvision.datasets.ImageFolder(root='./data_dogs_cats/validset', transform=image_valid_transforms)

train_loader = torch.utils.data.DataLoader(image_train, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(image_valid, batch_size=64, shuffle=True)

class Classifier(nn.Module):
    """Convnet Classifier"""
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),

            # Layer 2
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            # Layer 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),

            # Layer 4
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            # Layer 5
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),

            # Layer 6
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),

            # Layer 7
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            # Layer 8
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),

            # Layer 9
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),

            # Layer 10
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        self.fcl = nn.Sequential(
            # Layer 1
            nn.Linear(256*4*4, 2048),
            nn.ReLU(),

            # Layer 2
            nn.Linear(2048, 2048),
            nn.ReLU(),

            # Layer 3
            nn.Linear(2048, 2),
            nn.Softmax()
        )

    def forward(self, x):
        return self.fcl(self.conv(x).view(-1, 256*4*4))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    cuda_available = torch.cuda.is_available()
    print(cuda_available)

    clf = Classifier()
    if cuda_available:
        clf = clf.cuda()
    optimizer = torch.optim.SGD(clf.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    train_error = []
    train_loss = []
    valid_error = []
    valid_loss = []
    best_valid_error = float('inf')

    for epoch in range(500):
        losses_train = []
        losses_valid = []
        # Train
        total_train = 0
        correct_train = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if cuda_available:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = clf(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            losses_train.append(loss.data.item())
            total_train += targets.size(0)
            correct_train += predicted.eq(targets.data).cpu().sum().item()

            if batch_idx%50==0:
                print('Epoch : %d Loss : %.3f ' % (epoch, np.mean(losses_train)))

        train_error.append(100.*(1-correct_train/total_train))
        train_loss.append(np.mean(losses_train))

        # Evaluate
        clf.eval()
        total_valid = 0
        correct_valid = 0
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            if cuda_available:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = clf(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, targets)
            losses_valid.append(loss.data.item())
            total_valid += targets.size(0)
            correct_valid += predicted.eq(targets.data).cpu().sum().item()
        model_valid_error = 100.*(1-correct_valid/total_valid)
        valid_error.append(model_valid_error)
        valid_loss.append(np.mean(losses_valid))
        print('Epoch : %d Test Acc : %.3f' % (epoch, 100.*correct_valid/total_valid))
        print('Train error:'+str(100.*(1-correct_train/total_train)))
        print('Test error:'+str(model_valid_error))
        if model_valid_error < best_valid_error:
            print('Best model so far. Previous best: '+str(best_valid_error))
            best_valid_error = model_valid_error
            torch.save(clf.state_dict(), './submission_model')
        print('--------------------------------------------------------------')
        clf.train()

    print(count_parameters(clf))
    plt.plot(train_error, label='Train')
    plt.plot(valid_error, label='Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Error (%)')
    plt.legend()
    plt.show()

    plt.plot(train_loss, label='Train')
    plt.plot(valid_loss, label='Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()