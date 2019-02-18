__author__ = 'Cedric'
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import torchvision
import torchvision.transforms
import matplotlib.pyplot as plt

torch.manual_seed(12345)

mnist_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
mnist_train = torchvision.datasets.MNIST(root='./data_mnist', train=True, transform=mnist_transforms, download=False)
mnist_test = torchvision.datasets.MNIST(root='./data_mnist', train=False, transform=mnist_transforms, download=False)

train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=True)

class Classifier(nn.Module):
    """Convnet Classifier"""
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1),
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
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.fcl = nn.Sequential(
            # Layer 1
            nn.Linear(64*7*7, 230),
            nn.ReLU(),

            # Layer 2
            nn.Linear(230, 10),
            nn.Softmax()
        )


    def forward(self, x):
        return self.fcl(self.conv(x).view(-1, 64*7*7))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

cuda_available = torch.cuda.is_available()
print(cuda_available)

clf = Classifier()
print(count_parameters(clf))

if cuda_available:
    clf = clf.cuda()
optimizer = torch.optim.SGD(clf.parameters(), lr=1e-1)
criterion = nn.CrossEntropyLoss()

train_error = []
train_loss = []
test_error = []
test_loss = []

for epoch in range(10):
    losses_train = []
    losses_test = []
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
    total_test = 0
    correct_test = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if cuda_available:
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs = clf(inputs)
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, targets)
        losses_test.append(loss.data.item())
        total_test += targets.size(0)
        correct_test += predicted.eq(targets.data).cpu().sum().item()
    test_error.append(100.*(1-correct_test/total_test))
    test_loss.append(np.mean(losses_test))
    print('Epoch : %d Test Acc : %.3f' % (epoch, 100.*correct_test/total_test))
    print('Train error:'+str(100.*(1-correct_train/total_train)))
    print('Test error:'+str(100.*(1-correct_test/total_test)))
    print('--------------------------------------------------------------')
    clf.train()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(count_parameters(clf))
plt.plot(train_error, label='Train')
plt.plot(test_error, label='Valid')
plt.ylabel('Error (%)')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.plot(train_loss, label='Train')
plt.plot(test_loss, label='Valid')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
