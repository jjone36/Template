# https://jhui.github.io/2018/02/09/PyTorch-neural-networks/
# https://deeplizard.com/learn/video/0LhiS6yu2qQ
#################################################
# Installation
!pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl
!pip3 install torchvision

import torch
print(torch.__version__)

torch.zeros()

x = torch.range(1, 16)
x = x.view(4, 4)
x = x.view(2, -1, 2)    # x.view(2, 4, 2)

torch.ca((X1, X2), dim = 0)   # row-wise
torch.stack((X1, X2))

#################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pytorch.optim as optim

class Model(nn.Module):

    def __init__(self):

        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 12, kernel_size = 5)

        self.fc1 = nn.Linear(in_features = 4*4*12, out_features = 120)
        self.fc2 = nn.Linear(in_features = 120, out_features = 60)
        self.out = nn.Linear(in_features = 60, out_features = 10)

    def forward(self, X):

        X = self.conv1(X)
        X = F.relu(X)
        X = F.max_pool2d(X, kernel_size = 2)

        X = self.conv2(X)
        X = F.relu(X)
        X = F.max_pool2d(X, kernel_size = 2)

        X = X.view(-1, self.num_flat_featues(X))
        X = self.fc1(X)
        X = F.relu(X)

        X = self.fc2(X)
        X = F.relu(X)

        X = self.out(X)
        return F.softmax(X, dim = 1)

    def num_flat_featues(self, X):

        size = X.size()[1:]
        num_featues = 1
        for s in size:
            num_featues *= s
        return num_featues



# Load the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size = 64, shuffle = True)

test_set = torchvision.datasets.CIFAR10(root='./data',
                                        train=False,
                                        download=True,
                                        transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 64, shuffle = True)

# Initiate the model
model = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)
# optimizer = optim.Adam(model.parameters(), lr = 0.001)

for epoch in range(10):

    total_loss = 0
    total_correct = 0

    for batch in train_loader:

        images, labels = batch
        images, labels = Variable(images), Variable(labels)

        preds = model(images)

        # Training
        optimizer.zero_grad()
        loss = criterion(preds, labels)       # Loss function
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)

        print("Epoch: {}, Total correct: {}, Loss: {}".format(epoch, total_correct, total_loss))


total = 0
correct = 0

for data in test_loader:

    images, labels = data
    images = Variable(images)

    preds = model(images)
    _, preds = torch.max(preds.data, 1)

    total += labels.size(0)
    correct += (preds == labels).sum()

print("Accuracy on the test set: ", correct/total*100)
