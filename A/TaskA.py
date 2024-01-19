# Import libraries

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
import medmnist
from medmnist import INFO, Evaluator
import matplotlib
import matplotlib.pyplot as plt


# ResNet-18
class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, inCh, outCh, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels=inCh, out_channels=outCh,
                      kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(outCh),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=outCh, out_channels=outCh,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(outCh)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or inCh != outCh:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=inCh, out_channels=outCh,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(outCh)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, classes=2):
        super(ResNet18, self).__init__()
        self.classes = classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer_1 = self.make_layer(ResidualBlock, 64, 64, stride=1)
        self.layer_2 = self.make_layer(ResidualBlock, 64, 128, stride=2)
        self.layer_3 = self.make_layer(ResidualBlock, 128, 256, stride=2)
        self.layer_4 = self.make_layer(ResidualBlock, 256, 512, stride=2)
        self.avgpool = nn.AvgPool2d((3, 3), stride=2)
        self.fc = nn.Linear(512 * ResidualBlock.expansion, self.classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def make_layer(self, block, inCh, outCh, stride, block_num=2):
        layers = []
        layers.append(block(inCh, outCh, stride))
        for i in range(block_num - 1):
            layers.append(block(outCh, outCh, 1))
        return nn.Sequential(*layers)


# test
def test(split):
    model.eval()
    y_true = torch.Tensor([])
    y_score = torch.Tensor([])
    if torch.cuda.is_available():  # testing by using GPU if available
        y_true, y_score = y_true.cuda(), y_score.cuda()
        if split == 'train':
            data_loader = train_loader_at_eval
        elif split == 'test':
            data_loader = test_loader
        elif split == 'val':
            data_loader = val_loader

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.cuda()
                targets = targets.cuda()
                outputs = model(inputs)
                targets = targets.squeeze().long()
                outputs = outputs.softmax(dim=-1)
                targets = targets.float().resize_(len(targets), 1)

                y_true = torch.cat((y_true, targets), 0)
                y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()

        evaluator = Evaluator('pneumoniamnist', split)
        metrics = evaluator.evaluate(y_score)

        print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))

    else:
        if split == 'train':
            data_loader = train_loader_at_eval
        elif split == 'test':
            data_loader = test_loader
        elif split == 'val':
            data_loader = val_loader

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.cuda()
                targets = targets.cuda()
                outputs = model(inputs)
                targets = targets.squeeze().long()
                outputs = outputs.softmax(dim=-1)
                targets = targets.float().resize_(len(targets), 1)

                y_true = torch.cat((y_true, targets), 0)
                y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.numpy()
        y_score = y_score.detach().numpy()

        evaluator = Evaluator('pneumoniamnist', split)
        metrics = evaluator.evaluate(y_score)

        print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))




# Parameters
batch_size = 128
num_epoches = 10
lr = 0.001

info = INFO['pneumoniamnist']
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])


# Data processing and Load
# preprocessing
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# load the data
train_dataset = DataClass(split='train', transform=data_transform, download=True)
val_dataset = DataClass(split='val', transform=data_transform, download=True)
test_dataset = DataClass(split='test', transform=data_transform, download=True)

# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*batch_size, shuffle=False)
val_loader = data.DataLoader(dataset=val_dataset, batch_size=2*batch_size, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*batch_size, shuffle=False)

# train
model = ResNet18(classes=n_classes)
criterion = nn.CrossEntropyLoss()

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
    model = model.cuda()
    criterion = criterion.cuda()

optimizer = optim.Adam(model.parameters(), lr=lr)
loss_list = []

if torch.cuda.is_available():  # training by using GPU if available
    print('Task A is training on GPU ...')
    for epoch in range(num_epoches):
        train_correct = 0
        train_total = 0
        test_correct = 0
        test_total = 0

        model.train()

        for inputs, targets in tqdm(train_loader):
            # forward + backward + optimize
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        loss_list.append(loss / len(train_loader))

        print('Epoch: {}  Loss: {}'.format(epoch + 1, loss / len(train_loader)))

    print('Training finished')

else:
    print('Task A is training on CPU ...')
    for epoch in range(num_epoches):
        train_correct = 0
        train_total = 0
        test_correct = 0
        test_total = 0

        model.train()

        for inputs, targets in tqdm(train_loader):
            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(inputs)
            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        loss_list.append(loss / len(train_loader))

        print('Epoch: {}  Loss: {}'.format(epoch + 1, loss / len(train_loader)))

    print('Training finished')



# Train loss curve plot
iterition = range(num_epoches)
Loss = loss_list

plt.plot(iterition, Loss, '-')
plt.title('Training loss vs. epoches')
plt.ylabel('Training loss')
plt.xlabel('epoches')
plt.show()
# plt.savefig('./loss.png') # save plot


# testing
print('==> Evaluating ...')
train_result = test('train')
val_result = test('val')
test_result = test('test')