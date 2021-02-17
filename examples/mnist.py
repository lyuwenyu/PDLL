import argparse
import numpy as np 

import _init_path
import pdll as L
import pdll.nn as nn
import pdll.optim as optim
import pdll.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pooling = nn.MaxPool2d(3, 2, 1)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        n, _, _, _ = x.shape
        x = self.bn1(self.conv1(x)).relu()
        x = self.bn2(self.conv2(x)).relu()
        x = self.pooling(x)
        x = self.fc1(x.reshape(n, -1)).relu()
        x = self.fc2(x)
        return x


def train(args, model, train_loader, optimizer, epoch):
    '''
    '''
    model.train()

    for _idx, (data, label) in enumerate(train_loader):

        data = L.from_numpy(np.array(data))
        label = np.array(label)
        label = L.Variable(np.eye(10)[label])

        output = model(data)
        loss = F.cross_entropy(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if _idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, _idx * data.shape[0], len(train_loader.dataset),
                100. * _idx * data.shape[0] / len(train_loader.dataset), 
                loss.data))


def test(model, test_loader):
    model.eval()

    correct = 0
    for data, label in test_loader:
        # data = L.from_numpy(data.data.numpy())
        data = L.from_numpy(np.array(data))
        label = np.array(label)
        
        output = model(data)

        correct += (output.data.argmax(axis=1) == label).sum()


    print('\nAccuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 8}
    test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': False, 'num_workers': 8}

    model = Net()
    L.io.save(model, '../data/mnist.pickle')
    del model
    model = L.io.load('../data/mnist.pickle')

    train_dataset = L.io.dataset.MNIST(root='../data/MNIST/raw/', train=True)
    test_dataset = L.io.dataset.MNIST(root='../data/MNIST/raw/', train=False)

    train_loader = L.io.DataLoader(train_dataset, **train_kwargs)
    test_loader = L.io.DataLoader(test_dataset, **test_kwargs)

    print(len(list(model.parameters())))

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4])

    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, optimizer, epoch)
        test(model, test_loader)
        scheduler.step()
        print(epoch, optimizer.lr)

if __name__ == '__main__':
    main()