'''
EECS 445 - Introduction to Machine Learning
Winter 2019 - Project 2
Train Challenge
    Trains a neural network to classify images
    Periodically outputs training information, and saves model checkpoints
    Usage: python train_challenge.py
'''
import torch
import torchvision.models as models
import numpy as np
import utils
from dataset import get_train_val_test_loaders
from model.challenge import Challenge
from model.cnn import CNN
from train_common import *
from utils import config

def _train_epoch(data_loader, model, criterion, optimizer):
    """
    Train the `model` for one epoch of data from `data_loader`
    Use `optimizer` to optimize the specified `criterion`
    """
    # TODO: complete the training step
    for i, (X, y) in enumerate(data_loader):
        # clear parameter gradients
        optimizer.zero_grad()
        #

        # forward + backward + optimize
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        #
    #

def _evaluate_epoch(axes, tr_loader, val_loader, model, criterion, epoch, stats):
    with torch.no_grad():
        y_true, y_pred = [], []
        correct, total = 0, 0
        running_loss = []
        for X, y in tr_loader:
            output = model(X)
            predicted = predictions(output.data)
            y_true.append(y)
            y_pred.append(predicted)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())
        train_loss = np.mean(running_loss)
        train_acc = correct / total
    with torch.no_grad():
        y_true, y_pred = [], []
        correct, total = 0, 0
        running_loss = []
        for X, y in val_loader:
            output = model(X)
            predicted = predictions(output.data)
            y_true.append(y)
            y_pred.append(predicted)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())
        val_loss = np.mean(running_loss)
        val_acc = correct / total
    stats.append([val_acc, val_loss, train_acc, train_loss])
    utils.log_cnn_training(epoch, stats)
    utils.update_cnn_training_plot(axes, epoch, stats)

def main():
    # data loaders
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        num_classes=config('challenge.num_classes'))

    # TODO: define model, loss function, and optimizer
    # model = models.resnet18(pretrained = False)
    model = Challenge()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 0.005)
    #

    # Attempts to restore the latest checkpoint if exists
    print('Loading challenge...')
    model, start_epoch, stats = restore_checkpoint(model,
        config('challenge.checkpoint'))

    axes = utils.make_cnn_training_plot()

    # Evaluate model
    _evaluate_epoch(axes, tr_loader, va_loader, model, criterion, start_epoch,
        stats)

    # Loop over the entire dataset multiple times
    for epoch in range(start_epoch, config('challenge.num_epochs')):
        # Train model
        _train_epoch(tr_loader, model, criterion, optimizer)

        # Evaluate model
        _evaluate_epoch(axes, tr_loader, va_loader, model, criterion, epoch+1,
            stats)

        # Save model parameters
        save_checkpoint(model, epoch+1, config('challenge.checkpoint'), stats)

    print('Finished Training')

    with torch.no_grad():
        y_stat = np.zeros((2,5))
        for X, y in va_loader:
            output = model(X)
            predicted = predictions(output.data)
            for i in range(y.size(0)):
                if (y[i] < 5):
                    y_stat[0][y[i]] = y_stat[0][y[i]] + 1
                    if (predicted[i] == y[i]):
                        y_stat[1][y[i]] = y_stat[1][y[i]] + 1
        y_final = np.zeros(5)
        for i in range(5):
            y_final[i] = y_stat[1][i] / y_stat[0][i]
    print('classes accuracy:', y_final)
    overall_no = (y_stat[1][0] + y_stat[1][1] + y_stat[1][2] + y_stat[1][3] + y_stat[1][4])
    overall_de = (y_stat[0][0] + y_stat[0][1] + y_stat[0][2] + y_stat[0][3] + y_stat[0][4])
    overall = overall_no / overall_de
    print('overall accuracy:', overall)

    # Keep plot open
    utils.hold_training_plot()



if __name__ == '__main__':
    main()