import torch.nn as nn
import torch
from torch.autograd import Variable

def train_classifier(classifier, optimizer,train_loader):

    total_loss_value = 0.0
    number_of_observations =  len(train_loader.dataset)

    for i, (images, labels) in enumerate(train_loader):

        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        criterion = nn.CrossEntropyLoss()

        optimizer.zero_grad()
        outputs = classifier(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss_value += loss.item()

    normalized_loss = total_loss_value / number_of_observations

    return normalized_loss

def evaluate_classifier(classifier, test_loader):

    total_loss_value = 0.0
    number_of_observations = len(test_loader.dataset)

    for i, (images, labels) in enumerate(test_loader):

        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        criterion = nn.CrossEntropyLoss()

        outputs = classifier(images)
        loss = criterion(outputs, labels)

        total_loss_value += loss.item()

    normalized_loss = total_loss_value/number_of_observations

    return normalized_loss

def return_model_accurary(classifier, test_loader):

    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, 28 * 28))
        outputs = classifier(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    return 100 * correct / total

