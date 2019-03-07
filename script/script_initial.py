from tools.tools import setup_data_loaders

from VariationalAutoEncoder.utils import evaluate_vae, train_vae
from VariationalAutoEncoder.VariationalAutoEncoder import VAE

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torch.autograd import Variable
from tqdm import tqdm
from tools.training_monitoring import LossesMonitor
from classifier.ParametersLogisticRegressionVAE import ParameterLogisticRegressionVAE
from classifier.LogisticRegression_VAE import LogisticRegression
from tools.Visualisations import compare_images
import torchvision.datasets as dsets
import torchvision.transforms as transforms

train_dataset = dsets.MNIST(root="./data",
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root="./data",
                           train=False,
                           transform=transforms.ToTensor())


pyro.enable_validation(True)
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)


# Run options
LEARNING_RATE = 1.0e-3
USE_CUDA = False

# Run only for a single iteration for testing
NUM_EPOCHS = 31
TEST_FREQUENCY = 5

train_loader, test_loader = setup_data_loaders(batch_size=256, use_cuda=USE_CUDA)


# clear param store
pyro.clear_param_store()

# setup the VAE
vae = VAE(use_cuda=USE_CUDA)
# setup the optimizer
adam_args = {"lr": LEARNING_RATE}
optimizer = Adam(adam_args)

# setup the inference algorithm
svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

loss_monitor = LossesMonitor()

# training loop
for epoch in range(NUM_EPOCHS):

    total_epoch_loss_train = train_vae(svi, train_loader, use_cuda=USE_CUDA)
    loss_monitor.append_values(epoch, -total_epoch_loss_train, set="train")
    print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

    if epoch % TEST_FREQUENCY == 0:

        total_epoch_loss_test = evaluate_vae(svi, test_loader, use_cuda=USE_CUDA)
        loss_monitor.append_values(epoch, -total_epoch_loss_test, set="test")
        print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))

loss_monitor.show_losses()
compare_images(vae, test_loader)

vae.encoder.freeze()

classifier = LogisticRegression(number_hidden_units=vae.z_dim, num_classes=10, encoder=vae.encoder)


# Hyper Parameters
parameters_LR_VAE = ParameterLogisticRegressionVAE()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(classifier.parameters(), lr=parameters_LR_VAE.learning_rate)

# MNIST Dataset (Images and Labels)




# Training the Model
for epoch in tqdm(range(parameters_LR_VAE.num_epochs)):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = classifier(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

       # if epoch % TEST_FREQUENCY == 0:
            # report test diagnostics
            #total_epoch_loss_test = evaluate(svi, test_loader, use_cuda=USE_CUDA)
            #test_elbo.append(-total_epoch_loss_test)
            #test_epochs.append(epoch)
            #print("[epoch %03d] average test loss: %.4f" % (epoch, loss[0]))

        #if (i + 1) % 100 == 0:
        #    print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'
        #          % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))

# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28 * 28))
    outputs = classifier(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

