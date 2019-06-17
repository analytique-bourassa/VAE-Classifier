from tools.tools import setup_data_loaders

from VariationalAutoEncoder.utils import evaluate_vae, train_vae
from VariationalAutoEncoder.VariationalAutoEncoder import VAE

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from tqdm import tqdm
from tools.training_monitoring import LossesMonitor
from classifier.ParametersLogisticRegressionVAE import ParameterLogisticRegressionVAE
from classifier.LogisticRegression_VAE import LogisticRegression
from tools.Visualisations import compare_images

from classifier.utils import train_classifier, evaluate_classifier
from classifier.utils import return_model_accurary
from tools.Visualisations import t_SNE, describe_statistic_per_label, show_confusion_matrix

SMOKE_TEST = False
TEST_FREQUENCY = 5

pyro.enable_validation(True)
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)
pyro.clear_param_store()

LEARNING_RATE = 1.0e-3
USE_CUDA = False
NUM_EPOCHS = 1 if SMOKE_TEST else 31

train_loader, test_loader = setup_data_loaders(batch_size=256, use_cuda=USE_CUDA)

###################################################
# Variational Autoencoder definition and training #
###################################################

vae = VAE(use_cuda=USE_CUDA)

adam_args = {"lr": LEARNING_RATE}
optimizer = Adam(adam_args)
svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

loss_monitor_vae = LossesMonitor()

# training loop
for epoch in range(NUM_EPOCHS):

    total_epoch_loss_train = train_vae(svi, train_loader, use_cuda=USE_CUDA)
    loss_monitor_vae.append_values(epoch, -total_epoch_loss_train, set="train")
    print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

    if epoch % TEST_FREQUENCY == 0:

        total_epoch_loss_test = evaluate_vae(svi, test_loader, use_cuda=USE_CUDA)
        loss_monitor_vae.append_values(epoch, -total_epoch_loss_test, set="test")
        print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))

if not SMOKE_TEST:
    loss_monitor_vae.show_losses(type_loss="Evidence lower bound (ELBO)")
    compare_images(vae, test_loader)

vae.encoder.freeze()

######################################
# Classifier definition and training #
######################################

classifier = LogisticRegression(number_hidden_units=vae.z_dim, num_classes=10, encoder=vae.encoder)

parameters_LR_VAE = ParameterLogisticRegressionVAE()
if SMOKE_TEST:
    parameters_LR_VAE.num_epochs = 2

optimizer = torch.optim.SGD(classifier.parameters(), lr=parameters_LR_VAE.learning_rate)

loss_monitor_classifier = LossesMonitor()

for epoch in tqdm(range(parameters_LR_VAE.num_epochs)):

    loss_train = train_classifier(classifier, optimizer, train_loader)
    loss_monitor_classifier.append_values(epoch, loss_train, set="train")
    print("[epoch %03d]  average training loss: %.4f" % (epoch, loss_train))

    if epoch % TEST_FREQUENCY == 0:
        loss_test = evaluate_classifier(classifier, test_loader)
        loss_monitor_classifier.append_values(epoch, loss_train, set="test")
        print("[epoch %03d]  average testing loss: %.4f" % (epoch, loss_test))

if not SMOKE_TEST:
    loss_monitor_classifier.show_losses(type_loss="Cross entropy")

accuracy = return_model_accurary(classifier, test_loader)
print('Accuracy of the model on the 10000 test images: %d %%' % accuracy)

t_SNE(test_loader, vae)
describe_statistic_per_label(test_loader, vae)
show_confusion_matrix(train_loader,classifier)
