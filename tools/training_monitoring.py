import matplotlib.pyplot as plt


class LossesMonitor(object):

    def __init__(self):

        self.train_epochs = list()
        self.test_epochs = list()
        self.train_elbo = list()
        self.test_elbo = list()

    def append_values(self, epoch, loss, set="train"):

        if set == "train":

            self.train_epochs.append(epoch)
            self.train_elbo.append(loss)

        elif set == "test":

            self.test_epochs.append(epoch)
            self.test_elbo.append(loss)

        else:
            raise ValueError("the set must be train or test")

    def show_losses(self):

        assert len(self.train_epochs) > 0, "must have some values"
        assert len(self.test_epochs) > 0, "must have some values"
        assert len(self.train_elbo) > 0, "must have some values"
        assert len(self.test_elbo) > 0, "must have some values"

        plt.title("Evidence lower bound")
        plt.plot(self.test_epochs, self.test_elbo, ".-", label="test")
        plt.plot(self.train_epochs, self.train_elbo, ".-", label="train")
        plt.legend()
        plt.show()
