class ParameterLogisticRegressionVAE(object):

    def __init__(self):

        self._input_size = 784
        self._num_classes = 10
        self._num_epochs = 100
        self._batch_size = 100
        self._learning_rate = 0.001

    @property
    def input_size(self):
        return self._input_size

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_epochs(self):
        return self._num_epochs

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def learning_rate(self):
        return self._learning_rate

