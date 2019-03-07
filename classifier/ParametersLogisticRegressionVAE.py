import numbers

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

    @input_size.setter
    def input_size(self, value):

        assert isinstance(value, numbers.Real)
        assert isinstance(value, int), "must be an integer"
        assert value > 1, "must be greater than 1"

        self._input_size = value

    @property
    def num_classes(self):
        return self._num_classes

    @num_classes.setter
    def num_classes(self, value):
        assert isinstance(value, numbers.Real)
        assert isinstance(value, int), "must be an integer"
        assert value > 2, "must be greater than 2"

        self._num_classes = value

    @property
    def num_epochs(self):
        return self._num_epochs

    @num_epochs.setter
    def num_epochs(self, value):
        assert isinstance(value, numbers.Real)
        assert isinstance(value, int), "must be an integer"
        assert value > 1, "must be greater than 1"

        self._num_epochs = value

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        assert isinstance(value, numbers.Real)
        assert isinstance(value, int), "must be an integer"
        assert value > 1, "must be greater than 1"

        self._batch_size = value

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        assert isinstance(value, numbers.Real)
        assert value < 1.0, "must be lower than one"

        self._learning_rate = value

