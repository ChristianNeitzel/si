from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

from datasets import DATASETS_PATH

import os

from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from si.neural_networks.layers import DenseLayer, Dropout
from si.neural_networks.optimizers import Optimizer

class MockOptimizer(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        return w - self.learning_rate * grad_loss_w

class TestDenseLayer(TestCase):
    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")
        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_forward_propagation(self):
        dense_layer = DenseLayer(n_units=30)
        dense_layer.set_input_shape((self.dataset.X.shape[1], ))
        dense_layer.initialize(MockOptimizer(0.001))
        output = dense_layer.forward_propagation(self.dataset.X, training=False)

        self.assertEqual(output.shape[0], self.dataset.X.shape[0])
        self.assertEqual(output.shape[1], 30)

    def test_backward_propagation(self):
        dense_layer = DenseLayer(n_units=30)
        dense_layer.set_input_shape((self.dataset.X.shape[1], ))
        dense_layer.initialize(MockOptimizer(learning_rate=0.001))
        dense_layer.forward_propagation(self.dataset.X, training=True)
        input_error = dense_layer.backward_propagation(output_error=np.random.random((self.dataset.X.shape[0], 30)))

        self.assertEqual(input_error.shape[0], self.dataset.X.shape[0])
        self.assertEqual(input_error.shape[1], 9)


# Evaluation Exercise 12: Unittest of Dropout layer implementation.
class TestDropoutLayer(TestCase):
    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")
        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

        self.dropout_layer = Dropout(probability=0.3)

    def test_forward_propagation_training(self):
        output = self.dropout_layer.forward_propagation(self.train_dataset.X, training=True)    # Forward propagation in training mode

        self.assertEqual(output.shape, self.train_dataset.X.shape)
        self.assertGreater(np.sum(self.dropout_layer.mask == 0), 0)                             # Check that the mask is applied

    def test_forward_propagation_inference(self):
        output = self.dropout_layer.forward_propagation(self.train_dataset.X, training=False)   # Forward propagation in inference mode
        np.testing.assert_array_equal(output, self.train_dataset.X)                             # Check that output matches input

    def test_backward_propagation(self):
        self.dropout_layer.forward_propagation(self.train_dataset.X, training=True)             # Forward propagation in training mode to generate the mask

        output_error = np.random.random(self.train_dataset.X.shape)                             # Generate mock output error with same shape as input
        input_error = self.dropout_layer.backward_propagation(output_error)                     # backward propagation

        self.assertEqual(input_error.shape, output_error.shape)                                 # Check that input error has same shape as output error

        masked_elements = np.all(input_error[self.dropout_layer.mask == 0] == 0)
        self.assertTrue(masked_elements)                                                        # Check that mask is applied to the input error