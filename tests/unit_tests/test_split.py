from unittest import TestCase
from datasets import DATASETS_PATH

import os
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.model_selection.split import stratified_train_test_split

class TestSplits(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_train_test_split(self):

        train, test = train_test_split(self.dataset, test_size = 0.2, random_state=123)
        test_samples_size = int(self.dataset.shape()[0] * 0.2)
        self.assertEqual(test.shape()[0], test_samples_size)
        self.assertEqual(train.shape()[0], self.dataset.shape()[0] - test_samples_size)
    

# Evaluation Exercise 6.2: Testing stratified_train_test_split function with the iris dataset.
    def test_stratified_train_test_split(self):
        train, test = stratified_train_test_split(self.dataset, test_size=0.2, random_state=123)

        # Check that the number of samples is correct
        total_samples = self.dataset.shape()[0]
        test_samples_size = int(total_samples * 0.2)
        train_samples_size = total_samples - test_samples_size

        self.assertEqual(test.shape()[0], test_samples_size)
        self.assertEqual(train.shape()[0], train_samples_size)

        # Check that all classes in the original dataset are present in both splits
        original_classes = set(self.dataset.y)
        train_classes = set(train.y)
        test_classes = set(test.y)

        self.assertEqual(original_classes, train_classes)
        self.assertEqual(original_classes, test_classes)

        # Check that each class maintains its approximate proportions
        for class_label in original_classes:
            # Get all indices for the current class
            original_class_count = len([y for y in self.dataset.y if y == class_label])
            train_class_count = len([y for y in train.y if y == class_label])
            test_class_count = len([y for y in test.y if y == class_label])

            # Calculate the expected number of test samples
            expected_test_count = int(original_class_count * 0.2)
            expected_train_count = original_class_count - expected_test_count

            # Validate the counts
            self.assertEqual(train_class_count, expected_train_count)
            self.assertEqual(test_class_count, expected_test_count)