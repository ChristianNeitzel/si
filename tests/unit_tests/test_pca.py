import os
from unittest import TestCase
from si.decomposition.pca import PCA
from si.io.csv_file import read_csv
from datasets import DATASETS_PATH

class TestPCA(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        n_components = 2
        pca = PCA(n_components=n_components)
        pca.fit(self.dataset)

        self.assertEqual(pca.components.shape, (n_components, self.dataset.shape()[1]))
        self.assertEqual(pca.explained_variance.shape, (n_components,))

    def test_transform(self):
        n_components = 2
        pca = PCA(n_components=n_components)
        pca.fit(self.dataset)
        reduced_dataset = pca.transform(self.dataset)

        self.assertEqual(reduced_dataset.X.shape, (self.dataset.shape()[0], n_components))

        self.assertEqual(reduced_dataset.features, [f"PC{i+1}" for i in range(n_components)])



# Testing if ValueError raise works.
    def test_invalid_n_components_type(self):
        """
        Test if ValueError is raised when n_components is not an integer.
        """
        with self.assertRaises(ValueError) as context:
            PCA(n_components=2.5).fit(self.dataset)                                 # Invalid type (float)
        self.assertIn("Invalid type for n_components", str(context.exception))

    def test_invalid_n_components_value(self):
        """
        Test if ValueError is raised when n_components is <= 0 or greater than the number of features.
        """
        # n_components <= 0
        with self.assertRaises(ValueError) as context:
            PCA(n_components=0).fit(self.dataset)                                   # Invalid value (<= 0)
        self.assertIn("Invalid value for n_components", str(context.exception))

        # n_components > number of features
        with self.assertRaises(ValueError) as context:
            PCA(n_components=10).fit(self.dataset)                                  # Invalid value (> number of features)
        self.assertIn("Invalid value for n_components", str(context.exception))