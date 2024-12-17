from typing import Tuple

import numpy as np

from si.data.dataset import Dataset

def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # Set random state
    np.random.seed(random_state)
    
    # Get dataset size
    n_samples = dataset.shape()[0]

    # Get number of samples in the test set
    n_test = int(n_samples * test_size)

    # Get the dataset permutations
    permutations = np.random.permutation(n_samples)

    # Get samples in the test set
    test_indices = permutations[:n_test]

    # Get samples in the training set
    train_indices = permutations[n_test:]

    # Get the training and testing datasets
    train = Dataset(dataset.X[train_indices], dataset.y[train_indices], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_indices], dataset.y[test_indices], features=dataset.features, label=dataset.label)

    return train, test

# Evaluation Exercise 6.1: Implementing stratified splitting.
def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Stratified split of the dataset into training and testing sets. 
    Preserves class proportions between train and test sets after dataset splitting.

    Parameters
    ----------
    dataset : Dataset
        The dataset to split.
    test_size : float
        The proportion of the dataset to include in the test split.
    random_state : int
        The seed of the random number generator.

    Returns
    -------
    Tuple[Dataset, Dataset]
        A tuple containing the stratified training and testing datasets.
    """  
    # Set random state
    np.random.seed(random_state)

    # Get unique labels and their respective counts
    labels, counts = np.unique(dataset.y, return_counts=True)

    # Initialize lists to store train and test indices
    train_indices = []
    test_indices = []

    # Loop through each unique label and split indices based on counts
    for label, count in zip(labels, counts):

        # Get indices of samples belonging to the current label
        label_indices = np.where(dataset.y == label)[0]

        # Shuffle the indices
        np.random.shuffle(label_indices)

        # Determine the number of test samples for the current label
        n_test_samples = int(count * test_size)

        # Split indices into test and train for this label
        test_indices.extend(label_indices[:n_test_samples])
        train_indices.extend(label_indices[n_test_samples:])
    
    # Convert lists to numpy arrays for indexing
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    # Create stratified train and test datasets
    train = Dataset(dataset.X[train_indices], dataset.y[train_indices], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_indices], dataset.y[test_indices], features=dataset.features, label=dataset.label)

    return train, test