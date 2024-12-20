from typing import Tuple, Sequence, Union

import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray = None, features: Sequence[str] = None, label: str = None) -> None:
        """
        Dataset represents a tabular dataset for single output classification.

        Parameters
        ----------
        X: numpy.ndarray (n_samples, n_features)
            The feature matrix
        y: numpy.ndarray (n_samples, 1)
            The label vector
        features: list of str (n_features)
            The feature names
        label: str (1)
            The label name
        """
        if X is None:
            raise ValueError("X cannot be None")
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if features is not None and len(X[0]) != len(features):
            raise ValueError("Number of features must match the number of columns in X")
        if features is None:
            features = [f"feat_{str(i)}" for i in range(X.shape[1])]
        if y is not None and label is None:
            label = "y"
        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def shape(self) -> Tuple[int, int]:
        """
        Returns the shape of the dataset
        Returns
        -------
        tuple (n_samples, n_features)
        """
        return self.X.shape

    def has_label(self) -> bool:
        """
        Returns True if the dataset has a label
        Returns
        -------
        bool
        """
        return self.y is not None   # Returns boolean of whether or not there is a label (if self.y = None, function will return 'False')

    def get_classes(self) -> np.ndarray:
        """
        Returns the unique classes in the dataset
        Returns
        -------
        numpy.ndarray (n_classes)
        """
        if self.has_label():
            return np.unique(self.y)    # Returns unique values of self.y if there is a label
        else:
            raise ValueError("Dataset does not have a label")

    def get_mean(self) -> np.ndarray:
        """
        Returns the mean of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmean(self.X, axis=0)   # Returns the mean along the rows of each column (axis=0) whilst ignoring NaN values 

    def get_variance(self) -> np.ndarray:
        """
        Returns the variance of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanvar(self.X, axis=0)    # Returns the variance along the rows of each column (axis=0) whilst ignoring NaN values 

    def get_median(self) -> np.ndarray:
        """
        Returns the median of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmedian(self.X, axis=0) # Returns the median along the rows of each column (axis=0) whilst ignoring NaN values 

    def get_min(self) -> np.ndarray:
        """
        Returns the minimum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmin(self.X, axis=0)    # Returns the minimum row value of each column (axis=0) whilst ignoring NaN values 

    def get_max(self) -> np.ndarray:
        """
        Returns the maximum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmax(self.X, axis=0)    # Returns the maximum row value of each column (axis=0) whilst ignoring NaN values 

    def summary(self) -> pd.DataFrame:
        """
        Returns a summary of the dataset and displays them in a Pandas DataFrame format
        Returns
        -------
        pandas.DataFrame (n_features, 5)
        """
        data = {
            "mean": self.get_mean(),
            "median": self.get_median(),
            "min": self.get_min(),
            "max": self.get_max(),
            "var": self.get_variance()
        }
        return pd.DataFrame.from_dict(data, orient="index", columns=self.features)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, label: str = None):
        """
        Creates a Dataset object from a pandas DataFrame

        Parameters
        ----------
        df: pandas.DataFrame
            The DataFrame
        label: str
            The label name

        Returns
        -------
        Dataset
        """
        if label:
            X = df.drop(label, axis=1).to_numpy()
            y = df[label].to_numpy()
        else:
            X = df.to_numpy()
            y = None

        features = df.columns.tolist()
        return cls(X, y, features=features, label=label)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the dataset to a pandas DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        if self.y is None:
            return pd.DataFrame(self.X, columns=self.features)
        else:
            df = pd.DataFrame(self.X, columns=self.features)
            df[self.label] = self.y
            return df

    @classmethod
    def from_random(cls,
                    n_samples: int,
                    n_features: int,
                    n_classes: int = 2,
                    features: Sequence[str] = None,
                    label: str = None):
        """
        Creates a Dataset object from random data

        Parameters
        ----------
        n_samples: int
            The number of samples
        n_features: int
            The number of features
        n_classes: int
            The number of classes
        features: list of str
            The feature names
        label: str
            The label name

        Returns
        -------
        Dataset
        """
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        return cls(X, y, features=features, label=label)
    
    # Evaluation Exercise 2.1 - Implemention of method: dropna()
    def dropna(self):
        """
        Removes all rows in the dataset that contain at least one NaN value.

        Returns
        -------
        self : Dataset
            The modified Dataset object with NaN-containing rows removed.
        """
        # Mask variable that identifies the rows in X that contain NaNs
        mask = ~np.any(np.isnan(self.X), axis=1)

        # Filter X and y to only include rows that don't have NaNs
        self.X = self.X[mask] 
        self.y = self.y[mask]

        return self
    
    # Evaluation Exercise 2.2 - Implementation of method: fillna()
    def fillna(self, value: Union[float, str] = 'mean'):
        """
        Fills all NaN values in the dataset with a specific value, or the mean or median of the respective feature.
        
        Parameters
        ----------
        value: Union[float, str]
            If a float -> replaces NaNs with the given float value.
            If value='mean' -> replaces NaNs with the mean of the respective feature. (default setting)
            If value='median' -> replaces NaNs with the median of the respective feature.
        
        Returns
        -------
        self : Dataset
            The modified Dataset object with NaN values replaced.
        """

        if isinstance(value, str):
            if value == 'mean':
                fill_values = np.nanmean(self.X, axis=0)    # Prepares condition to perform the mean of the non-NaN elements of each column.
            elif value == 'median':
                fill_values = np.nanmedian(self.X, axis=0)  # Prepares condition to perform the median of the non-NaN elements of each column.
            else:
                raise ValueError("Invalid value for 'fillna'. Value must be 'mean', 'median' or a float")
        else:
            fill_values = np.full(self.X.shape[1], value)   # Prepares condition to replace NaN values with the defined float value.

        # Replace NaN values with the corresponding fill value
        indices = np.where(np.isnan(self.X))                # Get indices of NaN values
        self.X[indices] = np.take(fill_values, indices[1])  # Replace NaNs with the corresponding fill values
        return self
    
    # Evaluation Exercise 2.3 - Implementation of method: remove_by_index()
    def remove_by_index(self, index: int):
        """
        Removes the sample (row) at the given index from both the feature matrix X and the label vector y.
    
        Parameters
        ----------
        index: int
            The index of the sample to remove.

        Returns
        -------
        self : Dataset
            The modified Dataset object with the sample removed.
        """
        # Check if the index is valid
        if index < 0 or index >= len(self.X):
            raise IndexError("Index is out of bounds!")
        
        # Remove the row at the specified index from X
        self.X = np.delete(self.X, index, axis=0)

        # If the dataset has labels (y), remove the corresponding label entry
        if self.y is not None:
            self.y = np.delete(self.y, index)

        return self

if __name__ == '__main__':
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([1, 2])
    features = np.array(['a', 'b', 'c'])
    label = 'y'
    dataset = Dataset(X, y, features, label)
    print(dataset.shape())
    print(dataset.has_label())
    print(dataset.get_classes())
    print(dataset.get_mean())
    print(dataset.get_variance())
    print(dataset.get_median())
    print(dataset.get_min())
    print(dataset.get_max())
    print(dataset.summary())
