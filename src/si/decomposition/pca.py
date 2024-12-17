import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset

# Evaluation Exercise 5: Implementation of the PCA class.
class PCA(Transformer):
    """
    Principal Component Analysis (PCA).
    """
    def __init__(self, n_components, **kwargs):
        """
        Initializes the PCA with the desired number of components.

        Parameters
        ----------
        n_components : int
            Number of principal components to retain.
        """
        super().__init__(**kwargs)
        self.n_components = n_components
        self.is_fitted = False

        self.mean = None
        self.components = None
        self.explained_variance = None

    def _fit(self, dataset: Dataset) -> 'PCA':
        """
        Fit the PCA model to the data using eigenvalue decomposition of the covariance matrix.

        Parameters
        ----------
        dataset : Dataset
            The dataset used to estimate the PCA parameters.
        
        Returns
        -------
        self : PCA
            Returns the instance itself.
        
        Raises
        ------
        ValueError
            If n_components is not an integer.
            If n_components is not a positive value greater than 0 with less or greater than the number of features.
        """
        # Check for valid n_components: must be an integer
        if not isinstance(self.n_components, int):
            raise ValueError(
                f"Invalid type for n_components: {type(self.n_components).__name__}. "
                f"Expected an integer."
            )
        # Check for valid n_components: must be positive, and less or greater than the number of features
        if self.n_components <= 0 or self.n_components > dataset.shape()[1]:
            raise ValueError(
                f"Invalid value for n_components: {self.n_components}. "
                f"Must be a positive integer less than or equal to the number of features: {dataset.shape()[1]}."
            )

        # Step 1: Center the data
        self.mean = dataset.get_mean()          # Infering the mean of the samples
        X_centered = dataset.X - self.mean      # Subtracting the mean from the dataset

        # Step 2: Calculate the covariance matrix and perform eigenvalue decomposition
        self.covariance = np.cov(X_centered, rowvar=False)                      # Calculate covariance matrix of the centered data
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.covariance)   # Eigenvalue decomposition on the covariance matrix

        # Step 3: Infer the Principal Components
        sorted_idx = np.argsort(self.eigenvalues)[-self.n_components:][::-1] 

        # Step 4: Infer the Explained Variance (EV)
        # EV of one component is calculated by dividing the eigenvalue of that component with the sum of all eigenvalues
        # explained_variance corresponds to the first n_components of EV
        self.components = self.eigenvectors[:, sorted_idx].T
        self.explained_variance = self.eigenvalues[sorted_idx] / np.sum(self.eigenvalues)

        self.is_fitted = True

        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Transform the data to the reduced dimensional space using the principal components.

        Parameters
        ----------
        dataset : Dataset
            The data matrix of shape (n_samples, n_features).

        Returns
        -------
        Dataset
            The transformed data matrix of shape (n_samples, n_features) in the reduced dimension space.

        Raises
        ------
        ValueError
            If PCA has not been fitted. Ensure to call the 'fit' method before using 'transform'.
        """
        if not self.is_fitted:
            raise ValueError("PCA has not been fitted yet.")

        # Center the data
        X_centered = dataset.X - self.mean

        # Calculate the reduced data by projecting onto the principal components
        X_reduced = np.dot(X_centered, self.components.T)

        return Dataset(X_reduced, y=dataset.y, features=[f"PC{i+1}" for i in range(self.n_components)], label=dataset.label)