import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse

# Evaluation Exercise 8: Implementing Lasso Regression
class LassoRegression(Model):
    def __init__(self, l1_penalty: float = 1.0, scale: bool = True, max_iter: int = 1000, patience: int = 5, tolerance: float = 1e-4, **kwargs):
        """
        Initializes the Lasso Regression model.
        (Least Absolute Shrinkage and Selection Operator)

        Parameters
        ----------
        l1_penalty : float
            The L1 regularization parameter (lambda).
        scale : bool
            Whether to scale the features of the dataset by subtracting the mean or by the standard deviation. 
        max_iter : int
            The maximum number of iterations for the coordinate descent algorithm.
        patience : int
            The number of consecutive iterations allowed without significant improvement before stopping the optimization early.
        tolerance : float
            The convergence tolerance for the optimization (early stopping condition).
            Helps save computational resources by avoiding unnecessary iterations after reaching an acceptable solution.
            Setting the tolerance to 0 disables it.
            
        Attributes
        ----------
        theta : np.ndarray
            Coefficients for each feature in the dataset.
        theta_zero : float
            The intercept of the model.
        mean : np.ndarray
            The mean of each feature in the dataset.
        std : np.ndarray
            The standard deviation of each feature in the dataset.
        """
        # Parameters
        super().__init__(**kwargs)
        self.l1_penalty = l1_penalty
        self.scale = scale
        self.max_iter = max_iter
        self.patience = patience
        self.tolerance = tolerance  # Convergence tolerance

        # Attributes
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def _fit(self, dataset: Dataset) -> 'LassoRegression':
        """
        Fit the model using coordinate descent.

        Parameters
        ----------
        dataset : Dataset
            The training dataset.

        Returns
        -------
        self : LassoRegression
            The fitted model.
        """
        # Scale the data
        if self.scale:
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        y = dataset.y
        n_features = X.shape[1]

        # Initialize the coefficients
        self.theta = np.zeros(n_features)
        self.theta_zero = 0

        # Coordinate descent loop
        early_stopping = 0
        for iter in range(self.max_iter):
            theta_prev = self.theta.copy() # Associated with convergence tolerance

            # Update coefficients
            for j in range(n_features):
                residuals = y - (X.dot(self.theta) + self.theta_zero) + self.theta[j] * X[:, j]
                rho_j = X[:, j].T.dot(residuals)

                # Apply Soft-Thresholding and update theta j
                if rho_j > self.l1_penalty:
                    self.theta[j] = (rho_j - self.l1_penalty) / np.sum(X[:, j] ** 2)    # Condition 1: soft_threshold = rho_j - lambda if rho_j > lambda.
                elif rho_j < -self.l1_penalty:
                    self.theta[j] = (rho_j + self.l1_penalty) / np.sum(X[:, j] ** 2)    # Condition 3: soft_threshold = rho_j + lambda if rho_j < -lambda.
                else:
                    self.theta[j] = 0                                                   # Condition 2: soft_threshold = 0 if Module of rho_j =< lambda (or rather, if previous conditions don't occur).

            # Update intercept
            self.theta_zero = np.mean(y - X.dot(self.theta))

            # Check for convergence
            if np.linalg.norm(self.theta - theta_prev, ord=1) < self.tolerance:
                early_stopping += 1
            else:
                early_stopping = 0

            if early_stopping >= self.patience:
                break                               # Interrupt process if patience threshold is exceeded!

        return self


    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the dependent variable using the estimated theta coefficients.

        Parameters
        ----------
        dataset : Dataset
            The dataset to predict the target variable for.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        if self.scale:
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        predictions = X.dot(self.theta) + self.theta_zero

        return predictions


    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calculate the MSE score between the true and predicted values.

        Parameters
        ----------
        dataset : Dataset
            The dataset to evaluate the model on.

        predictions: np.ndarray
            Predictions.

        Returns
        -------
        float
            Mean Squared Error (MSE) score.
        """
        return mse(dataset.y, predictions)