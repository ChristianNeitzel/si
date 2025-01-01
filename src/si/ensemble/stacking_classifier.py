# Evaluation Exercise 10: Implementing the StackingClassifier ensemble.

from typing import List
import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class StackingClassifier(Model):
    """
    Stacking Classifier ensemble.

    Ensemble learning technique that combines multiple base models to make predictions. 
    The base models are trained independently, and their predictions are used as features 
    for training the final model.
    """
    def __init__(self, models: List[Model], final_model: Model, **kwargs):
        """
        Initialize the Stacking Classifier ensemble.

        Parameters
        ----------
        models : List[Model]
            List of base models to be used in the stacking ensemble.
        final_model : Model
            The model used to make final predictions based on the base models.
        
        Attributes
        ----------
        predictions_dataset : Dataset
            The dataset created from the predictions of the base models. This dataset is used to train the final model.
        """
        # Parameters
        super().__init__(**kwargs)
        self.models = models            # Initial set of models
        self.final_model = final_model  # The model to make the final predictions
        
        # Attributes
        self.predictions_dataset = None # Dataset of base models predictions


    def _fit(self, dataset: Dataset) -> 'StackingClassifier':
        """
        Train the base models and the final model.

        Parameters
        ----------
        dataset : Dataset
            The training data.

        Returns
        -------
        self : StackingClassifier
            The fitted model.
        """
        # Train each base model on the training dataset
        for model in self.models:
            model.fit(dataset)

        # Generate predictions from the base models
        base_predictions = np.column_stack([model.predict(dataset) for model in self.models])

        # Create a dataset out of the base models predictions
        self.predictions_dataset = Dataset(X=base_predictions, y=dataset.y, label=dataset.label)

        # Train the final model on the predictions of the base models
        self.final_model.fit(self.predictions_dataset)

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the labels for the given dataset using the Stacking Classifier.

        Parameters
        ----------
        dataset : Dataset
            The dataset to predict the labels for.

        Returns
        -------
        numpy.ndarray
            The predicted labels.
        """
        # Generate predictions from the base models
        base_predictions = np.column_stack([model.predict(dataset) for model in self.models])

        # Get the final predictions using the final model and the predictions of the base models
        predictions = self.final_model.predict(Dataset(X=base_predictions))

        return predictions


    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calculate the accuracy of the model.

        Parameters
        ----------
        dataset : Dataset
            The dataset to evaluate the model on.
        predictions : numpy.ndarray
            The predictions.

        Returns
        -------
        score : float
            The mean accuracy.
        """
        return accuracy(dataset.y, predictions)