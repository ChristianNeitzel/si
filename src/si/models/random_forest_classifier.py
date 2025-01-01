# Evaluation Exercise 9: Implementing the RandomForestClassifier class.

from typing import Literal

import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.models.decision_tree_classifier import DecisionTreeClassifier
# from si.statistics.impurity import gini_impurity, entropy_impurity


class RandomForestClassifier(Model):
    def __init__(self, n_estimators: int = 100, 
                 max_features: int = None, 
                 min_sample_split: int = 2, 
                 max_depth: int = 10, 
                 mode: Literal['gini', 'entropy'] = 'gini', 
                 seed: int = None, 
                 **kwargs):
        """
        Initializes the Random Forest Classifier.

        Parameters
        ----------
        n_estimators : int
            The number of decision trees to use.
        max_features : int
            The maximum number of features to use per tree.
        min_sample_split : int
            The minimum samples allowed in a split.
        max_depth
            The maximum depth of the trees.
        mode : {"gini", "entropy"}
            Impurity calculation mode.
            Function to measure the quality of a split (refered to as "criterion" in sklearn).
        seed : int
            The random seed to use to assure reproducibility.

        Attributes
        ----------
        trees : List[]
            The list containing tuples of features used and the corresponding trained decision trees.
       """
        # Parameters
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed

        # Attributes
        self.trees = []

    def _fit(self, dataset: Dataset) -> 'RandomForestClassifier':
        """
        Fit the Random Forest model using bootstrapped datasets.

        Parameters
        ----------
        dataset : Dataset
            The training dataset.

        Returns
        -------
        self : RandomForestClassifier
            The trained decisions trees of the random forest.
        """
        # Set the random seed
        if self.seed is not None:
            np.random.seed(self.seed)

        # Define max_features
        X = dataset.X
        n_samples, n_features = X.shape

        if self.max_features == None:
            self.max_features = int(np.sqrt(n_features))

        for _ in range(self.n_estimators):
            # Create a bootstrap dataset (n_samples random samples from dataset WITH replacement)
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_X = dataset.X[bootstrap_indices]
            bootstrap_y = dataset.y[bootstrap_indices]

            # Select random features (self.max_features random features WITHOUT replacement from the original dataset)
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
            bootstrap_X = bootstrap_X[:, feature_indices]

            # Create and train a decision tree on the bootstrap dataset
            tree = DecisionTreeClassifier(
                min_sample_split=self.min_sample_split,
                max_depth=self.max_depth,
                mode=self.mode
            )

            tree.fit(Dataset(X=bootstrap_X, y=bootstrap_y))

            # Store the features used and the trained tree
            self.trees.append((feature_indices, tree))

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the labels for the given dataset using majority voting.
        
        Parameters
        ----------
        dataset : Dataset
            The dataset to make predictions for.

        Returns
        -------
        numpy.ndarray
            The predicted values.
        """
        tree_predictions = []

        for feature_indices, tree in self.trees:
            # Extract the relevant features' names
            selected_features = [dataset.features[i] for i in feature_indices]
            
            # Use only the features relevant to the tree
            tree_X = Dataset(X=dataset.X[:, feature_indices], 
                            y=dataset.y, 
                            features=selected_features, 
                            label=dataset.label)
            
            # Predict using the current tree
            predictions = tree.predict(tree_X)
            tree_predictions.append(predictions)

        # Convert tree predictions to a NumPy array for easy manipulation
        tree_predictions = np.array(tree_predictions).T

        # Perform majority voting: iterate through rows and choose the most common label
        def majority_vote(preds):
            values, counts = np.unique(preds, return_counts=True)
            return values[np.argmax(counts)]

        predictions = np.apply_along_axis(majority_vote, axis=1, arr=tree_predictions)

        return predictions


    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calculate the accuracy of the model.

        Parameters
        ----------
        dataset : Dataset
            The dataset to evaluate the model on.
        predictions : numpy.ndarray
            The predicted labels for the dataset.
        
        Return
        ------
        score : float
            The accuracy of the model.
        """
        return accuracy(dataset.y, predictions)
    
# Evaluation Exercise 9.2: Testing the random forest class directly to get model score on the test set (iris dataset).
if __name__ == '__main__':
    from si.io.csv_file import read_csv
    from si.model_selection.split import train_test_split

    # Note: Ensure you are connect to myenv and that your terminal is on the ..\si directory!
    data = read_csv('datasets/iris/iris.csv', sep=',', features=True, label=True)

    train, test = train_test_split(data, test_size=0.33, random_state=42)
    model = RandomForestClassifier()
    model.fit(train)
    print(f'Model score on the test set (test_size=0.33) of iris dataset: {model.score(test)}')