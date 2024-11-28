from itertools import product
from typing import Any, Callable, Dict, Tuple

import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.model_selection.cross_validate import k_fold_cross_validation

# Evaluation Exercise 11: Implementing the randomized_search_cv function.
def randomized_search_cv(model: Model, 
                         dataset: Dataset,
                         hyperparameter_grid: Dict[str, Tuple],
                         scoring: Callable = None,
                         cv: int = 5,
                         n_iter: int = None) -> Dict[str, Any]:
    """
    Perform randomized search cross-validation on a model.
    
    Parameters
    ----------
    model
        The model to cross validate.

    dataset : Dataset
        The dataset to cross validate on.

    hyperparameter_grid : Dict[str, Tuple]
        The hyperparameter grid to use.

    scoring : Callable
        The scoring function to use.

    cv : int
        The cross validation folds.

    n_iter: int
        The number of random hyperparameter combinations to evaluate.

    Returns
    -------
    results : Dict[str, Any]
        The results of the randomized search cross validation. Includes the scores, hyperparameters,
        best hyperparameters and best score.
    """
    # Validate the parameter grid
    for parameter in hyperparameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f'Model {model} does not have parameter {parameter}.')
        

    # Get all possible hyperparameter combinations
    all_combinations = list(product(*hyperparameter_grid.values()))

    # Randomly sample n_iter combinations
    if n_iter > len(all_combinations):
        raise ValueError(f"n_iter cannot exceed the total number of combinations: {len(all_combinations)}")
    sampled_combinations = np.random.choice(len(all_combinations), size=n_iter, replace=False)
    sampled_combinations = [all_combinations[i] for i in sampled_combinations]


    # Initializing the results dictionary
    results = {
        "hyperparameters": [],
        "scores": [],
        "best_hyperparameters": None,
        "best_score": -np.inf
    }

    # Perform randomized search
    for combination in sampled_combinations:
        parameters_dict = {key: value for key, value in zip(hyperparameter_grid.keys(), combination)}

        # Set model parameters
        for parameters, value in parameters_dict.items():
            setattr(model, parameters, value)

        # Cross-validate
        scores = k_fold_cross_validation(model, dataset, scoring=scoring, cv=cv)
        mean_score = np.mean(scores)

        # Update results
        results["hyperparameters"].append(parameters_dict)
        results["scores"].append(mean_score)

        if mean_score > results["best_score"]:
            results["best_score"] = mean_score
            results["best_hyperparameters"] = parameters_dict

    return results


if __name__ == '__main__':
    # Import dataset
    from si.data.dataset import Dataset
    from si.models.logistic_regression import LogisticRegression
    from si.model_selection.split import train_test_split
    from si.io.csv_file import read_csv

    # Load the dataset
    data = read_csv('datasets/breast_bin/breast-bin.csv', sep=',', features=True, label=True) # Ensure your terminal is in the si directory!

    # Split the dataset into training and testing sets
    train, test = train_test_split(data, test_size=0.33, random_state=42)

    # Initialize the Logistic Regression model
    lr = LogisticRegression()

    # Parameter grid
    parameter_grid_ = {
        'l2_penalty': np.linspace(1, 10, 10),
        'alpha': np.linspace(0.001, 0.0001, 100),
        'max_iter': np.linspace(1000, 2000, 200)
    }

    # Cross validate the model
    results_ = randomized_search_cv(model=lr,
                                    dataset=data,
                                    hyperparameter_grid=parameter_grid_,
                                    cv=3,
                                    n_iter=10)

    # Get the scores
    scores = results_['scores']
    print(f"Scores: {scores}")

    # Get the best score
    best_score = results_['best_score']
    print(f"Best score: {best_score}")