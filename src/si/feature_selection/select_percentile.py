from typing import Callable
from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification

import numpy as np

class SelectPercentile(Transformer):
    """
    Select features according to the highest scores up to a certain percentile.
    Feature ranking is performed by computing the scores of each feature using a scoring function:
        - f_classification: ANOVA F-value between label/feature for classification tasks.
        - f_regression: F-value obtained from F-value of r's pearson correlation coefficients for regression tasks.

    Parameters
    ----------
    score_func: callable
        Function taking dataset and returning a pair of arrays (scores, p_values)
    percentile: float, default=10
        Percentile of top features to select.

    Attributes
    ----------
    F: array, shape (n_features,)
        F scores of features.
    p: array, shape (n_features,)
        p-values of F-scores.
    """
    def __init__(self, score_func: Callable = f_classification, percentile: float = 10, **kwargs):
        """
        Select features according to the highest scores up to a certain percentile

        Parameters
        ----------

        score_func: callable
            Function taking dataset and returning a pair of arrays (scores, p_values)
        percentile: float, default=10
            Percentile of top features to select.
        
        """
        super().__init__(**kwargs)
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        It fits SelectPercentile to compute the F scores and p-values.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        self: object
            Returns self.
        """
        self.F, self.p = self.score_func(dataset)
        return self


    def _transform(self, dataset: Dataset) -> Dataset:
        """
        It transforms the dataset by selecting the highest scoring features according to the given percentile.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        dataset: Dataset
            A labeled dataset with the highest scoring features according to the given percentile.
        """
        num_features = int(np.ceil(self.percentile / 100 * dataset.X.shape[1]))

        idxs = np.argsort(self.F)[-num_features:]
        features = np.array(dataset.features)[idxs]
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)
    
if __name__ == '__main__':
    from si.data.dataset import Dataset

    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")

    selector = SelectPercentile(percentile=50)
    selector = selector.fit(dataset)
    dataset = selector.transform(dataset)
    print(dataset.features)