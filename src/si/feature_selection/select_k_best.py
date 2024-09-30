import numpy as np

from si.base.estimator import Estimator
from si.base.transformer import Transformer
from si.data.dataset import Dataset

class SelectKBest(Transformer):
    """
    Parameters:
        score_func: variance analysis function
        k: number of features to select
    Estimated Parameters:
        F: the F value for each feature estimated by the score_func
        p: the p value for each feature estimated by the score_func
    Methods
        _fit: estimates the F and p values for each feature using the scoring_func; returns self
        _transform: selects the top k features with the highest F value and returns the selected X
    """
    def __init__(self, score_func: callable, k: int, **kwargs):
        super().__init__(**kwargs)
        self.score_func = score_func
        self.k = k
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> 'SelectKBest':
            self.F, self.p = self.score_func(dataset)

            return self

    def _transform(self, dataset: Dataset) -> Dataset:
        idx = np.argsort(self.F)
        mask = idx[-self.k:]  # Mask to obtain the top k features with the highest F value
        new_X = dataset.X[:, mask]
        new_features = dataset.features[mask]

        return Dataset(X=new_X, features=new_features, y=dataset.y, label=dataset.label)
