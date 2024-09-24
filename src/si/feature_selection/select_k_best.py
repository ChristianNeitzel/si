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