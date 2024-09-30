import scipy

from si.data.dataset import Dataset

def f_classification(dataset: Dataset) -> tuple:
    """
    Args: 
        dataset: dataset object
    Output:
        tuple with F values + tuple with p values
    """
    classes = dataset.get_classes()         # Obtain classes from the dataset via the function get_classes() from the "from si.data.dataset import Dataset" import

    groups = []                             # List of arrays that will contain our attributes
    for class_ in classes:                  # Note: the term 'class' cannot be defined here as it is already a natively defined python class.
        mask = dataset.y == class_
        group = dataset.X[ mask , :]
        groups.append(group)

    return scipy.stats.f_oneway(*groups)

