import numpy as np

# Evaluation Exercise 4: Implement the Cosine distance function.
def cosine_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate the cosine distance between a single sample 'x' and multiple samples 'y'.

    Similarity ranges from 0 to 1:
        similarity = 0 -> vectors are unrelated;
        similarity = 1 -> vectors are identical (maximum similarity).

    Distance (output) ranges from 0 to 2:
        distance = 0 -> vectors are identical (maximum similarity);
        distance = 1 -> vectors are unrelated;
        distance = 2 -> vectors are diametrically opposite (maximum dissimilarity).

    Note: distance = 1 - similarity

    Parameters
    ----------
    x : numpy.ndarray
        A 1D array representing a single sample (n_features,).
    y : numpy.ndarray
        A 2D array where each row is a sample (n_samples, n_features).

    Returns
    -------
    numpy.ndarray
        An array of cosine distances between 'x' and each sample in 'y'.
    """
    # Prepare variables for the similarity formula
    dot_product = np.dot(y, x)          # Calculate dot product between y and x
    norm_x = np.linalg.norm(x)          # Obtain norm of x (||x||)
    norm_y = np.linalg.norm(y, axis=1)  # Obtain norm of y (||y||)

    # Compute similarity
    similarity = dot_product / (norm_x * norm_y)

    # Calculate and output cosine distance
    return 1 - similarity