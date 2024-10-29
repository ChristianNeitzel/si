# Evaluation Exercise 4: Implement the Cosine distance function.

import numpy as np

def cosine_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculates the Cosine Distance between a single sample x and multiple samples y.

    Measures the dissimilarity between vectors by calculating the cosine of the angle between them.
        Cosine distance of 0 -> vectors are perfectly aligned (maximum similarity)
        Cosine distance closer to 2 -> vectors are diametrically opposite (maximum dissimilarity)

    Parameters:
        x (array): A 1D array representing a single sample.
        y (array): A 2D array where each row is a sample.

    Returns:
        distances (array): An array of Cosine Distances between x and each row in y.
    """

    # Initialize an array to store distances
    distances = np.zeros(y.shape[0])

    x_norm = np.sqrt(np.sum(x ** 2))            # Calculate the norm of x (||x||)

    for i in range(y.shape[0]):
        dot_product = np.dot(x, y[i])           # Calculate the dot product bwettn x and the i-th sample in y
        y_norm = np.sqrt(np.sum(y[i] ** 2))     # Calculate the norm of y[i] (||y[i]||)

        # Calculare cosine similarity
        similarity = dot_product / (x_norm * y_norm)

        # Calculate cosine distance
        distances[i] = 1 - similarity

    return distances