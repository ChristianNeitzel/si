import unittest

import numpy as np

from si.data.dataset import Dataset


class TestDataset(unittest.TestCase):

    def test_dataset_construction(self):

        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])

        features = np.array(['a', 'b', 'c'])
        label = 'y'
        dataset = Dataset(X, y, features, label)

        self.assertEqual(2.5, dataset.get_mean()[0])
        self.assertEqual((2, 3), dataset.shape())
        self.assertTrue(dataset.has_label())
        self.assertEqual(1, dataset.get_classes()[0])
        self.assertEqual(2.25, dataset.get_variance()[0])
        self.assertEqual(1, dataset.get_min()[0])
        self.assertEqual(4, dataset.get_max()[0])
        self.assertEqual(2.5, dataset.summary().iloc[0, 0])

    def test_dataset_from_random(self):
        dataset = Dataset.from_random(10, 5, 3, features=['a', 'b', 'c', 'd', 'e'], label='y')
        self.assertEqual((10, 5), dataset.shape())
        self.assertTrue(dataset.has_label())


    def test_dropna(self):
        """
        Unittest of Exercise 2.1 (dropna() function implementation on class Dataset object)
        """
        X = np.array([[1, 2, 3], [np.nan, 5, 6], [7, np.nan, 9], [10, 11, 12]])
        y = np.array([1, 2, 3, 4])
        features = ['a', 'b', 'c']
        label = 'y'

        dataset = Dataset(X, y, features, label)                    # Initialize Dataset
        dataset.dropna()                                            # Applying dropna() function on dataset

        self.assertEqual(dataset.shape(), (2, 3))                   # Check shape after applying dropna(); Only 2 rows should remain
        self.assertFalse(np.isnan(dataset.X).any())                 # Check that no NaN values remain in X
        
        np.testing.assert_array_equal(dataset.y, np.array([1, 4]))  # Check that y was filtered correctly


    def test_fillna(self):
        """
        Unittest of Exercise 2.2 (fillna() function implementation on class Dataset object)
        """
        ## Test 1: Fill NaNs with a specific float value
        X = np.array([[1,       2, np.nan], 
                      [4,  np.nan,      6], 
                      [7,       8,      9], 
                      [np.nan, 11,    12]])
        y = np.array([1, 2, 3, 4])
        dataset = Dataset(X.copy(), y)

        # Replace NaNs with a specific value (e.g., 0.0)
        dataset.fillna(0.0)
        expected_X_value = np.array([[1, 2,   0], 
                                     [4, 0,   6], 
                                     [7, 8,   9], 
                                     [0, 11, 12]])
        self.assertFalse(np.isnan(dataset.X).any())
        np.testing.assert_array_equal(dataset.X, expected_X_value)

        ## Test 2: Fill NaNs with the mean of each feature
        dataset = Dataset(X.copy(), y)
        dataset.fillna("mean")
        expected_X_mean = np.array([[1, 2,   9], 
                                    [4, 7,   6], 
                                    [7, 8,   9], 
                                    [4, 11, 12]])  # Means: [4, 7, 9]
        self.assertFalse(np.isnan(dataset.X).any())
        np.testing.assert_array_almost_equal(dataset.X, expected_X_mean)

        ## Test 3: Fill NaNs with the median of each feature
        dataset = Dataset(X.copy(), y)
        dataset.fillna("median")
        expected_X_median = np.array([[1, 2,   9], 
                                      [4, 8,   6], 
                                      [7, 8,   9], 
                                      [4, 11, 12]])  # Medians: [4, 8, 9]
        self.assertFalse(np.isnan(dataset.X).any())
        np.testing.assert_array_almost_equal(dataset.X, expected_X_median)

        # Test 4: When there are no NaNs present in the array
        X_no_nan = np.array([[1, 2, 3], 
                             [4, 5, 6], 
                             [7, 8, 9]])
        dataset = Dataset(X_no_nan.copy(), y[:3])

        # Fill NaN with "mean" (or any value), no NaNs should be changed
        dataset.fillna("mean")
        expected_X_no_nan = X_no_nan
        np.testing.assert_array_equal(dataset.X, expected_X_no_nan)

        # Test 5: Invalid value for fillna()
        with self.assertRaises(ValueError):
            dataset.fillna("invalid")

    
    def test_remove_by_index(self):
        """
        Unittest of Exercise 2.3 (remove_by_index() function implementation on class Dataset object)
        """
        # Sample dataset (original array)
        X = np.array([[1, 2, 3], 
                      [4, 5, 6], 
                      [7, 8, 9]])
        y = np.array([1, 2, 3])
        dataset = Dataset(X, y)

        ## Test 1: Remove a valid index
        dataset.remove_by_index(1)  # Remove index 1 (second row)
        expected_X_after_removal = np.array([[1, 2, 3], 
                                             [7, 8, 9]])
        expected_y_after_removal = np.array([1, 3])
        
        np.testing.assert_array_equal(dataset.X, expected_X_after_removal)
        np.testing.assert_array_equal(dataset.y, expected_y_after_removal)

        ## Test 2: Remove another valid index
        dataset.remove_by_index(0)  # Remove index 0 (first row)
        expected_X_after_second_removal = np.array([[7, 8, 9]])
        expected_y_after_second_removal = np.array([3])
        
        np.testing.assert_array_equal(dataset.X, expected_X_after_second_removal)
        np.testing.assert_array_equal(dataset.y, expected_y_after_second_removal)

        ## Test 3: Remove the last remaining sample (sample was index 2 in the original array, since we removed the other two, it should be index 0 now)
        dataset.remove_by_index(0)  # Remove the last sample
        expected_X_after_final_removal = np.empty((0, 3))   # No samples left, empty array with 3 features
        expected_y_after_final_removal = np.array([])       # No labels left
        
        np.testing.assert_array_equal(dataset.X, expected_X_after_final_removal)
        np.testing.assert_array_equal(dataset.y, expected_y_after_final_removal)

        ## Test 4: Attempt to remove an out of bounds index
        with self.assertRaises(IndexError):
            dataset.remove_by_index(0)          # Should raise an error since the dataset is now empty

        ## Test 5: Attempt to remove a negative index
        with self.assertRaises(IndexError):
            dataset.remove_by_index(-1)         # Should raise an error for negative index