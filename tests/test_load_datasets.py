import unittest
import numpy as np

from ..code.load_datasets import load_iris_np, load_iris_dataset


class TestLoadIrisDataset(unittest.TestCase):
    def test_load_iris_np(self):
        iris_data = load_iris_np()

        # check it's a numpy array
        self.assertIsInstance(iris_data, np.ndarray, "expected a numpy array")

        # dataset should have 150 entries
        self.assertEqual(
            iris_data.shape, (150,), f"expected 150 entries, got {iris_data.shape}"
        )

        # should have five fields per record
        self.assertEqual(
            len(iris_data.dtype),
            5,
            f"expected five fields per record, got {len(iris_data.dtype)}",
        )

        # species column should be string type
        self.assertEqual(
            iris_data.dtype["species"].kind,
            "U",
            f"species should be string, got {iris_data.dtype['species'].kind}",
        )

        # all species names should be correct
        valid_species = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"}
        self.assertTrue(
            all(species in valid_species for species in iris_data["species"]),
            "found incorrect species names",
        )

    def test_load_iris_dataset(self):
        train_ratio = 0.5
        train, train_labels, test, test_labels = load_iris_dataset(train_ratio)

        # check all four outputs are arrays
        self.assertIsInstance(train, np.ndarray, "train data should be an array")
        self.assertIsInstance(
            train_labels, np.ndarray, "train labels should be an array"
        )
        self.assertIsInstance(test, np.ndarray, "test data should be an array")
        self.assertIsInstance(test_labels, np.ndarray, "test labels should be an array")

        # check correct training/test split
        total_samples = len(train) + len(test)
        expected_train_size = int(total_samples * train_ratio)
        self.assertEqual(len(train), expected_train_size, "train set size mismatch")

        # data arrays should have 4 features each
        self.assertEqual(train.shape[1], 4, "train should have 4 features")
        self.assertEqual(test.shape[1], 4, "test should have 4 features")

        # feature arrays should be float32
        self.assertEqual(
            train.dtype,
            np.float32,
            "train features should be float32 (see code/load_datasets.py f4)",
        )
        self.assertEqual(
            test.dtype,
            np.float32,
            "test features should be float32 (see code/load_datasets.py f4)",
        )

        # labels should be strings
        self.assertTrue(
            np.issubdtype(train_labels.dtype, np.str_),
            "train labels should be strings (see code/load_datasets.py U20)",
        )
        self.assertTrue(
            np.issubdtype(test_labels.dtype, np.str_),
            "test labels should be strings (see code/load_datasets.py U20)",
        )

        # all labels should be valid species names
        valid_species = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"}
        all_labels = np.concatenate([train_labels, test_labels])
        self.assertTrue(
            all(species in valid_species for species in all_labels),
            "some labels not valid species names",
        )
