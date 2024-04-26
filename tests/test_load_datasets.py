import unittest
import numpy as np

from ..code.load_datasets import (
    load_iris_np,
    load_iris_dataset,
    load_iris_dataset_folds,
)


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
        train_ratio = 0.7
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

    def test_load_iris_dataset_folds(self):
        train_ratio = 0.7
        n_folds = 5

        # load data and splits
        n_features, n_classes, folds, test_dataset = load_iris_dataset_folds(
            train_ratio, n_folds
        )

        # check number of features and classes
        self.assertEqual(n_features, 4, "Expected 4 features")
        self.assertEqual(n_classes, 3, "Expected 3 unique classes (species)")

        # check test and train data type
        self.assertIsInstance(folds, np.ndarray, "Folds should be an array")
        self.assertIsInstance(
            test_dataset, np.ndarray, "Test dataset should be an array"
        )

        # check number of folds
        self.assertEqual(
            len(folds), n_folds, "Number of folds does not match requested folds"
        )

        # check train/test split
        dataset = load_iris_np()
        expected_test_size = int(len(dataset) * (1 - train_ratio))
        self.assertEqual(
            len(test_dataset), expected_test_size, "Test set size mismatch"
        )

        # check that each fold is non-empty and has correct structure
        for fold in folds:
            self.assertNotEqual(len(fold), 0, "Fold should not be empty")
            self.assertIsInstance(fold, np.ndarray, "Each fold should be a numpy array")
            self.assertTrue(
                all(len(item) == 5 for item in fold),
                "Each entry in a fold should have 5 fields",
            )

        # check data consistency across folds
        all_fold_data = np.concatenate(folds)
        unique_fold_species = np.unique(all_fold_data["species"])
        expected_species = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"}
        self.assertTrue(
            all(species in expected_species for species in unique_fold_species),
            "Some species names in folds are incorrect",
        )

        # check the stratification: Each class should be roughly equally represented across folds
        species_counts = {species: 0 for species in expected_species}
        for fold in folds:
            for species in fold["species"]:
                species_counts[species] += 1
        average_count_per_species = np.mean(list(species_counts.values()))
        # check equal distribution
        for count in species_counts.values():
            self.assertTrue(
                np.isclose(count, average_count_per_species, rtol=0.1),
                "Species not equally distributed across folds",
            )
