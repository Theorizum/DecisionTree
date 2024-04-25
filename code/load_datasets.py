import numpy as np

datasetPath = "datasets"


def load_iris_np():
    # reproducibility
    np.random.seed(1)

    dtype = [
        ("sepal_length", "f4"),
        ("sepal_width", "f4"),
        ("petal_length", "f4"),
        ("petal_width", "f4"),
        ("species", "U20"),
    ]

    dataset = np.genfromtxt(
        f"{datasetPath}/bezdekIris.data", delimiter=",", dtype=dtype
    )

    # prevent any potential biases
    np.random.shuffle(dataset)

    return dataset


def load_iris_dataset(train_ratio):
    """
    Loads the Iris dataset and splits it into training and testing sets based on a specified ratio.

    Args:
        train_ratio (float): The proportion of examples to be assigned to training.
                             For example, if the ratio is 0.5, 50% of the examples (75 examples)
                             will be used for training and the remaining 50% for testing.

    Returns:
        tuple: Contains four numpy arrays:
               - train: Features for training examples.
               - train_labels: Labels corresponding to train examples.
               - test: Features for testing examples.
               - test_labels: Labels corresponding to test examples.
    """
    dataset = load_iris_np()

    train_size = int(len(dataset) * train_ratio)
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]

    # Extract features and labels for training and testing
    train = np.array([tuple(row)[:4] for row in train_dataset], dtype=np.float32)
    train_labels = np.array([row["species"] for row in train_dataset])

    test = np.array([tuple(row)[:4] for row in test_dataset], dtype=np.float32)
    test_labels = np.array([row["species"] for row in test_dataset])

    return train, train_labels, test, test_labels


def load_iris_dataset_folds(train_ratio, n_folds):
    """Load the Iris dataset and split it into train and test sets for cross-validation."""
    pass


def load_wine_np():
    """Load the Binary Wine quality dataset into a numpy array."""
    pass


def load_wine_dataset(train_ratio):
    """Load the Binary Wine quality dataset and split it into train and test sets."""
    pass


def load_wine_dataset_folds(train_ratio, n_folds):
    """Load the Binary Wine quality dataset and split it into train and test sets for cross-validation."""
    pass


def load_abalone_np():
    """Load the Abalone-intervalles dataset into a numpy array."""
    pass


def load_abalone_dataset(train_ratio):
    """Load the Abalone-intervalles dataset and split it into train and test sets."""
    pass


def load_abalone_dataset_folds(train_ratio, n_folds):
    """Load the Abalone-intervalles dataset and split it into train and test sets for cross-validation."""
    pass
