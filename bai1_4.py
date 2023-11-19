import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

def evaluatePerformance(data, labels):
    """
    Evaluate the performance of decision tree, decision stumps, and 3-level decision tree
    using 100 trials of 10-fold cross-validation.

    Parameters:
    - data: Feature matrix
    - labels: Corresponding labels

    Returns:
    - performance_matrix: Matrix of statistics (mean and std) for accuracy over all trials and classifiers
    """

    # Number of trials and folds
    num_trials = 100
    num_folds = 10

    # Placeholder for accuracy values
    accuracy_values = np.zeros((num_trials * num_folds, 3))  # 3 classifiers

    # Loop over trials
    for trial in range(num_trials):
        # Shuffle data at the start of each trial
        data, labels = shuffle(data, labels)

        # Split data into folds
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=None)

        # Loop over folds
        for i, (train_index, test_index) in enumerate(skf.split(data, labels)):
            # Prepare training and testing sets
            train_data, test_data = data[train_index], data[test_index]
            train_labels, test_labels = labels[train_index], labels[test_index]

            # Train and evaluate the SKLearn decision tree classifier
            clf_sklearn = DecisionTreeClassifier()
            clf_sklearn.fit(train_data, train_labels)
            predictions_sklearn = clf_sklearn.predict(test_data)
            accuracy_values[trial * num_folds + i, 0] = accuracy_score(test_labels, predictions_sklearn)

            # Train and evaluate decision stumps (1-level decision tree)
            clf_stumps = DecisionTreeClassifier(max_depth=1)
            clf_stumps.fit(train_data, train_labels)
            predictions_stumps = clf_stumps.predict(test_data)
            accuracy_values[trial * num_folds + i, 1] = accuracy_score(test_labels, predictions_stumps)

            # Train and evaluate 3-level decision tree
            clf_3level = DecisionTreeClassifier(max_depth=3)
            clf_3level.fit(train_data, train_labels)
            predictions_3level = clf_3level.predict(test_data)
            accuracy_values[trial * num_folds + i, 2] = accuracy_score(test_labels, predictions_3level)

    # Calculate mean and standard deviation of accuracy values
    mean_accuracy = np.mean(accuracy_values, axis=0)
    std_dev_accuracy = np.std(accuracy_values, axis=0)

    # Create a matrix of statistics (mean and std) for accuracy
    performance_matrix = np.vstack((mean_accuracy, std_dev_accuracy)).T

    return performance_matrix
