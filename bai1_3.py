import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_digits

# Load or define your dataset
digits = load_digits()
X, y = digits.data, digits.target

# Define classifiers (add your additional decision trees with varying depths)
classifiers = [
    ("Decision Stumps", AdaBoostClassifier(DecisionTreeClassifier(max_depth=1))),
    ("3-level Decision Tree", DecisionTreeClassifier(max_depth=3)),
    # Add more decision trees with varying depths
]

# Specify the percentage of training data to be used in the learning curve
training_percentages = np.arange(0.1, 1.1, 0.1)

# Initialize a figure for plotting
plt.figure(figsize=(10, 6))

# Iterate over classifiers
for clf_name, clf in classifiers:
    # Initialize lists to store mean and std of test accuracy for each training percentage
    mean_accuracy = []
    std_accuracy = []

    # Iterate over training percentages
    for training_percentage in training_percentages:
        # Calculate the number of training samples based on the percentage
        num_training_samples = int(len(X) * training_percentage)

        # Initialize lists to store accuracy for each trial
        accuracies = []

        # Perform 100 trials of 10-fold cross-validation
        for _ in range(100):
            # Create a stratified KFold object for cross-validation
            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

            # Initialize a list to store accuracy for each fold
            fold_accuracies = []

            # Iterate over folds
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Train the classifier on a subset of training data
                clf.fit(X_train[:num_training_samples], y_train[:num_training_samples])

                # Evaluate the classifier on the test set
                accuracy = clf.score(X_test, y_test)
                fold_accuracies.append(accuracy)

            # Calculate the mean accuracy for the current trial
            trial_accuracy = np.mean(fold_accuracies)
            accuracies.append(trial_accuracy)

        # Calculate the mean and std of accuracy over 100 trials for the current training percentage
        mean_accuracy.append(np.mean(accuracies))
        std_accuracy.append(np.std(accuracies))

    # Plot the learning curve for the current classifier
    plt.errorbar(
        training_percentages,
        mean_accuracy,
        yerr=std_accuracy,
        label=clf_name,
        capsize=4,
        marker='o'
    )

# Add labels, title, legend, etc., to the plot
plt.xlabel("Percentage of Training Data")
plt.ylabel("Mean Test Accuracy")
plt.title("Learning Curve for Different Classifiers")
plt.legend()
plt.grid(True)
plt.show()
