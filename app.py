import numpy as np
import pandas as pd
import NaiveBayes
import LogisticRegression
import knn
from sklearn.model_selection import train_test_split


# Read dataset
dataset = pd.read_csv('spambase.csv')

# Split dataset into x and y
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

## NAIVE BAYES ALGORITHM ##
    
# Instantiate the Naive Bayes Classifier
nb_classifier = NaiveBayes.NaiveBayes()

# Train the classifier
nb_classifier.fit(x_train, y_train)

# Make predictions
y_pred = nb_classifier.predict(x_test)


# Evaluation
# Accuracy
def accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total


# False Positive
def false_positive(y_true, y_pred):
    return np.sum((y_true != y_pred) & (y_pred == 1))


# True Positive
def true_positive(y_true, y_pred):
    return np.sum((y_true == y_pred) & (y_pred == 1))


# Calculate metrics
acc = accuracy(y_test, y_pred)
fp = false_positive(y_test, y_pred)
tp = true_positive(y_test, y_pred)

print("Naive Bayes Algorithm:")

print("Accuracy:", acc)
print("False Positive:", fp)
print("True Positive:", tp)


# AUC calculation
def auc_score(y_true, y_prob):
    # Sorting by probabilities
    sorted_indices = np.argsort(y_prob)
    y_true_sorted = y_true[sorted_indices]

    # Counting positive samples
    num_positives = np.sum(y_true)
    num_negatives = len(y_true) - num_positives

    # Summing ranks of positive samples
    ranks = np.zeros_like(y_true_sorted, dtype=float)
    ranks[y_true_sorted == 0] = np.arange(1, num_negatives + 1)
    ranks[y_true_sorted == 1] = np.arange(1, num_positives + 1)

    # Calculating AUC
    auc = (np.sum(ranks[y_true_sorted == 1]) - num_positives * (num_positives + 1) / 2) / (
                num_positives * num_negatives)
    return auc


# Calculate predicted probabilities
y_prob = nb_classifier.predict_proba(x_test)
y_prob = np.array([probs[1] for probs in y_prob])

# Calculate AUC
auc = auc_score(y_test, y_prob)
print("AUC:", auc, "\n")

## LOGISTIC REGRESSION ALGORITHM ##
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# Normalize input features
def normalize(X):
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    return (X - min_vals) / (max_vals - min_vals)

x_train_normalized = normalize(x_train)
x_test_normalized = normalize(x_test)

logistic_classifier = LogisticRegression.LogisticRegression()

logistic_classifier.fit(x_train_normalized, y_train)

y_pred = logistic_classifier.predict(x_test_normalized)

acc = accuracy(y_test, y_pred)
fp = false_positive(y_test, y_pred)
tp = true_positive(y_test, y_pred)


print("Logistic Regression Algorithm:")
print("Accuracy:", acc)
print("False Positive:", fp)
print("True Positive:", tp)

# Calculate AUC
def auc_score(y_true, y_prob):
    # Sorting by probabilities
    sorted_indices = np.argsort(y_prob)
    y_true_sorted = y_true[sorted_indices]

    # Counting positive samples
    num_positives = np.sum(y_true)
    num_negatives = len(y_true) - num_positives

    # Summing ranks of positive samples
    ranks = np.zeros_like(y_true_sorted, dtype=float)
    ranks[y_true_sorted == 0] = np.arange(1, num_negatives + 1)
    ranks[y_true_sorted == 1] = np.arange(1, num_positives + 1)

    # Calculating AUC
    auc = (np.sum(ranks[y_true_sorted == 1]) - num_positives * (num_positives + 1) / 2) / (
                num_positives * num_negatives)
    return auc


# Calculate predicted probabilities
y_prob = logistic_classifier.sigmoid(np.dot(x_test, logistic_classifier.theta))

# Calculate AUC
auc = auc_score(y_test, y_prob)
print("AUC:", auc, '\n')


## KNN ALGORITHM ##
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

knn_classifier = knn.KNN(k=5)

knn_classifier.fit(x_train, y_train)

y_pred = knn_classifier.predict(x_test)


acc = accuracy(y_test, y_pred)
fp = false_positive(y_test, y_pred)
tp = true_positive(y_test, y_pred)

print("KNN Algorithm:")
print("Accuracy:", acc)
print("False Positive:", fp)
print("True Positive:", tp)


def auc_score(y_true, y_prob):
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos

    # Mann-Whitney U statistic
    u_statistic = 0
    for i in range(len(y_prob)):
        for j in range(len(y_prob)):
            if y_true[i] == 1 and y_true[j] == 0:
                if y_prob[i] > y_prob[j]:
                    u_statistic += 1
                elif y_prob[i] == y_prob[j]:
                    u_statistic += 0.5

    # AUC
    auc = u_statistic / (n_pos * n_neg)
    return auc


y_prob = knn_classifier.predict_proba(x_test)

auc = auc_score(y_test, y_prob)
print("AUC: ", auc, "\n")


