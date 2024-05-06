import numpy as np


class NaiveBayes:

    def __init__(self):
        self.classes = None
        self.class_probs = {}
        self.feature_probs = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        num_samples, num_features = X.shape

        # Calculate class probabilities
        for c in self.classes:
            self.class_probs[c] = np.sum(y == c) / num_samples

        # Calculate conditional probabilities for each feature
        for c in self.classes:
            self.feature_probs[c] = {}
            class_samples = X[y == c]
            for i in range(num_features):
                self.feature_probs[c][i] = {
                    'mean': np.mean(class_samples[:, i]),
                    'std': np.std(class_samples[:, i]) + 1e-10  # Add a small value to prevent division by zero
                }

    def predict(self, X):
        predictions = []
        for sample in X:
            probs = {c: np.log(self.class_probs[c]) for c in self.classes}
            for c in self.classes:
                for i, feature in enumerate(sample):
                    mean = self.feature_probs[c][i]['mean']
                    std = self.feature_probs[c][i]['std']
                    exponent = (feature - mean) ** 2 / (2 * std ** 2)
                    probs[c] += -np.log(std) - exponent
            predicted_class = max(probs, key=probs.get)
            predictions.append(predicted_class)
        return predictions

    def predict_proba(self, X):
        probs = []
        for sample in X:
            sample_probs = {}
            for c in self.classes:
                class_prob = np.log(self.class_probs[c])
                for i, feature in enumerate(sample):
                    mean = self.feature_probs[c][i]['mean']
                    std = self.feature_probs[c][i]['std']
                    exponent = (feature - mean) ** 2 / (2 * std ** 2)
                    class_prob += -np.log(std) - exponent
                sample_probs[c] = np.exp(class_prob)
            probs.append(sample_probs)
        return probs
    