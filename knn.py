import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # Compute distances between each test point and all training points
        distances = np.linalg.norm(self.X_train - X[:, np.newaxis], axis=2)
        # Get the indices of the k nearest neighbors
        k_indices = np.argsort(distances, axis=1)[:, :self.k]
        # Gather the k nearest labels
        k_nearest_labels = self.y_train[k_indices]
        # Use Counter to find the most common class for each test point
        most_common = [Counter(labels).most_common(1)[0][0] for labels in k_nearest_labels]
        return np.array(most_common)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Example usage:
if __name__ == "__main__":
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create KNN instance and fit data
    knn = KNN(k=3)
    knn.fit(X_train, y_train)

    # Predict using custom implementation
    predictions_custom = knn.predict(X_test)
    print("Custom KNN Predictions:", predictions_custom)

    # Predict using scikit-learn
    knn_sklearn = KNeighborsClassifier(n_neighbors=3)
    knn_sklearn.fit(X_train, y_train)
    predictions_sklearn = knn_sklearn.predict(X_test)
    print("Scikit-learn KNN Predictions:", predictions_sklearn)

    # Compare accuracy
    accuracy_custom = np.mean(predictions_custom == y_test)
    accuracy_sklearn = np.mean(predictions_sklearn == y_test)
    print(f"Custom KNN Accuracy: {accuracy_custom:.2f}")
    print(f"Scikit-learn KNN Accuracy: {accuracy_sklearn:.2f}")
