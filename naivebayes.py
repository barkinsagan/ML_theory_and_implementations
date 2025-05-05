import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Initialize mean, var, and priors
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)

        # Prior is the freq based on the dataset.
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

if __name__ == "__main__":
    # Load the Iris dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the custom Naive Bayes classifier
    nb_custom = NaiveBayes()

    # Train the custom model
    nb_custom.fit(X_train, y_train)

    # Make predictions with the custom model
    y_pred_custom = nb_custom.predict(X_test)

    # Calculate accuracy for the custom model
    accuracy_custom = accuracy_score(y_test, y_pred_custom)
    print(f"Custom Naive Bayes Accuracy: {accuracy_custom * 100:.2f}%")

    # Initialize the built-in Gaussian Naive Bayes classifier
    nb_builtin = GaussianNB()

    # Train the built-in model
    nb_builtin.fit(X_train, y_train)

    # Make predictions with the built-in model
    y_pred_builtin = nb_builtin.predict(X_test)

    # Calculate accuracy for the built-in model
    accuracy_builtin = accuracy_score(y_test, y_pred_builtin)
    print(f"Built-in Gaussian Naive Bayes Accuracy: {accuracy_builtin * 100:.2f}%")

    # Apply PCA to reduce the dataset to 2 dimensions
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)

    # Plot PDFs for each class
    plt.figure(figsize=(10, 6))
    colors = ['r', 'g', 'b']
    for idx, c in enumerate(nb_custom._classes):
        X_c_pca = X_train_pca[y_train == c]
        plt.scatter(X_c_pca[:, 0], X_c_pca[:, 1], alpha=0.5, color=colors[idx], label=f'Class {c}')
        # Plot the mean
        mean_pca = pca.transform(nb_custom._mean[idx].reshape(1, -1))
        plt.scatter(mean_pca[0, 0], mean_pca[0, 1], color=colors[idx], edgecolor='k', s=100, marker='x')

    # Create a mesh grid for plotting decision boundaries
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Predict class using the custom model for each point in the mesh grid
    Z = nb_custom.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    # Plot decision boundaries
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)

    plt.title('PCA of Iris Dataset with Class PDFs and Decision Boundaries')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()


