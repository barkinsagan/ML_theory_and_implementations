import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._unit_step_function(linear_output)

                update = self.learning_rate * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self._unit_step_function(linear_output)
        return y_predicted

    def _unit_step_function(self, x):
        return np.where(x >= 0, 1, 0)

if __name__ == "__main__":
    from sklearn.linear_model import Perceptron as SKPerceptron
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt

    # Create a synthetic dataset
    np.random.seed(0)
    X = np.random.randn(100, 2)
    y = np.array([1 if x[0] + x[1] > 0 else 0 for x in X])

    # Train custom Perceptron
    custom_perceptron = Perceptron(learning_rate=0.1, n_iters=1000)
    custom_perceptron.fit(X, y)
    y_pred_custom = custom_perceptron.predict(X)

    # Train scikit-learn Perceptron
    sk_perceptron = SKPerceptron(max_iter=1000, tol=1e-3, random_state=0)
    sk_perceptron.fit(X, y)
    y_pred_sk = sk_perceptron.predict(X)

    # Compare accuracy
    accuracy_custom = accuracy_score(y, y_pred_custom)
    accuracy_sk = accuracy_score(y, y_pred_sk)

    print(f"Custom Perceptron Accuracy: {accuracy_custom}")
    print(f"Scikit-learn Perceptron Accuracy: {accuracy_sk}")

    # Plot the dataset and decision boundaries
    plt.figure(figsize=(10, 5))

    # Plot the dataset
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k', s=50)
    plt.title("Dataset")

    # Plot decision boundary for custom Perceptron
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred_custom, cmap='bwr', edgecolor='k', s=50)
    plt.title("Custom Perceptron Decision Boundary")

    # Calculate and plot the decision boundary line
    x_values = np.array([min(X[:, 0]), max(X[:, 0])])
    y_values = -(custom_perceptron.weights[0] * x_values + custom_perceptron.bias) / custom_perceptron.weights[1]
    plt.plot(x_values, y_values, label='Decision Boundary', color='black')
    plt.legend()

    plt.tight_layout()
    plt.show()
