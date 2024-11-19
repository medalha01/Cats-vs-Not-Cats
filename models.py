import h5py
import numpy as np
from typing import Tuple

# Data Loader Function
def load_data(train_path: str, test_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Loads the training and testing data from H5 files.

    Args:
        train_path (str): Path to the training data H5 file.
        test_path (str): Path to the testing data H5 file.

    Returns:
        Tuple containing:
            - train_x_orig (np.ndarray): Original training images.
            - train_y (np.ndarray): Training labels.
            - test_x_orig (np.ndarray): Original testing images.
            - test_y (np.ndarray): Testing labels.
            - classes (list): List of class labels.
    """
    with h5py.File(train_path, 'r') as train_dataset:
        train_x_orig = np.array(train_dataset['train_set_x'][:])  # train set features
        train_y = np.array(train_dataset['train_set_y'][:])       # train set labels

    with h5py.File(test_path, 'r') as test_dataset:
        test_x_orig = np.array(test_dataset['test_set_x'][:])     # test set features
        test_y = np.array(test_dataset['test_set_y'][:])          # test set labels
        classes = np.array(test_dataset['list_classes'][:])       # list of classes

    # Reshape labels to (1, number of examples)
    train_y = train_y.reshape((1, train_y.shape[0]))
    test_y = test_y.reshape((1, test_y.shape[0]))

    return train_x_orig, train_y, test_x_orig, test_y, classes

# Logistic Regression Model Class
class LogisticRegressionModel:
    def __init__(self, learning_rate: float = 0.005, num_iterations: int = 2000):
        """
        Initializes the logistic regression model.

        Args:
            learning_rate (float): Learning rate for gradient descent optimization.
            num_iterations (int): Maximum number of iterations for training.
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.w = None
        self.b = None

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Computes the sigmoid activation function.

        Args:
            z (np.ndarray): Input array.

        Returns:
            np.ndarray: Output after applying sigmoid function.
        """
        return 1 / (1 + np.exp(-z))

    def initialize_parameters(self, dim: int):
        """
        Initializes parameters w and b to zeros.

        Args:
            dim (int): Number of features.
        """
        self.w = np.zeros((dim, 1))
        self.b = 0.0

    def propagate(self, X: np.ndarray, Y: np.ndarray) -> Tuple[dict, float]:
        """
        Implements the cost function and its gradient.

        Args:
            X (np.ndarray): Input data of shape (num_features, num_examples).
            Y (np.ndarray): True labels of shape (1, num_examples).

        Returns:
            grads (dict): Gradients of the cost w.r.t. parameters.
            cost (float): Cost for logistic regression.
        """
        m = X.shape[1]

        # Forward propagation
        A = self.sigmoid(np.dot(self.w.T, X) + self.b)
        cost = (-1 / m) * np.sum(Y * np.log(A + 1e-8) + (1 - Y) * np.log(1 - A + 1e-8))  # Added epsilon for numerical stability

        # Backward propagation
        dw = (1 / m) * np.dot(X, (A - Y).T)
        db = (1 / m) * np.sum(A - Y)

        grads = {'dw': dw, 'db': db}

        return grads, cost

    def fit(self, X: np.ndarray, Y: np.ndarray, X_val: np.ndarray = None, Y_val: np.ndarray = None, print_cost: bool = False):
        """
        Trains the logistic regression model with early stopping.

        Args:
            X (np.ndarray): Training data.
            Y (np.ndarray): Training labels.
            X_val (np.ndarray): Validation data.
            Y_val (np.ndarray): Validation labels.
            print_cost (bool): Whether to print the cost every 100 iterations.
        """
        num_features = X.shape[0]
        self.initialize_parameters(num_features)

        costs = []
        val_costs = []
        patience = 100  # Early stopping patience
        best_val_cost = float('inf')
        patience_counter = 0

        for i in range(self.num_iterations):
            grads, cost = self.propagate(X, Y)

            # Update parameters
            self.w -= self.learning_rate * grads['dw']
            self.b -= self.learning_rate * grads['db']

            if X_val is not None and Y_val is not None:
                # Calculate validation cost
                A_val = self.sigmoid(np.dot(self.w.T, X_val) + self.b)
                val_cost = (-1 / X_val.shape[1]) * np.sum(Y_val * np.log(A_val + 1e-8) + (1 - Y_val) * np.log(1 - A_val + 1e-8))
                val_costs.append(val_cost)

                # Early stopping check
                if val_cost < best_val_cost:
                    best_val_cost = val_cost
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if print_cost:
                            print(f"Early stopping at iteration {i}")
                        break

            costs.append(cost)

            if print_cost and i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts labels for given input data.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Predicted labels.
        """
        A = self.sigmoid(np.dot(self.w.T, X) + self.b)
        predictions = (A > 0.5).astype(int)
        return predictions

# Shallow Neural Network Class
class ShallowNeuralNetwork:
    def __init__(self, n_x: int, n_h: int, n_y: int, learning_rate: float = 0.0075, num_iterations: int = 3000):
        """
        Initializes the shallow neural network model.

        Args:
            n_x (int): Size of the input layer.
            n_h (int): Size of the hidden layer.
            n_y (int): Size of the output layer.
            learning_rate (float): Learning rate for gradient descent optimization.
            num_iterations (int): Maximum number of iterations for training.
        """
        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.parameters = {}

    def initialize_parameters(self):
        """
        Initializes parameters with small random values.
        """
        np.random.seed(1)
        self.parameters['W1'] = np.random.randn(self.n_h, self.n_x) * 0.01
        self.parameters['b1'] = np.zeros((self.n_h, 1))
        self.parameters['W2'] = np.random.randn(self.n_y, self.n_h) * 0.01
        self.parameters['b2'] = np.zeros((self.n_y, 1))

    def sigmoid(self, Z: np.ndarray) -> np.ndarray:
        """
        Computes the sigmoid activation function.

        Args:
            Z (np.ndarray): Linear component of activation.

        Returns:
            np.ndarray: Activation after applying sigmoid function.
        """
        return 1 / (1 + np.exp(-Z))

    def tanh(self, Z: np.ndarray) -> np.ndarray:
        """
        Computes the hyperbolic tangent activation function.

        Args:
            Z (np.ndarray): Linear component of activation.

        Returns:
            np.ndarray: Activation after applying tanh function.
        """
        return np.tanh(Z)

    def forward_propagation(self, X: np.ndarray) -> dict:
        """
        Implements forward propagation.

        Args:
            X (np.ndarray): Input data.

        Returns:
            dict: Cache containing linear and activation values.
        """
        W1, b1 = self.parameters['W1'], self.parameters['b1']
        W2, b2 = self.parameters['W2'], self.parameters['b2']

        Z1 = np.dot(W1, X) + b1
        A1 = self.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.sigmoid(Z2)

        cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}

        return cache

    def compute_cost(self, A2: np.ndarray, Y: np.ndarray) -> float:
        """
        Computes the cross-entropy cost.

        Args:
            A2 (np.ndarray): The sigmoid output of the second activation.
            Y (np.ndarray): True labels.

        Returns:
            float: Cross-entropy cost.
        """
        m = Y.shape[1]
        cost = (-1 / m) * np.sum(Y * np.log(A2 + 1e-8) + (1 - Y) * np.log(1 - A2 + 1e-8))  # Added epsilon for numerical stability
        cost = np.squeeze(cost)  # Ensures cost is a scalar.
        return cost

    def backward_propagation(self, X: np.ndarray, Y: np.ndarray, cache: dict) -> dict:
        """
        Implements backward propagation.

        Args:
            X (np.ndarray): Input data.
            Y (np.ndarray): True labels.
            cache (dict): Cache from forward propagation.

        Returns:
            dict: Gradients w.r.t. parameters.
        """
        m = X.shape[1]
        W2 = self.parameters['W2']

        A1, A2 = cache['A1'], cache['A2']

        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

        return grads

    def update_parameters(self, grads: dict):
        """
        Updates parameters using gradient descent.

        Args:
            grads (dict): Gradients w.r.t. parameters.
        """
        self.parameters['W1'] -= self.learning_rate * grads['dW1']
        self.parameters['b1'] -= self.learning_rate * grads['db1']
        self.parameters['W2'] -= self.learning_rate * grads['dW2']
        self.parameters['b2'] -= self.learning_rate * grads['db2']

    def fit(self, X: np.ndarray, Y: np.ndarray, X_val: np.ndarray = None, Y_val: np.ndarray = None, print_cost: bool = False):
        """
        Trains the neural network with early stopping.

        Args:
            X (np.ndarray): Training data.
            Y (np.ndarray): Training labels.
            X_val (np.ndarray): Validation data.
            Y_val (np.ndarray): Validation labels.
            print_cost (bool): Whether to print the cost every 100 iterations.
        """
        self.initialize_parameters()

        costs = []
        val_costs = []
        patience = 100  # Early stopping patience
        best_val_cost = float('inf')
        patience_counter = 0

        for i in range(self.num_iterations):
            cache = self.forward_propagation(X)
            cost = self.compute_cost(cache['A2'], Y)
            grads = self.backward_propagation(X, Y, cache)
            self.update_parameters(grads)

            if X_val is not None and Y_val is not None:
                # Validation step
                cache_val = self.forward_propagation(X_val)
                val_cost = self.compute_cost(cache_val['A2'], Y_val)
                val_costs.append(val_cost)

                # Early stopping check
                if val_cost < best_val_cost:
                    best_val_cost = val_cost
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if print_cost:
                            print(f"Early stopping at iteration {i}")
                        break

            costs.append(cost)

            if print_cost and i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts labels for given input data.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Predicted labels.
        """
        cache = self.forward_propagation(X)
        predictions = (cache['A2'] > 0.5).astype(int)
        return predictions

# Deep Neural Network Class
class DeepNeuralNetwork:
    def __init__(self, layer_dims: list, learning_rate: float = 0.0075, num_iterations: int = 3000):
        """
        Initializes the deep neural network model.

        Args:
            layer_dims (list): List containing the dimensions of each layer.
            learning_rate (float): Learning rate for gradient descent optimization.
            num_iterations (int): Maximum number of iterations for training.
        """
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.parameters = {}

    def initialize_parameters(self):
        """
        Initializes parameters with He initialization.
        """
        np.random.seed(1)
        L = len(self.layer_dims)
        for l in range(1, L):
            self.parameters['W' + str(l)] = np.random.randn(
                self.layer_dims[l], self.layer_dims[l - 1]) * np.sqrt(2 / self.layer_dims[l - 1])
            self.parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))

    def linear_activation_forward(self, A_prev, W, b, activation):
        """
        Implements the forward propagation for a single layer.

        Args:
            A_prev: Activations from previous layer.
            W: Weights matrix.
            b: Bias vector.
            activation: Activation function to use ("relu" or "sigmoid").

        Returns:
            A: Output of the activation function.
            cache: Tuple containing linear cache and activation cache.
        """
        Z = np.dot(W, A_prev) + b
        linear_cache = (A_prev, W, b)

        if activation == "relu":
            A = np.maximum(0, Z)
        elif activation == "sigmoid":
            A = 1 / (1 + np.exp(-Z))
        else:
            raise Exception("Unsupported activation function")

        activation_cache = Z
        cache = (linear_cache, activation_cache)

        return A, cache

    def forward_propagation(self, X):
        """
        Implements forward propagation for the deep network.

        Args:
            X: Input data.

        Returns:
            AL: Last post-activation value.
            caches: List of caches from each layer.
        """
        caches = []
        A = X
        L = len(self.layer_dims) - 1

        # Hidden layers
        for l in range(1, L):
            A_prev = A
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            A, cache = self.linear_activation_forward(A_prev, W, b, activation='relu')
            caches.append(cache)

        # Output layer
        W = self.parameters['W' + str(L)]
        b = self.parameters['b' + str(L)]
        AL, cache = self.linear_activation_forward(A, W, b, activation='sigmoid')
        caches.append(cache)

        return AL, caches

    def compute_cost(self, AL, Y):
        """
        Computes the cross-entropy cost.

        Args:
            AL: Probability vector corresponding to label predictions.
            Y: True labels.

        Returns:
            cost: Cross-entropy cost.
        """
        m = Y.shape[1]
        cost = (-1 / m) * np.sum(Y * np.log(AL + 1e-8) + (1 - Y) * np.log(1 - AL + 1e-8))
        cost = np.squeeze(cost)
        return cost

    def linear_activation_backward(self, dA, cache, activation):
        """
        Implements backward propagation for a single layer.

        Args:
            dA: Post-activation gradient.
            cache: Tuple of values (linear_cache, activation_cache).
            activation: Activation function used ("relu" or "sigmoid").

        Returns:
            dA_prev, dW, db: Gradients w.r.t. previous layer activations, weights, and biases.
        """
        linear_cache, activation_cache = cache
        A_prev, W, b = linear_cache
        Z = activation_cache

        m = A_prev.shape[1]

        if activation == "relu":
            dZ = np.array(dA, copy=True)
            dZ[Z <= 0] = 0
        elif activation == "sigmoid":
            s = 1 / (1 + np.exp(-Z))
            dZ = dA * s * (1 - s)
        else:
            raise Exception("Unsupported activation function")

        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def backward_propagation(self, AL, Y, caches):
        """
        Implements backward propagation for the deep network.

        Args:
            AL: Probability vector from forward propagation.
            Y: True labels.
            caches: List of caches from each layer.

        Returns:
            grads: Dictionary with gradients.
        """
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        # Initialize backpropagation
        dAL = -(np.divide(Y, AL + 1e-8) - np.divide(1 - Y, 1 - AL + 1e-8))

        # Output layer gradients
        current_cache = caches[L - 1]
        grads['dA' + str(L - 1)], grads['dW' + str(L)], grads['db' + str(L)] = self.linear_activation_backward(
            dAL, current_cache, activation='sigmoid')

        # Loop from L-2 to 0
        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(
                grads['dA' + str(l + 1)], current_cache, activation='relu')
            grads['dA' + str(l)] = dA_prev_temp
            grads['dW' + str(l + 1)] = dW_temp
            grads['db' + str(l + 1)] = db_temp

        return grads

    def update_parameters(self, grads):
        """
        Updates parameters using gradient descent.

        Args:
            grads: Dictionary with gradients.
        """
        L = len(self.layer_dims) - 1  # Number of layers

        for l in range(1, L + 1):
            self.parameters['W' + str(l)] -= self.learning_rate * grads['dW' + str(l)]
            self.parameters['b' + str(l)] -= self.learning_rate * grads['db' + str(l)]

    def fit(self, X, Y, X_val=None, Y_val=None, print_cost=False):
        """
        Trains the deep neural network with early stopping.

        Args:
            X: Training data.
            Y: Training labels.
            X_val: Validation data.
            Y_val: Validation labels.
            print_cost: Whether to print the cost every 100 iterations.
        """
        self.initialize_parameters()

        costs = []
        val_costs = []
        patience = 100  # Early stopping patience
        best_val_cost = float('inf')
        patience_counter = 0

        for i in range(self.num_iterations):
            # Forward propagation
            AL, caches = self.forward_propagation(X)

            # Compute cost
            cost = self.compute_cost(AL, Y)

            # Backward propagation
            grads = self.backward_propagation(AL, Y, caches)

            # Update parameters
            self.update_parameters(grads)

            if X_val is not None and Y_val is not None:
                # Validation step
                AL_val, _ = self.forward_propagation(X_val)
                val_cost = self.compute_cost(AL_val, Y_val)
                val_costs.append(val_cost)

                # Early stopping check
                if val_cost < best_val_cost:
                    best_val_cost = val_cost
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if print_cost:
                            print(f"Early stopping at iteration {i}")
                        break

            # Print cost
            if print_cost and i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")

    def predict(self, X):
        """
        Predicts labels for given input data.

        Args:
            X: Input data.

        Returns:
            np.ndarray: Predicted labels.
        """
        AL, _ = self.forward_propagation(X)
        predictions = (AL > 0.5).astype(int)
        return predictions

# CNN Model Function
def create_cnn_model(input_shape):
    """
    Creates and returns a CNN model with early stopping.

    Args:
        input_shape (tuple): Shape of the input data (height, width, channels).

    Returns:
        model (tf.keras.Model): Compiled CNN model.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential()

    # Convolutional Layer 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional Layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten and Dense Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))

    # Output Layer
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compile Model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

