import numpy as np
from models import load_data, LogisticRegressionModel, ShallowNeuralNetwork, DeepNeuralNetwork, create_cnn_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
def train_logistic_regression():
    # Load and preprocess data
    train_x_orig, train_y, test_x_orig, test_y, _ = load_data('train_catvnoncat.h5', 'test_catvnoncat.h5')

    # Split data into training and validation sets
    val_split = 0.1
    num_train = int(train_x_orig.shape[0] * (1 - val_split))
    train_indices = np.arange(num_train)
    val_indices = np.arange(num_train, train_x_orig.shape[0])

    # Flatten images
    train_x_flatten = train_x_orig[train_indices].reshape(num_train, -1).T
    val_x_flatten = train_x_orig[val_indices].reshape(len(val_indices), -1).T  # Corrected line
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x_flatten.T).T
    val_x = scaler.transform(val_x_flatten.T).T
    test_x = scaler.transform(test_x_flatten.T).T

    train_y_split = train_y[:, train_indices]
    val_y_split = train_y[:, val_indices]

    # Train model
    model = LogisticRegressionModel(learning_rate=0.005, num_iterations=2000)
    model.fit(train_x, train_y_split, X_val=val_x, Y_val=val_y_split, print_cost=True)

    # Evaluate model
    train_pred = model.predict(train_x)
    test_pred = model.predict(test_x)

    train_accuracy = 100 - np.mean(np.abs(train_pred - train_y_split)) * 100
    test_accuracy = 100 - np.mean(np.abs(test_pred - test_y)) * 100

    print(f"Logistic Regression Training accuracy: {train_accuracy}%")
    print(f"Logistic Regression Test accuracy: {test_accuracy}%")

    # Confusion Matrix
    cm = confusion_matrix(test_y.flatten(), test_pred.flatten())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Logistic Regression Confusion Matrix")
    plt.show()

def train_shallow_neural_network():
    # Load and preprocess data
    train_x_orig, train_y, test_x_orig, test_y, _ = load_data('train_catvnoncat.h5', 'test_catvnoncat.h5')

    # Split data into training and validation sets
    val_split = 0.1
    num_train = int(train_x_orig.shape[0] * (1 - val_split))
    train_indices = np.arange(num_train)
    val_indices = np.arange(num_train, train_x_orig.shape[0])

    # Flatten images
    train_x_flatten = train_x_orig[train_indices].reshape(num_train, -1).T
    val_x_flatten = train_x_orig[val_indices].reshape(len(val_indices), -1).T  # Corrected line
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x_flatten.T).T
    val_x = scaler.transform(val_x_flatten.T).T
    test_x = scaler.transform(test_x_flatten.T).T

    train_y_split = train_y[:, train_indices]
    val_y_split = train_y[:, val_indices]

    # Get layer sizes
    n_x = train_x.shape[0]
    n_h = 7  # Hyperparameter to be tuned
    n_y = 1

    # Train model
    model = ShallowNeuralNetwork(n_x, n_h, n_y, learning_rate=0.0075, num_iterations=2500)
    model.fit(train_x, train_y_split, X_val=val_x, Y_val=val_y_split, print_cost=True)

    # Evaluate model
    train_pred = model.predict(train_x)
    test_pred = model.predict(test_x)

    train_accuracy = 100 - np.mean(np.abs(train_pred - train_y_split)) * 100
    test_accuracy = 100 - np.mean(np.abs(test_pred - test_y)) * 100

    print(f"Shallow Neural Network Training accuracy: {train_accuracy}%")
    print(f"Shallow Neural Network Test accuracy: {test_accuracy}%")

    # Confusion Matrix
    cm = confusion_matrix(test_y.flatten(), test_pred.flatten())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Shallow Neural Network Confusion Matrix")
    plt.show()

def train_deep_neural_network():
    # Load and preprocess data
    train_x_orig, train_y, test_x_orig, test_y, _ = load_data('train_catvnoncat.h5', 'test_catvnoncat.h5')

    # Split data into training and validation sets
    val_split = 0.1
    num_train = int(train_x_orig.shape[0] * (1 - val_split))
    train_indices = np.arange(num_train)
    val_indices = np.arange(num_train, train_x_orig.shape[0])

    # Flatten images
    train_x_flatten = train_x_orig[train_indices].reshape(num_train, -1).T
    val_x_flatten = train_x_orig[val_indices].reshape(len(val_indices), -1).T  # Corrected line
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x_flatten.T).T
    val_x = scaler.transform(val_x_flatten.T).T
    test_x = scaler.transform(test_x_flatten.T).T

    train_y_split = train_y[:, train_indices]
    val_y_split = train_y[:, val_indices]

    # Define layer dimensions
    layer_dims = [train_x.shape[0], 20, 7, 5, 1]  # Example architecture

    # Train model
    model = DeepNeuralNetwork(layer_dims, learning_rate=0.0075, num_iterations=2500)
    model.fit(train_x, train_y_split, X_val=val_x, Y_val=val_y_split, print_cost=True)

    # Evaluate model
    train_pred = model.predict(train_x)
    test_pred = model.predict(test_x)

    train_accuracy = 100 - np.mean(np.abs(train_pred - train_y_split)) * 100
    test_accuracy = 100 - np.mean(np.abs(test_pred - test_y)) * 100

    print(f"Deep Neural Network Training accuracy: {train_accuracy}%")
    print(f"Deep Neural Network Test accuracy: {test_accuracy}%")

    # Confusion Matrix
    cm = confusion_matrix(test_y.flatten(), test_pred.flatten())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Deep Neural Network Confusion Matrix")
    plt.show()

def train_cnn():

    from tensorflow.keras.callbacks import EarlyStopping

    # Load and preprocess data
    train_x_orig, train_y, test_x_orig, test_y, _ = load_data('train_catvnoncat.h5', 'test_catvnoncat.h5')

    # Normalize pixel values
    train_x = train_x_orig / 255.0
    test_x = test_x_orig / 255.0

    # Split data into training and validation sets
    val_split = 0.1
    num_train = int(train_x.shape[0] * (1 - val_split))
    train_x_split = train_x[:num_train]
    val_x_split = train_x[num_train:]
    train_y_split = train_y[:, :num_train].T
    val_y_split = train_y[:, num_train:].T

    # Reshape labels for TensorFlow
    test_y_tf = test_y.T

    # Create and train model
    input_shape = train_x.shape[1:]
    model = create_cnn_model(input_shape)

    # Early Stopping Callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    model.fit(train_x_split, train_y_split, epochs=50, batch_size=32,
              validation_data=(val_x_split, val_y_split), callbacks=[early_stopping], verbose=1)

    # Evaluate model
    train_loss, train_accuracy = model.evaluate(train_x_split, train_y_split, verbose=0)
    test_loss, test_accuracy = model.evaluate(test_x, test_y_tf, verbose=0)

    print(f"CNN Training accuracy: {train_accuracy * 100}%")
    print(f"CNN Test accuracy: {test_accuracy * 100}%")

    # Confusion Matrix
    test_pred_prob = model.predict(test_x)
    test_pred = (test_pred_prob > 0.5).astype(int)
    cm = confusion_matrix(test_y.flatten(), test_pred.flatten())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("CNN Confusion Matrix")
    plt.show()

def run_all_models():
    train_logistic_regression()
    train_shallow_neural_network()
    train_deep_neural_network()
    train_cnn()

if __name__ == "__main__":
    print("Choose a model to train:")
    print("1. Logistic Regression")
    print("2. Shallow Neural Network")
    print("3. Deep Neural Network")
    print("4. Convolutional Neural Network")
    print("5. Run All Models")

    choice = input("Enter the number corresponding to your choice: ")

    if choice == '1':
        train_logistic_regression()
    elif choice == '2':
        train_shallow_neural_network()
    elif choice == '3':
        train_deep_neural_network()
    elif choice == '4':
        train_cnn()
    elif choice == '5':
        run_all_models()
    else:
        print("Invalid choice. Please select a number between 1 and 5.")

