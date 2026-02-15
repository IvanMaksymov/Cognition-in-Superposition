import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from keras.datasets import fashion_mnist

# Load and preprocess the dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values
x_train, x_test = x_train.reshape(-1, 28 * 28), x_test.reshape(-1, 28 * 28)  # Flatten images
y_train_onehot = np.eye(10)[y_train]  # One-hot encode labels

# Bayesian Neural Network Parameters
input_dim = 28 * 28
hidden_dim = 512
output_dim = 10
num_samples = 50

# Initialize weights and biases
np.random.seed(42)
W1_mean, W1_std = np.random.randn(input_dim, hidden_dim), np.ones((input_dim, hidden_dim)) * 0.1
b1_mean, b1_std = np.random.randn(hidden_dim), np.ones(hidden_dim) * 0.1
W2_mean, W2_std = np.random.randn(hidden_dim, output_dim), np.ones((hidden_dim, output_dim)) * 0.1
b2_mean, b2_std = np.random.randn(output_dim), np.ones(output_dim) * 0.1

# Helper functions

# Physical constants and parameters of the potential barrier
m = 9.1093837e-31
hbar = 1.054571817e-34 
qe = 1.60217663e-19

m1 = 0.5
Vpot = 30.0 # eV
ampl = 10.0
L = m1*hbar/np.sqrt(2*m*qe*Vpot)

def diffQT(L, E, U0):
    # Precompute constants
    m_qe = m * qe
    hbar_squared = hbar ** 2

    # Create output array
    diffT = np.zeros_like(E)

    # Mask arrays for conditions
    mask_E_neg = E < 0.0
    mask_E_between = (E >= 0.0) & (E < U0)
    mask_E_greater = E > U0
    mask_E_equal = E == U0

    # For E < 0.0, diffT is already 0 (default)

    # For 0 <= E < U0
    if np.any(mask_E_between):
        E_between = E[mask_E_between]
        alpha = E_between - U0
        kappa1 = np.sqrt(-2.0 * m_qe * alpha) / hbar
        beta = U0**2 / (4.0 * E_between * alpha)
        delta1 = kappa1 * L
        T1 = 1.0 / (1.0 - beta * np.sinh(delta1) ** 2)
        diffT[mask_E_between] = -beta * (
            (np.sinh(delta1) ** 2 / E_between)
            + (np.sinh(delta1) ** 2 - delta1 * np.sinh(delta1) * np.cosh(delta1)) / alpha
        ) * T1**2

    # For E > U0
    if np.any(mask_E_greater):
        E_greater = E[mask_E_greater]
        alpha = E_greater - U0
        kappa = np.sqrt(2.0 * m_qe * alpha) / hbar
        beta = U0**2 / (4.0 * E_greater * alpha)
        delta = kappa * L
        T2 = 1.0 / (1.0 + beta * np.sin(delta) ** 2)
        diffT[mask_E_greater] = beta * (
            (np.sin(delta) ** 2 / E_greater)
            + (np.sin(delta) ** 2 - delta * np.sin(delta) * np.cos(delta)) / alpha
        ) * T2**2

    # For E == U0
    if np.any(mask_E_equal):
        diffT[mask_E_equal] = (
            4 * L**4 * U0 * m**2 * qe**2
            + 6 * L**2 * hbar_squared * m * qe
        ) / (
            3 * L**4 * U0**2 * m**2 * qe**2
            + 12 * L**2 * U0 * hbar_squared * m * qe
            + 12 * hbar_squared**2
        )

    return diffT

def QT(L, E, U0):
    # Precompute constants
    m_qe = m * qe
    hbar_squared = hbar ** 2

    # Create output array
    T = np.zeros_like(E)

    # Mask arrays for conditions
    mask_E_neg = E < 0.0
    mask_E_between = (E >= 0.0) & (E < U0)
    mask_E_greater = E > U0
    mask_E_equal = E == U0

    # For E < 0.0, T is already 0 (default)

    # For 0 <= E < U0
    if np.any(mask_E_between):
        E_between = E[mask_E_between]
        alpha = E_between - U0
        kappa1 = np.sqrt(-2.0 * m_qe * alpha) / hbar
        beta = U0**2 / (4.0 * E_between * alpha)
        T[mask_E_between] = 1.0 / (1.0 - beta * np.sinh(kappa1 * L) ** 2)

    # For E > U0
    if np.any(mask_E_greater):
        E_greater = E[mask_E_greater]
        alpha = E_greater - U0
        kappa = np.sqrt(2.0 * m_qe * alpha) / hbar
        beta = U0**2 / (4.0 * E_greater * alpha)
        T[mask_E_greater] = 1.0 / (1.0 + beta * np.sin(kappa * L) ** 2)

    # For E == U0
    if np.any(mask_E_equal):
        T[mask_E_equal] = 1.0 / (
            1.0 + m * L**2 * U0 * qe / (2.0 * hbar_squared)
        )

    return T

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def forward_pass(x, W1, b1, W2, b2):
    z1 = np.dot(x, W1) + b1
    a1 = QT(L, z1*ampl, Vpot)
    z2 = np.dot(a1, W2) + b2
    return softmax(z2)

def bayesian_predict(x, num_samples, W1_mean, W1_std, b1_mean, b1_std, W2_mean, W2_std, b2_mean, b2_std):
    predictions = np.zeros((num_samples, x.shape[0], output_dim))
    for i in range(num_samples):
        W1_sample = W1_mean + W1_std * np.random.randn(*W1_mean.shape)
        b1_sample = b1_mean + b1_std * np.random.randn(*b1_mean.shape)
        W2_sample = W2_mean + W2_std * np.random.randn(*W2_mean.shape)
        b2_sample = b2_mean + b2_std * np.random.randn(*b2_mean.shape)
        predictions[i] = forward_pass(x, W1_sample, b1_sample, W2_sample, b2_sample)
    return predictions.mean(axis=0)

# Training Function
def train_bnn(x_train, y_train, epochs, lr):
    global W1_mean, W1_std, b1_mean, b1_std, W2_mean, W2_std, b2_mean, b2_std
    loss_history = []
    accuracy_history = []
    
    for epoch in range(epochs):
        W1_sample = W1_mean + W1_std * np.random.randn(*W1_mean.shape)
        b1_sample = b1_mean + b1_std * np.random.randn(*b1_mean.shape)
        W2_sample = W2_mean + W2_std * np.random.randn(*W2_mean.shape)
        b2_sample = b2_mean + b2_std * np.random.randn(*b2_mean.shape)
        
        predictions = forward_pass(x_train, W1_sample, b1_sample, W2_sample, b2_sample)
        loss = -np.mean(np.sum(y_train * np.log(predictions + 1e-9), axis=1))
        loss_history.append(loss)
        
        grad_output = predictions - y_train
        grad_W2 = np.dot(QT(L, ampl*(np.dot(x_train, W1_sample) + b1_sample), Vpot).T, grad_output) / x_train.shape[0]
        grad_b2 = np.mean(grad_output, axis=0)
        grad_hidden = np.dot(grad_output, W2_sample.T) * diffQT(L, ampl*(np.dot(x_train, W1_sample) + b1_sample), Vpot)

        grad_W1 = np.dot(x_train.T, grad_hidden) / x_train.shape[0]
        grad_b1 = np.mean(grad_hidden, axis=0)
        
        W1_mean -= lr * grad_W1
        W1_std -= lr * grad_W1  # Update W1_std as well
        b1_mean -= lr * grad_b1
        b1_std -= lr * grad_b1  # Update b1_std as well
        W2_mean -= lr * grad_W2
        W2_std -= lr * grad_W2  # Update W2_std as well
        b2_mean -= lr * grad_b2
        b2_std -= lr * grad_b2  # Update b2_std as well
        
        train_acc = accuracy_score(np.argmax(y_train, axis=1), np.argmax(predictions, axis=1))
        accuracy_history.append(train_acc)    
        
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, Accuracy: {train_acc:.4f}")
    
    return loss_history, accuracy_history

# Visualize Results
def visualize_results(loss_history, accuracy_history, W1_std, W2_std):
    plt.figure(figsize=(12, 6))

    # Loss vs. Epochs
    plt.subplot(1, 3, 1)
    plt.plot(loss_history, label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    # Accuracy vs. Epochs
    plt.subplot(1, 3, 2)
    plt.plot(accuracy_history, label="Accuracy", color="green")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()

    # Weight Standard Deviations
    plt.subplot(1, 3, 3)
    plt.hist(W1_std.flatten(), bins=50, alpha=0.7, label="W1 std")
    plt.hist(W2_std.flatten(), bins=50, alpha=0.7, label="W2 std")
    plt.xlabel("Standard Deviation")
    plt.ylabel("Frequency")
    plt.title("Weight Standard Deviations")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Train the BNN
#loss_history, accuracy_history = train_bnn(x_train[:1000], y_train_onehot[:1000], epochs=500, lr=0.5)
loss_history, accuracy_history = train_bnn(x_train, y_train_onehot, epochs=1000, lr=0.5)

# Visualize the training process
visualize_results(loss_history, accuracy_history, W1_std, W2_std)

# Predict on test data
test_predictions = np.argmax(bayesian_predict(x_test, num_samples, W1_mean, W1_std, b1_mean, b1_std, W2_mean, W2_std, b2_mean, b2_std), axis=1)
test_accuracy = np.mean(test_predictions == y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Display some classified images
labels_map = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
x_test_images = x_test.reshape(-1, 28, 28)

plt.figure(figsize=(10, 10))
for i in range(16):
    idx = np.random.randint(0, len(x_test))
    plt.subplot(4, 4, i + 1)
    plt.imshow(x_test_images[idx], cmap='gray')
    plt.title(f"Pred: {labels_map[test_predictions[idx]]}\nTrue: {labels_map[y_test[idx]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
