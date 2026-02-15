import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Physical constants and parameters of the potential barrier
m = 9.1093837e-31
hbar = 1.054571817e-34 
qe = 1.60217663e-19

m1 = 0.5
Vpot = 30.0  # eV
ampl = 10.0
L = m1 * hbar / np.sqrt(2 * m * qe * Vpot)

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

# Define helper functions
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def one_hot_encode(y, num_classes):
    encoded = np.zeros((y.size, num_classes))
    encoded[np.arange(y.size), y] = 1
    return encoded

def cross_entropy_loss(predictions, targets):
    n_samples = targets.shape[0]
    log_likelihood = -np.log(predictions[np.arange(n_samples), targets])
    return np.mean(log_likelihood)

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess data
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0  # Flatten and normalize
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
y_train = y_train.flatten()
y_test = y_test.flatten()

# Configuration
input_size = 28 * 28  # Adjusted for MNIST input size
hidden_size = 512
output_size = 10  # Digits 0-9
learning_rate = 0.01
epochs = 100
batch_size = 64
gradient_clip_value = 5.0  # Gradient clipping threshold

# Initialize weights and biases
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

# Training loop
for epoch in range(epochs):
    # Shuffle data
    indices = np.random.permutation(x_train.shape[0])
    x_train = x_train[indices]
    y_train = y_train[indices]
    
    # Mini-batch training
    for i in range(0, x_train.shape[0], batch_size):
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        
        # Forward pass
        z1 = np.dot(x_batch, W1) + b1
        a1 = QT(L, z1 * ampl, Vpot)
        z2 = np.dot(a1, W2) + b2
        predictions = softmax(z2)
        
        # Compute loss
        y_batch_one_hot = one_hot_encode(y_batch, output_size)
        loss = cross_entropy_loss(predictions, y_batch)

        # Backward pass
        dz2 = predictions - y_batch_one_hot
        dW2 = np.dot(a1.T, dz2) / batch_size
        db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size
        
        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * diffQT(L, z1 * ampl, Vpot)
        dW1 = np.dot(x_batch.T, dz1) / batch_size
        db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size
        
        # Gradient clipping
        gradients = [dW1, db1, dW2, db2]
        total_grad_norm = np.sqrt(np.sum([np.sum(g**2) for g in gradients]))
        if total_grad_norm > gradient_clip_value:
            scaling_factor = gradient_clip_value / total_grad_norm
            gradients = [g * scaling_factor for g in gradients]
            dW1, db1, dW2, db2 = gradients
        
        # Update parameters
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
    
    # Evaluate training accuracy
    z1 = np.dot(x_train, W1) + b1
    a1 = QT(L, z1 * ampl, Vpot)
    z2 = np.dot(a1, W2) + b2
    train_predictions = np.argmax(softmax(z2), axis=1)
    train_accuracy = np.mean(train_predictions == y_train)
    
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Training Accuracy: {train_accuracy:.4f}")

# Evaluate on test data
z1 = np.dot(x_test, W1) + b1
a1 = QT(L, z1 * ampl, Vpot)
z2 = np.dot(a1, W2) + b2
test_predictions = np.argmax(softmax(z2), axis=1)
test_accuracy = np.mean(test_predictions == y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Display some classified images
labels_map = [str(i) for i in range(10)]  # Digits 0-9
x_test_images = x_test.reshape(-1, 28, 28)

plt.figure(figsize=(10, 10))
for i in range(16):
    idx = np.random.randint(0, len(x_test))
    plt.subplot(4, 4, i + 1)
    plt.imshow(x_test_images[idx], cmap='gray')
    plt.title(f"Predicted: {labels_map[test_predictions[idx]]}, True: {labels_map[y_test[idx]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
