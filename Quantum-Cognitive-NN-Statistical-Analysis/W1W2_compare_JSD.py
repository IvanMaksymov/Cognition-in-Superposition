import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon

# Function to load data and ensure the same size
def load_and_prepare_data(file1, file2):
    data1 = np.loadtxt(file1)
    data2 = np.loadtxt(file2)

    if data1.size != data2.size:
        min_size = min(data1.size, data2.size)
        data1 = data1[:min_size]
        data2 = data2[:min_size]

    return data1, data2

# Function to calculate and return PDFs and JSD
def calculate_js_divergence(data1, data2):
    kde1 = gaussian_kde(data1)
    kde2 = gaussian_kde(data2)

    x_values = np.linspace(min(np.min(data1), np.min(data2)), 
                           max(np.max(data1), np.max(data2)), 1000)
    
    pdf1 = kde1(x_values)
    pdf2 = kde2(x_values)

    # Prevent division by zero or log of zero
    pdf1_safe = np.maximum(pdf1, 1e-10)
    pdf2_safe = np.maximum(pdf2, 1e-10)
    
    js_divergence = jensenshannon(pdf1_safe, pdf2_safe) ** 2

    return x_values, pdf1, pdf2, js_divergence

# Step 1: Load data for classical and quantum models
w1_init_data, w1_training_data = load_and_prepare_data("Relu_W1_initial_normal.dat", "Relu_W1_weight_after_training.dat")
quantum_w1_init_data, quantum_w1_training_data = load_and_prepare_data("Quantum_W1_initial_normal.dat", "Quantum_W1_weight_after_training.dat")
w2_init_data, w2_training_data = load_and_prepare_data("Relu_W2_initial_normal.dat", "Relu_W2_weight_after_training.dat")
quantum_w2_init_data, quantum_w2_training_data = load_and_prepare_data("Quantum_W2_initial_normal.dat", "Quantum_W2_weight_after_training.dat")

# Step 2: Calculate the PDFs and JSD for each comparison
x_w1_classical, pdf_w1_classical_init, pdf_w1_classical_train, js_w1_classical = calculate_js_divergence(w1_init_data, w1_training_data)
x_w1_quantum, pdf_w1_quantum_init, pdf_w1_quantum_train, js_w1_quantum = calculate_js_divergence(quantum_w1_init_data, quantum_w1_training_data)
x_w1_comparison, pdf_w1_classical_train, pdf_w1_quantum_train, js_w1_comparison = calculate_js_divergence(w1_training_data, quantum_w1_training_data)

x_w2_classical, pdf_w2_classical_init, pdf_w2_classical_train, js_w2_classical = calculate_js_divergence(w2_init_data, w2_training_data)
x_w2_quantum, pdf_w2_quantum_init, pdf_w2_quantum_train, js_w2_quantum = calculate_js_divergence(quantum_w2_init_data, quantum_w2_training_data)
x_w2_comparison, pdf_w2_classical_train, pdf_w2_quantum_train, js_w2_comparison = calculate_js_divergence(w2_training_data, quantum_w2_training_data)

# Step 3: Create a figure with 2 rows and 3 columns
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Top row - W1
axes[0, 0].plot(x_w1_classical, pdf_w1_classical_init, label='Init. Class.', color='blue')
axes[0, 0].plot(x_w1_classical, pdf_w1_classical_train, label='Trained Class.', color='orange')
axes[0, 0].fill_between(x_w1_classical, pdf_w1_classical_init, pdf_w1_classical_train, 
                        where=(pdf_w1_classical_init > pdf_w1_classical_train), color='green', alpha=0.5)
axes[0, 0].set_title(f"Class. W1 Init. vs Class. W1 Trained: {js_w1_classical:.4f}", fontsize=18)
axes[0, 0].legend(fontsize=16, loc = "upper right")

axes[0, 1].plot(x_w1_quantum, pdf_w1_quantum_init, label='Init. QT-NN', color='blue')
axes[0, 1].plot(x_w1_quantum, pdf_w1_quantum_train, label='Trained QT-NN', color='orange')
axes[0, 1].fill_between(x_w1_quantum, pdf_w1_quantum_init, pdf_w1_quantum_train, 
                        where=(pdf_w1_quantum_init > pdf_w1_quantum_train), color='green', alpha=0.5)
axes[0, 1].set_title(f"QT-NN W1 Init. vs QT-NN W1 Trained: {js_w1_quantum:.4f}", fontsize=18)
axes[0, 1].legend(fontsize=16, loc = "upper right")

axes[0, 2].plot(x_w1_comparison, pdf_w1_classical_train, label='Trained Class.', color='blue')
axes[0, 2].plot(x_w1_comparison, pdf_w1_quantum_train, label='Trained QT-NN', color='orange')
axes[0, 2].fill_between(x_w1_comparison, pdf_w1_classical_train, pdf_w1_quantum_train, 
                        where=(pdf_w1_classical_train > pdf_w1_quantum_train), color='green', alpha=0.5)
axes[0, 2].set_title(f"Class. W1 Trained vs QT-NN W1 Trained: {js_w1_comparison:.4f}", fontsize=18)
axes[0, 2].legend(fontsize=16, loc = "upper right")

# Bottom row - W2
axes[1, 0].plot(x_w2_classical, pdf_w2_classical_init, label='Init. Class.', color='blue')
axes[1, 0].plot(x_w2_classical, pdf_w2_classical_train, label='Trained Class.', color='orange')
axes[1, 0].fill_between(x_w2_classical, pdf_w2_classical_init, pdf_w2_classical_train, 
                        where=(pdf_w2_classical_init > pdf_w2_classical_train), color='green', alpha=0.5)
axes[1, 0].set_title(f"Class. W2 Init. vs Class. W2 Trained: {js_w2_classical:.4f}", fontsize=18)
axes[1, 0].legend(fontsize=16, loc = "upper right")

axes[1, 1].plot(x_w2_quantum, pdf_w2_quantum_init, label='Init. QT-NN', color='blue')
axes[1, 1].plot(x_w2_quantum, pdf_w2_quantum_train, label='Trained QT-NN', color='orange')
axes[1, 1].fill_between(x_w2_quantum, pdf_w2_quantum_init, pdf_w2_quantum_train, 
                        where=(pdf_w2_quantum_init > pdf_w2_quantum_train), color='green', alpha=0.5)
axes[1, 1].set_title(f"QT-NN W2 Init. vs QT-NN W2 Trained: {js_w2_quantum:.4f}", fontsize=18)
axes[1, 1].legend(fontsize=16, loc = "upper right")

axes[1, 2].plot(x_w2_comparison, pdf_w2_classical_train, label='Trained Class.', color='blue')
axes[1, 2].plot(x_w2_comparison, pdf_w2_quantum_train, label='Trained QT-NN', color='orange')
axes[1, 2].fill_between(x_w2_comparison, pdf_w2_classical_train, pdf_w2_quantum_train, 
                        where=(pdf_w2_classical_train > pdf_w2_quantum_train), color='green', alpha=0.5)
axes[1, 2].set_title(f"Class. W2 Trained vs QT-NN W2 Trained: {js_w2_comparison:.4f}", fontsize=18)
axes[1, 2].legend(fontsize=16, loc = "upper right")


axes[0, 0].tick_params(axis='both', labelsize=18)
axes[0, 1].tick_params(axis='both', labelsize=18)
axes[0, 2].tick_params(axis='both', labelsize=18)
axes[1, 0].tick_params(axis='both', labelsize=18)
axes[1, 1].tick_params(axis='both', labelsize=18)
axes[1, 2].tick_params(axis='both', labelsize=18)

# Final plot adjustments
plt.tight_layout()
plt.savefig("JSD_comparison_plot.pdf", dpi=300, bbox_inches='tight')
plt.savefig("JSD_comparison_plot.png", dpi=300, bbox_inches='tight')
plt.show()

# Step 4: Print the JSD values for all comparisons
print(f"JSD Classical W1 Init vs W1 Train: {js_w1_classical:.8f}")
print(f"JSD Quantum W1 Init vs W1 Train: {js_w1_quantum:.8f}")
print(f"JSD W1 Classical vs Quantum after training: {js_w1_comparison:.8f}")
print(f"JSD Classical W2 Init vs W2 Train: {js_w2_classical:.8f}")
print(f"JSD Quantum W2 Init vs W2 Train: {js_w2_quantum:.8f}")
print(f"JSD W2 Classical vs Quantum after training: {js_w2_comparison:.8f}")
