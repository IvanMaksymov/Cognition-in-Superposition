import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
from scipy.spatial.distance import jensenshannon

# Map labels
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# Prepare arrays to store data
res_Q_iters10 = np.zeros((10, 10))
res_ReLU = np.zeros((10, 10))
jsd_values = np.zeros(10)  # Initialize JSD values
entropy_Q = np.zeros(10)    # Initialize entropy for Quantum iterations
entropy_ReLU = np.zeros(10)  # Initialize entropy for ReLU

# Labels for plotting
llabels_Q = np.arange(10)  # Numeric labels for x-axis

# Load data
for testLabel in range(10):
    res_Q_iters10[testLabel, :] = loadtxt(f"Quantum_iters10_{testLabel}.dat")
    res_ReLU[testLabel, :] = loadtxt(f"ReLU_{testLabel}.dat")

# Calculate JSD and SE values
for testLabel in range(10):
    # Normalize results to form valid probability distributions
    prob_Q = res_Q_iters10[testLabel, :] / np.sum(res_Q_iters10[testLabel, :])
    prob_ReLU = res_ReLU[testLabel, :] / np.sum(res_ReLU[testLabel, :])
    
    # Calculate JSD and Shannon entropy for each label
    jsd_values[testLabel] = jensenshannon(prob_Q, prob_ReLU)
    entropy_Q[testLabel] = -np.sum(prob_Q * np.log(prob_Q + 1e-10))  # Avoid log(0)
    entropy_ReLU[testLabel] = -np.sum(prob_ReLU * np.log(prob_ReLU + 1e-10))

# Plot 1: Combined Line Plot for JSD and SE Values
plt.figure(figsize=(10, 6))
plt.plot(llabels_Q, jsd_values, marker='o', label='JSD', color='purple')
plt.plot(llabels_Q, entropy_Q, marker='o', label='SE QT-NN', color='blue')
plt.plot(llabels_Q, entropy_ReLU, marker='o', label='SE Class.', color='orange')

# Set plot properties
plt.title('JSD and SE Comparison for Q Iterations and ReLU')
plt.xlabel('Class Labels')
plt.ylabel('Values')
plt.xticks(llabels_Q, labels_map.values(), rotation=45)
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()

# Save the combined plot as PDF
plt.savefig('JSD_SE_plot1.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Grouped Bar Chart for JSD and SE Values
x = np.arange(len(labels_map))  # label locations
width = 0.25  # width of the bars

# Create a figure for the bar chart
fig, ax = plt.subplots(figsize=(12, 6))

# Plot each model's JSD and SE values as grouped bars
bar1 = ax.bar(x - width, jsd_values, width, label='JSD', color='purple')
bar2 = ax.bar(x, entropy_Q, width, label='SE QT-NN', color='blue')
bar3 = ax.bar(x + width, entropy_ReLU, width, label='SE Class.', color='orange')

# Set plot properties
#ax.set_title('Grouped Bar Chart for JSD and SE Values')
ax.set_xlabel('Class Labels', fontsize=16)
ax.set_ylabel('Values', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(labels_map.values(), rotation=45)
ax.legend()
ax.grid(False)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Save the grouped bar chart as PDF
plt.tight_layout()
plt.savefig('JSD_SE_plot2.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Print the calculated values
print("\nJensen-Shannon Divergence Values:")
for i in range(10):
    print(f"JSD {i} ({labels_map[i]}): {jsd_values[i]:.3f}")

print("\nShannon Entropy Values:")
print(f"{'Class':<15}{'Entropy Q Iterations':<25}{'Entropy ReLU':<25}")
for i in range(10):
    print(f"{labels_map[i]:<15}{entropy_Q[i]:<25.4f}{entropy_ReLU[i]:<25.4f}")
