import matplotlib.pyplot as plt
import numpy as np

# Constants
pi = np.pi
melec = 9.1093837e-31
hbar = 1.054571817e-34
elcharge = 1.60217663e-19
kf = 6.241509e18
DomainSize = 4001
kstart = DomainSize // 2
kcentre = kstart // 2
NSTEPS = 100000
STEPPLOT = 5000
ddx = 0.1E-11
ra = 0.125
lambda_ = 1.6e-10
sigma = 1.6e-10
U0 = 600.0  # eV

# Arrays
PsiReal = np.zeros(DomainSize)
PsiImag = np.zeros(DomainSize)
vp = np.zeros(DomainSize)

# Variables
dt = 0.25 * (melec / hbar) * (ddx ** 2)
L = DomainSize * ddx

# Generate the potential well
for k in range(DomainSize - 1):
    xx = k * ddx
    Vpot = (4.0 * U0 / (L ** 2)) * (xx ** 2) - (4.0 * U0 / L) * xx
    vp[k] = Vpot * elcharge

# Add a barrier to the potential well
for k in range(kstart - 10, kstart + 11):
    vp[k] += 400.0 * elcharge

# Initialise the wavefunction
ptot = 0.0
for k in range(1, DomainSize - 2):
    PsiReal[k] = np.cos(2.0 * pi * ddx * (k - kcentre) / lambda_) * \
                 np.exp(-0.5 * ((ddx * (k - kcentre)) / sigma) ** 2)
    PsiImag[k] = np.sin(2.0 * pi * ddx * (k - kcentre) / lambda_) * \
                 np.exp(-0.5 * ((ddx * (k - kcentre)) / sigma) ** 2)
    ptot += (PsiReal[k] ** 2 + PsiImag[k] ** 2)

# Normalisation of the wavefunction
norm_factor = np.sqrt(ptot)
PsiReal[1:DomainSize - 2] /= norm_factor
PsiImag[1:DomainSize - 2] /= norm_factor

# Time evolution 
for n in range(NSTEPS - 1):
    PsiReal[1:-1] = PsiReal[1:-1] - ra * (PsiImag[2:] - 2.0 * PsiImag[1:-1] + PsiImag[:-2]) + \
                     (dt / hbar) * vp[1:-1] * PsiImag[1:-1]
    
    PsiImag[1:-1] = PsiImag[1:-1] + ra * (PsiReal[2:] - 2.0 * PsiReal[1:-1] + PsiReal[:-2]) - \
                     (dt / hbar) * vp[1:-1] * PsiReal[1:-1]

# Energy calculation
KeReal = KeImag = Pe = 0.0
for k in range(1, DomainSize - 2):
    LapReal = PsiReal[k + 1] - 2.0 * PsiReal[k] + PsiReal[k - 1]
    LapImag = PsiImag[k + 1] - 2.0 * PsiImag[k] + PsiImag[k - 1]

    KeReal += PsiReal[k] * LapReal + PsiImag[k] * LapImag
    KeImag += PsiReal[k] * LapImag - PsiImag[k] * LapReal
    Pe += vp[k] * (PsiReal[k] ** 2 + PsiImag[k] ** 2)

kine = 0.5 * (hbar / melec) * (hbar / (ddx ** 2)) * np.sqrt(KeReal ** 2 + KeImag ** 2)
print(f"Kinetic Energy (ke): {kine * kf:.16e}, Potential Energy (pe): {Pe * kf:.16e}")

# Compute values
x_values = np.arange(DomainSize) * ddx  # Convert indices to positions
vp_scaled = vp / elcharge               # Potential in eV
wavefunction_amplitude = np.sqrt(PsiReal**2 + PsiImag**2)  # Combined amplitude

# Create figure and axis
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot the wavefunction (left y-axis)
ax1.plot(x_values, wavefunction_amplitude, label="Wavefunction Amplitude", color="blue")
ax1.set_xlabel("Position (m)")
ax1.set_ylabel("Wavefunction Amplitude", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")
ax1.grid(True)

# Plot the potential (right y-axis)
ax2 = ax1.twinx()
ax2.plot(x_values, vp_scaled, label="Potential (eV)", color="green", linestyle="--")
ax2.set_ylabel("Potential (eV)", color="green")
ax2.tick_params(axis="y", labelcolor="green")

# Title and layout
plt.title("Wavefunction and Potential")
fig.tight_layout()
plt.show()


