import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt

# ========================================
# Parameters
# ========================================
fnt_size = 22
dz = 2e-10  # discretization step in meters
n = 30      # number of eigenstates to compute
mass = 0.067
barrier = 0.1  # depth of potential wells in eV
super_lattice_N = 10
Nkv = 20  # number of k-points (use 50 for full accuracy)

# ========================================
# Function: Sch1D (finite-difference Schrödinger solver)
# ========================================
def Sch1D(z, V0, Mass, n, kvec, super_lattice_N):
    h = 6.62606896e-34
    hbar = h / (2 * np.pi)
    e = 1.602176487e-19
    m0 = 9.10938188e-31

    Nz = len(z)
    dz_val = z[1] - z[0]
    period = Nz * dz_val * (super_lattice_N + 1)

    # Second derivative matrix with Bloch boundary conditions
    diagonals = [-2*np.ones(Nz), np.ones(Nz-1), np.ones(Nz-1)]
    offsets = [0, -1, 1]
    DZ2 = diags(diagonals, offsets, shape=(Nz, Nz)).tocsr()

    # Apply Bloch boundary conditions
    DZ2[0, -1] = np.exp(-1j * kvec * period)
    DZ2[-1, 0] = np.exp(1j * kvec * period)
    DZ2 /= dz_val ** 2

    # Hamiltonian
    H = -hbar**2 / (2 * m0 * Mass) * DZ2 + diags(V0 * e)
    H = H.tocsr()

    # Solve for n lowest eigenvalues
    Energy, psi = eigs(H, k=n, which='SM')  # SM = smallest magnitude

    E = np.real(Energy) / e  # Convert to eV

    # Sort energies ascending
    idx = np.argsort(E)
    E = E[idx]
    psi = psi[:, idx]

    # Normalize wavefunctions
    for i in range(n):
        norm = np.trapz(np.abs(psi[:, i])**2, z)
        if norm > 0:
            psi[:, i] /= np.sqrt(norm)

    return E, psi

# ========================================
# Build superlattice: Perfect case
# ========================================
periodic_structure = np.array(
    [20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
     19.999, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
) * 1e-9

wells = np.array(
    [0.7, 0, 0.7, 0, 0.7, 0, 0.7, 0, 0.7, 0,
     0.6999, 0, 0.7, 0, 0.7, 0, 0.7, 0, 0.7, 0, 0.7]
) * barrier

z, V0 = np.array([]), np.array([])
for i, width in enumerate(periodic_structure):
    zv = np.arange(0 if i == 0 else z[-1] + dz, (z[-1] + width) + dz if i > 0 else width + dz, dz)
    z = np.concatenate([z, zv]) if i > 0 else zv
    V0 = np.concatenate([V0, np.full_like(zv, wells[i])]) if i > 0 else np.full_like(zv, wells[i])

Nz = len(z)
period = Nz * dz * (super_lattice_N + 1)

# ========================================
# Plotting setup
# ========================================
fig = plt.figure(figsize=(20, 8))
ax1 = plt.subplot(1, 9, 1)  # Potential
ax2 = plt.subplot(1, 9, (2, 6))  # Band structure

# ========================================
# Perfect superlattice (red)
# ========================================
for ii in range(-Nkv + 1, Nkv + 1):
    kvec = ((ii - 1) / Nkv) * (np.pi / period)
    E1, _ = Sch1D(z, V0, mass, n, kvec, super_lattice_N)
    E1 = E1[E1 < max(V0) + 0.1]

    # Left panel
    for E_val in E1:
        ax1.hlines(E_val, 0, 1, color='r', linewidth=1)
    # Right panel
    for E_val in E1:
        ax2.hlines(E_val, ii - 1, ii, color='r', linewidth=2)

# ========================================
# Defective superlattice (blue)
# ========================================
periodic_structure_defect = np.array(
    [20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
     9.999, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
) * 1e-9

wells_defect = np.array(
    [0.7, 0, 0.7, 0, 0.7, 0, 0.7, 0, 0.7, 0,
     1.0, 0, 0.7, 0, 0.7, 0, 0.7, 0, 0.7, 0, 0.7]
) * barrier

z_d, V0_d = np.array([]), np.array([])
for i, width in enumerate(periodic_structure_defect):
    zv = np.arange(0 if i == 0 else z_d[-1] + dz, (z_d[-1] + width) + dz if i > 0 else width + dz, dz)
    z_d = np.concatenate([z_d, zv]) if i > 0 else zv
    V0_d = np.concatenate([V0_d, np.full_like(zv, wells_defect[i])]) if i > 0 else np.full_like(zv, wells_defect[i])

Nz_d = len(z_d)
period_d = Nz_d * dz * (super_lattice_N + 1)

for ii in range(-Nkv + 1, Nkv + 1):
    kvec = ((ii - 1) / Nkv) * (np.pi / period_d)
    E1, _ = Sch1D(z_d, V0_d, mass, n, kvec, super_lattice_N)
    E1 = E1[E1 < 0.1]

    # Left panel
    for E_val in E1:
        ax1.hlines(E_val, 1, 2, color='b', linewidth=1, linestyle='-')
    # Right panel
    for E_val in E1:
        ax2.hlines(E_val, ii - 1, ii, color='b', linewidth=2, linestyle=':')

# ========================================
# Finalise left panel
# ========================================
ax1.set_ylim(0, 0.1)
ax1.set_xlim(0, 2)
ax1.set_xticks([])
ax1.tick_params(axis='y', labelsize=fnt_size)
ax1.set_ylabel('Energy (eV)', fontsize=fnt_size)
ax1.hlines(ax1.get_ylim()[0], *ax1.get_xlim(), color='w', linewidth=2)

# ========================================
# Finalise right panel
# ========================================
ax2.set_ylim(-0.0001, 0.1)
ax2.set_xlim(-21, 21)
ax2.set_xlabel(r'$kaN_{cell}/\pi$', fontsize=fnt_size)
ax2.set_xticks(np.linspace(-20, 20, 11))
ax2.set_xticklabels(['-1.0','-0.8','-0.6','-0.4','-0.2','0.0','0.2','0.4','0.6','0.8','1.0'])
ax2.tick_params(axis='both', labelsize=fnt_size)
ax2.spines[['left','bottom']].set_linewidth(2)
ax2.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig("Figure_test.pdf", format='pdf', dpi=300, bbox_inches='tight')
plt.show()
