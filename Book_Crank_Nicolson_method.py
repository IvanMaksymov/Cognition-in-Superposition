import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import os, cv2

# Parameters
L, dy, Nt = 8, 0.05, 300
dt = dy**2/4; N = round(L/dy)+1; rx = ry = -dt/(2j*dy**2)
x0, y0, sigma, k = L/5, L/2, 0.5, 15*np.pi
x = np.linspace(0,L,N); y = np.linspace(0,L,N)
X,Y = np.meshgrid(x,y)

# Initial wave function
psi0 = np.exp(-0.5*((X-x0)**2+(Y-y0)**2)/sigma**2)*np.exp(1j*k*(X-x0))

# Double slit
w,s,a,v0 = 0.6,0.8,0.0,500
j0,j1 = round((L-w)/(2*dy)), round((L+w)/(2*dy))
i0,i1 = round((L+s)/(2*dy)+a/dy), round((L+s)/(2*dy))
i2,i3 = round((L-s)/(2*dy)), round((L-s)/(2*dy)-a/dy)
v = np.zeros((N,N)); v[0:i3,j0:j1]=v0
v[i2:i1,j0:j1]=v0; v[i0:,j0:j1]=v0

# Sparse matrices
Ni = N*N; A = sp.lil_matrix((Ni,Ni),dtype=complex)
M = sp.lil_matrix((Ni,Ni),dtype=complex)
index = lambda i,j: i*N+j
for j in range(N):
    for i in range(N):
        k = index(i,j)
        A[k,k] = 1+2*rx+2*ry+1j*dt/2*v[i,j]
        M[k,k]=1-2*rx-2*ry-1j*dt/2*v[i,j]
        if i>0: A[k,index(i-1,j)]=-rx; M[k,index(i-1,j)]=rx
        if i<N-1: A[k,index(i+1,j)]=-rx; M[k,index(i+1,j)]=rx
        if j>0: A[k,index(i,j-1)]=-ry; M[k,index(i,j-1)]=ry
        if j<N-1: A[k,index(i,j+1)]=-ry; M[k,index(i,j+1)]=ry
A, M = sp.csc_matrix(A), sp.csc_matrix(M)

# Time evolution
psi = np.zeros((N,N,Nt),dtype=complex); psi[:,:,0]=psi0
psi_vect=psi0.flatten()
for t in range(1,Nt):
    psi_vect = spla.spsolve(A,M@psi_vect)
    psi[:,:,t]=psi_vect.reshape(N,N); psi[:,[0,-1],t]=0
    psi[[0,-1],:,t]=0

# Visualisation
time_steps=[5,115,165,295]
fig,axes=plt.subplots(1,4,figsize=(16,4))
for ax,t in zip(axes,time_steps):
    im=ax.imshow(np.abs(psi[:,:,t])**2,extent=[0,L,0,L],origin="lower",cmap="hot")
    ax.contour(X,Y,v,levels=[v0/2],colors='white')
    ax.set_title(f"Time step {t}",fontsize=10,pad=8)
    ax.set_xlabel("x-coordinate",fontsize=10,labelpad=5)
    cbar=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    cbar.ax.tick_params(labelsize=8)
plt.tight_layout(rect=[0,0,1,0.95]); plt.show()