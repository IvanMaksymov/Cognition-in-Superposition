# -*- coding: utf-8 -*-
"""
This QT-ESN code is based on the minimalistic 
Echo State Networks demo with Mackey-Glass (delay 17) data 
from https://mantas.info/code/simple_esn/
(c) 2012-2020 Mantas Lukoševičius
Distributed under MIT license https://opensource.org/licenses/MIT
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg 
# numpy.linalg is also an option for even fewer dependencies


# Physical constants and parameters of the potential barrier
m = 9.1093837e-31
hbar = 1.054571817e-34 
qe = 1.60217663e-19

m1 = 0.5
Vpot = 30.0 # eV
ampl = 3.0
L = m1*hbar/np.sqrt(2*m*qe*Vpot)

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

    return T#.reshape((E.size, 1))

# load the data
trainLen = 2000
testLen = 2000
initLen = 100
data = np.loadtxt('MackeyGlass_t17.txt')

# plot some of it
plt.figure(10).clear()
plt.plot(data[:1000])
plt.title('A sample of data')

# generate the ESN reservoir
inSize = outSize = 1
resSize = 1000
a = 0.3 # leaking rate
np.random.seed(42)
Win = (np.random.rand(resSize,1+inSize) - 0.5) * 1
W = np.random.rand(resSize,resSize) - 0.5 
# normalizing and setting spectral radius (correct, slow):
print('Computing spectral radius...')
rhoW = max(abs(linalg.eig(W)[0]))
print('done.')
W *= 1.25 / rhoW

# allocated memory for the design (collected states) matrix
X = np.zeros((1+inSize+resSize,trainLen-initLen))
# set the corresponding target matrix directly
Yt = data[None,initLen+1:trainLen+1] 

# run the reservoir with the data and collect X
x = np.zeros((resSize,1))
for t in range(trainLen):
    u = data[t]
    tmp = np.dot( Win, np.vstack((1,u)) ) + np.dot( W, x )
    x = (1-a)*x + a*QT(L, ampl*tmp, Vpot)
    
    if t >= initLen:
        X[:,t-initLen] = np.vstack((1,u,x))[:,0]
    
# train the output by ridge regression
reg = 1e-8  # regularization coefficient
# direct equations from texts:
#X_T = X.T
#Wout = np.dot( np.dot(Yt,X_T), linalg.inv( np.dot(X,X_T) + \
#    reg*np.eye(1+inSize+resSize) ) )
# using scipy.linalg.solve:
Wout = linalg.solve( np.dot(X,X.T) + reg*np.eye(1+inSize+resSize), 
    np.dot(X,Yt.T) ).T

# run the trained ESN in a generative mode. no need to initialize here, 
# because x is initialized with training data and we continue from there.
Y = np.zeros((outSize,testLen))
u = data[trainLen]
for t in range(testLen):
    tmp = np.dot( Win, np.vstack((1,u)) ) + np.dot( W, x )
    x = (1-a)*x + a*QT(L, ampl*tmp, Vpot)
    y = np.dot( Wout, np.vstack((1,u,x)) )
    Y[:,t] = y
    # generative mode:
    u = y
    ## this would be a predictive mode:
    #u = data[trainLen+t+1] 

# compute MSE for the first errorLen time steps
errorLen = 500
mse = sum( np.square( data[trainLen+1:trainLen+errorLen+1] - 
    Y[0,0:errorLen] ) ) / errorLen
print('MSE = ' + str( mse ))
    
# plot some signals
plt.figure(1).clear()
plt.plot( data[trainLen+1:trainLen+testLen+1], 'g--' )
plt.plot( Y.T, 'b' )
#plt.title('Target and generated signals $y(n)$ starting at $n=0$')
plt.legend(['Target signal', 'Free-running predicted signal'], loc='upper right')
plt.xlabel('Time (arb. units)', fontsize=16)
plt.ylabel('Amplitude (arb. units)', fontsize=16)
plt.tick_params(axis='both', labelsize=16)

plt.figure(2).clear()
plt.plot( X[0:20,0:200].T )
plt.title(r'Some reservoir activations $\mathbf{x}(n)$')

plt.figure(3).clear()
plt.bar( np.arange(1+inSize+resSize), Wout[0].T )
plt.title(r'Output weights $\mathbf{W}^{out}$')

plt.show()

