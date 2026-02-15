import numpy as np
from numpy.random import randn

class RNN_QT:
  # A many-to-one Vanilla Quantum Tunnelling Recurrent Neural Network.

  # Physical constants and parameters of the potential barrier
  m = 9.1093837e-31
  hbar = 1.054571817e-34 
  qe = 1.60217663e-19

  m1 = 0.5
  Vpot = 30.0 # eV
  ampl = 10.0
  L = m1*hbar/np.sqrt(2*m*qe*Vpot)

  def __init__(self, input_size, output_size, hidden_size=64):
    # Weights
    self.Whh = randn(hidden_size, hidden_size) / 1000
    self.Wxh = randn(hidden_size, input_size) / 1000
    self.Why = randn(output_size, hidden_size) / 1000

    # Biases
    self.bh = np.zeros((hidden_size, 1))
    self.by = np.zeros((output_size, 1))

  def diffQT(self, L, E, U0):
    # Precompute constants
    m_qe = self.m * self.qe
    hbar_squared = self.hbar ** 2

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
        kappa1 = np.sqrt(-2.0 * m_qe * alpha) / self.hbar
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
        kappa = np.sqrt(2.0 * m_qe * alpha) / self.hbar
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
            4 * L**4 * U0 * self.m**2 * self.qe**2
            + 6 * L**2 * hbar_squared * self.m * self.qe
        ) / (
            3 * L**4 * U0**2 * self.m**2 * self.qe**2
            + 12 * L**2 * U0 * hbar_squared * self.m * self.qe
            + 12 * hbar_squared**2
        )

    return diffT.reshape((E.size, 1))

  def QT(self, L, E, U0):
    # Precompute constants
    m_qe = self.m * self.qe
    hbar_squared = self.hbar ** 2

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
        kappa1 = np.sqrt(-2.0 * m_qe * alpha) / self.hbar
        beta = U0**2 / (4.0 * E_between * alpha)
        T[mask_E_between] = 1.0 / (1.0 - beta * np.sinh(kappa1 * L) ** 2)

    # For E > U0
    if np.any(mask_E_greater):
        E_greater = E[mask_E_greater]
        alpha = E_greater - U0
        kappa = np.sqrt(2.0 * m_qe * alpha) / self.hbar
        beta = U0**2 / (4.0 * E_greater * alpha)
        T[mask_E_greater] = 1.0 / (1.0 + beta * np.sin(kappa * L) ** 2)

    # For E == U0
    if np.any(mask_E_equal):
        T[mask_E_equal] = 1.0 / (
            1.0 + self.m * L**2 * U0 * self.qe / (2.0 * hbar_squared)
        )

    return T.reshape((E.size, 1))

  def forward(self, inputs):
    '''
    Perform a forward pass of the RNN using the given inputs.
    Returns the final output and hidden state.
    - inputs is an array of one hot vectors with shape (input_size, 1).
    '''
    h = np.zeros((self.Whh.shape[0], 1))
    dh = np.zeros((self.Whh.shape[0], 1))

    self.last_inputs = inputs
    self.last_hs = { 0: h }
    self.last_dhs = { 0: h }

    # Perform each step of the RNN
    for i, x in enumerate(inputs):
      #h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
      #self.last_hs[i + 1] = h
      
      self.last_arg = self.Wxh @ x + self.Whh @ h + self.bh
      h = self.QT(self.L, self.last_arg*self.ampl, self.Vpot)
      self.last_hs[i + 1] = h
      
      dh = self.diffQT(self.L, self.last_arg*self.ampl, self.Vpot)
      self.last_dhs[i + 1] = dh

    # Compute the output
    y = self.Why @ h + self.by

    return y, h

  def backprop(self, d_y, learn_rate=2e-2*9.0):#*9.0
    '''
    Perform a backward pass of the RNN.
    - d_y (dL/dy) has shape (output_size, 1).
    - learn_rate is a float.
    '''
    n = len(self.last_inputs)

    # Calculate dL/dWhy and dL/dby.
    d_Why = d_y @ self.last_hs[n].T
    d_by = d_y

    # Initialize dL/dWhh, dL/dWxh, and dL/dbh to zero.
    d_Whh = np.zeros(self.Whh.shape)
    d_Wxh = np.zeros(self.Wxh.shape)
    d_bh = np.zeros(self.bh.shape)

    # Calculate dL/dh for the last h.
    # dL/dh = dL/dy * dy/dh
    d_h = self.Why.T @ d_y

    # Backpropagate through time.
    for t in reversed(range(n)):
      # An intermediate value: dL/dh * (1 - h^2)
      #temp = ((1 - self.last_hs[t + 1] ** 2) * d_h)
      temp = (self.last_dhs[t+1] * d_h)

      # dL/db = dL/dh * (1 - h^2)
      d_bh += temp

      # dL/dWhh = dL/dh * (1 - h^2) * h_{t-1}
      d_Whh += temp @ self.last_hs[t].T

      # dL/dWxh = dL/dh * (1 - h^2) * x
      d_Wxh += temp @ self.last_inputs[t].T

      # Next dL/dh = dL/dh * (1 - h^2) * Whh
      d_h = self.Whh @ temp

    # Clip to prevent exploding gradients.
    for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
      np.clip(d, -1, 1, out=d)

    # Update weights and biases using gradient descent.
    self.Whh -= learn_rate * d_Whh
    self.Wxh -= learn_rate * d_Wxh
    self.Why -= learn_rate * d_Why
    self.bh -= learn_rate * d_bh
    self.by -= learn_rate * d_by
