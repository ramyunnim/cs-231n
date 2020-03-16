import numpy as np

p = 0.5  # probablility of keeping a unit active. higher = less dropout

def train_step:

    # forward pass for example 3-layer neural network
    H1 = np.maximum(0, np.dot(W1, X) + b1)
    U1 = np.random.rand(*H1.shape) < p # first dropout mask
    H1 *= U1 # drop!

    H2 = np.maximum(0, np.dot(W2, H1) + b2)
    U2 = np.random.rand(*H2.shape) < p  # first dropout mask
    H2 *= U2  # drop!
    out = np.dot(W3, H2) + b3

