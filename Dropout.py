import numpy as np

def predict(X):
    # ensembled forward pass
    H1 = np.maximum(0, np.dot(W1, X) + b1) * p
    H2 = np.maximum(0, np.dot(W2, X) + b2) * p
    out = np.dot(W3, H2) + b3 