import numpy as np
from active_learning.data_recommended.hitandrun import billiardwalk

def convex(M):

    # Check if positive definite
    try:
        np.linalg.cholesky(0.5*(M+M.T))
        return 1
    except np.linalg.LinAlgError:
        return 0

def convexMult(Ms):

    ind = np.zeros(Ms.shape[0],dtype=bool)
    for i in range(Ms.shape[0]):
        ind[i] = convex(Ms[i])

    return ind

