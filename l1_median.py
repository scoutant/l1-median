import numpy as np
from scipy.spatial.distance import cdist, euclidean

def l1_median(X, eps=1e-6):
    """
    Computes weighted geometric median
    :param X: the list of sample points, a 2D ndarray
    :param eps: acceptable error margin
    :return: first estimate meeting eps
    """
    y = np.mean(X,0) # the geometric mean is a fare start
    while True:
        while np.any(cdist(X,[y])==0): # Euclidean distances, let's move away to avoid any null
            y +=0.1*np.ones(len(y))
        # set of weights that are the inverse of the distances from current estimate to the observations
        W = 1/cdist(X,[y]) # element-wise
        # new estimate is the weighted average of the observations
        y1 = np.sum(W*X,0)/np.sum(W) # sum along axis 0
        if euclidean(y,y1) < eps:
            return y1
        y = y1

def weighted_l1_median(X, WX, eps=1e-6):
    """
    Computes weighted geometric median
    :param X: the list of sample points, a 2D ndarray
    :param WX: the list of weights
    :param eps: acceptable error margin
    :return: first estimate meeting eps
    """
    y = np.average(X,axis=0,weights=WX)
    while True:
        while np.any(cdist(X,[y])==0):
            y +=0.1*np.ones(len(y))
        W = np.expand_dims(WX,axis=1)/cdist(X,[y]) # element-wise operation
        y1 = np.sum(W*X,0)/np.sum(W)
        if euclidean(y,y1) < eps:
            return y1
        y = y1