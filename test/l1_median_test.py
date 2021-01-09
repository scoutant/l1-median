import unittest
import numpy as np
from l1_median import l1_median
from l1_median import weighted_l1_median
from scipy.spatial.distance import euclidean

eps = 0.001
class L1MedianTest(unittest.TestCase):
    def test_4_points(self):
        X = np.array( [
            [100, 0],
            [-1, 0],
            [0, +1],
            [0, -1]
        ])
        m = l1_median(X,eps)
        self.assertTrue(euclidean(m, np.array([0,0]))<eps )

    def test_with_a_point_on_the_way(self):
        """ first estimate (the mean) happens to fall on one of the sample points, we need to handle null distances too """
        X = np.array( [
            [1, 0],
            [-1, 0],
            [0, +1],
            [0, -1],
            [0,0] # inferring a null distance for the first estimate
        ])
        m = l1_median(X,eps)
        self.assertTrue(euclidean(m, np.array([0,0]))<eps ) # good news, could handle the null distance

    def test_4_weighted_points(self):
        X = np.array( [
            [100, 0],
            [-1, 0],
            [0, +1],
            [0, -1]
        ])
        m = weighted_l1_median(X, [0.5,1,1,1], eps) # smaller weight for the outlier
        self.assertFalse(euclidean(m, np.array([0,0]))<eps ) # the weighted median is distinct to the unweighted result
        self.assertTrue(euclidean(m, np.array([-0.2576,0]))<eps ) # even more closer to the 3 main sample points