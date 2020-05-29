import importlib

import numpy as np
import networkx as nx
#import gmatch4py as gm
from scipy import stats
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from utils.ged.ged_tune import GedTune
class distance_requester(object):
    def __init__(self, conn1, conn2, eig_thresh=10**(-3)):
        self.conn1 = conn1
        self.conn2 = conn2
        self.eig_thresh = eig_thresh

        # ensure symmetric
        self.conn1 = self._ensure_symmetric(self.conn1)
        self.conn2 = self._ensure_symmetric(self.conn2)

    def _ensure_symmetric(self, Q):
        '''
        computation is sometimes not precise (round errors),
        so ensure matrices that are supposed to be
        symmetric are symmetric
        '''
        return (Q + np.transpose(Q))/2

    def _vectorize(self, Q):
        '''
        given a symmetric matrix (conn), return unique
        elements as an array. Ignore diagonal elements
        '''
        # extract lower triangular matrix
        vec = Q[np.tril_indices(len(Q), -1)]
        return vec

    def geodesic(self):
        '''
        dist = sqrt(trace(log^2(M)))
        M = Q_1^{-1/2}*Q_2*Q_1^{-1/2}
        '''
        # compute Q_1^{-1/2} via eigen value decmposition
        u, s, _ = np.linalg.svd(self.conn1, full_matrices=True)

        ## lift very small eigen values
        for ii, s_ii in enumerate(s):
            if s_ii < self.eig_thresh:
                s[ii] = self.eig_thresh

        '''
        since conn1 is in S+, u = v, u^{-1} = u'
        conn1 = usu^(-1)
        conn1^{1/2} = u[s^{1/2}]u'
        conn1^{-1/2} = u[s^{-1/2}]u'
        '''
        conn1_mod = u @ np.diag(s**(-1/2)) @ np.transpose(u)
        M = conn1_mod @ self.conn2 @ conn1_mod

        '''
        trace = sum of eigenvalues;
        np.logm might have round errors,
        implement using svd instead
        '''
        _, s, _ = np.linalg.svd(M, full_matrices=True)

        return np.sqrt(np.sum(np.log(s)**2))

    def pearson(self):
        '''
        conventional Pearson distance between
        two conn matrices. The matrices are vectorized
        '''
        vec1 = self._vectorize(self.conn1)
        vec2 = self._vectorize(self.conn2)

        corr, p = stats.pearsonr(vec1, vec2)
        return (1 - corr)/2, p

    def spearman(self):
        '''
        Spearman distance between
        two conn matrices. The matrices are vectorized
        '''
        vec1 = self._vectorize(self.conn1)
        vec2 = self._vectorize(self.conn2)

        corr, p = stats.spearmanr(vec1, vec2)
        return (1 - corr)/2, p

    def euclidean(self):
        '''
        Euclidean distance between
        two conn matrices. The matrices are vectorized
        '''
        vec1 = self._vectorize(self.conn1)
        vec2 = self._vectorize(self.conn2)
        return distance.euclidean(vec1, vec2)

    def canberra(self):
        '''
        Euclidean distance between
        two conn matrices. The matrices are vectorized
        '''
        vec1 = self._vectorize(self.conn1)
        vec2 = self._vectorize(self.conn2)
        return distance.canberra(vec1, vec2)
    
    def graphedit(self):
        G1 = nx.from_numpy_array(self.conn1)
        G2 = nx.from_numpy_array(self.conn2)

        #dist_fun = gm.GraphEditDistance(1,1,1,1)
        #dist = dist_fun.compare([G1, G2], None)[0, 1]
        GED = GedTune(G1, G2)
        dist = GED.distance()
        return dist