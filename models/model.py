import importlib

import numpy as np
import torch
import pdb

class distance_requestor(torch.nn.Module):
    def __init__(self, distance, tau = 0, eig_thresh=10**(-3), **kwargs):
        super(distance_requestor, self).__init__()
        self.tau = tau
        self.eig_thresh = eig_thresh
        self.distance = getattr(self, distance)

    def _ensure_symmetric(self, Q):
        '''
        computation is sometimes not precise (round errors),
        so ensure matrices that are supposed to be
        symmetric are symmetric
        '''
        return (Q + Q.transpose())/2

    def _vectorize(self, Q):
        tri_indices = Q.tril(-1).nonzero().transpose()

        vec = Q[tri_indices[0], tri_indices[1]]
        return vec

    def forward(self, Q1, Q2):
        return self.distance(Q1, Q2)

    def geodesic(self, Q1, Q2):
        # add regularization
        Q1 = Q1 + self.tau*torch.eye(Q1.size[0])
        Q2 = Q2 + self.tau*torch.eye(Q2.size[0])
        '''
        dist = sqrt(trace(log^2(M)))
        M = Q_1^{-1/2}*Q_2*Q_1^{-1/2}
        '''
        # compute Q_1^{-1/2} via eigen value decmposition
        u, s, _ = torch.svd(Q1, some=False)
        ## lift very small eigen values
        s[s<self.eig_thresh] = self.eig_thresh
        '''
        since Q1 is in S+, u = v, u^{-1} = u', Q1 = usu^(-1)
        Q1^{1/2} = u[s^{1/2}]u', Q1^{-1/2} = u[s^{-1/2}]u'
        '''
        Q1_mod = torch.matmul(torch.matmul(u, torch.diag(s**(-1/2))), u.transpose())
        M = torch.matmul(torch.matmul(Q1_mod, Q2), Q1_mod)
        # trace is sum of eigenvalues, np.logm might have round errors, implement using svd instead
        _, s, _ = torch.svd(M, some=False)
        return torch.sqrt((torch.log(s)**2).sum())

    def pearson(self, Q1, Q2):
        '''
        conventional Pearson distance between
        two FC matrices. The matrices are vectorized
        '''
        vec1 = self._vectorize(Q1)
        vec1 = vec1 - torch.mean(vec1)

        vec2 = self._vectorize(Q2)
        vec2 = vec2 - torch.mean(vec2)
        corrcoef = vec1.dot(vec2)/(torch.norm(vec1, 2)*torch.norm(vec2,2))
        return (1 - corrcoef)/2