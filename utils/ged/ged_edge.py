# -*- coding: utf-8 -*-
"""
@author: sk1712
"""
from numpy import linalg
from scipy.spatial import distance

from utils.ged.ged_base import GedBase


class GedEdge(GedBase):
    """
    Calculates the edit distance between the edges of two nodes
    A node is regarded as a graph and edges are regarded as nodes
    """
    
    def __init__(self, g1, g2, greedy, v1, v2, verbose):
        GedBase.__init__(self, g1, g1, greedy, verbose)
        # List of edge weights for v1
        self.e1 = [g1.edges[v1, e]['weight'] for e in g1.neighbors(v1)]
        self.N = len(self.e1)
        
        # List of edge weights for v2
        self.e2 = [g2.edges[v2, e]['weight'] for e in g2.neighbors(v2)]
        self.M = len(self.e2)
        
    def insert_cost(self, i, j):
        cost = 0
        if i == j:
            cost = linalg.norm(self.e2[j])
        else:
            cost = float('inf')
            
        return cost
        
    def delete_cost(self, i, j):
        cost = 0
        
        if i == j:
            cost = linalg.norm(self.e1[i])
        else:
            cost = float('inf')
            
        return cost
        
    def substitute_cost(self, i, j):
        cost = distance.euclidean(self.e1[i], self.e2[j])
        return cost