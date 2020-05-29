from scipy.spatial import distance
import numpy as np

from utils.ged.ged_base import GedBase

class GedTune(GedBase):
    """ 
    Base class from Graph Edit Distance
    """
    
    def __init__(self, g1, g2, greedy=False, verbose=False):
        GedBase.__init__(self, g1, g2, greedy, verbose)
                  
        self.N = g1.number_of_nodes()
        self.M = g2.number_of_nodes()

    def insert_cost(self, i, j):       
        if i == j:
            # Need to add the edge cost
            edge_list = [np.abs(self.g2.edges[j, e]['weight'])
                        for e in self.g2.neighbors(j)]
            cost = np.sum(edge_list)
        else:
            cost = float('inf')
            
        return cost

    def delete_cost(self, i, j):
        if i == j:                     
            # Need to add the edge cost                                            
            edge_list = [np.abs(self.g1.edges[i, e]['weight'])
                        for e in self.g1.neighbors(i)]
            cost = np.sum(edge_list)
        else:
            cost = float('inf')
            
        return cost
        
    def substitute_cost(self, i, j):
        cost = self.edge_diff(i, j)
        return cost
        
    def edge_diff(self, i, j):      
        edges_i = np.array(list(self.g1.neighbors(i)))
        edges_j = np.array(list(self.g2.neighbors(j)))
        
        if len(edges_i) == 0:
            edge_list = [np.abs(self.g2.edges[j, e]['weight']) for e in edges_j]
            return np.sum(edge_list)
            
        elif len(edges_j) == 0:
            edge_list = [np.abs(self.g1.edges[i, e]['weight']) for e in edges_i]
            return np.sum(edge_list)
        
        intersect, comm_i, comm_j = np.intersect1d(edges_i, edges_j, return_indices=True)
        mask_i = np.ones(edges_i.size, dtype=bool)
        mask_i[comm_i] = False
        mask_j = np.ones(edges_j.size, dtype=bool)
        mask_j[comm_j] = False
        edge_list_i = [np.abs(self.g1.edges[i, e]['weight']) for e in edges_i[mask_i]]
        edge_list_j = [np.abs(self.g2.edges[j, e]['weight']) for e in edges_j[mask_j]]
        edge_list_comm = [np.abs(self.g1.edges[i, e]['weight'] - self.g2.edges[j, e]['weight']) for e in intersect]
        return np.sum(edge_list_i + edge_list_j + edge_list_comm)
          
    def distance(self):
        """
        Total distance between the two graphs
        """
        _, cols, costs = self.calculate_costs()
        self.Mindices = cols[:self.N]
        return np.sum(costs)
