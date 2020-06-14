import importlib

from scipy import optimize
import numpy as np
import networkx as nx

def linear_sum_assignment_with_inf(cost_matrix):
    cost_matrix = np.asarray(cost_matrix)
    min_inf = np.isneginf(cost_matrix).any()
    max_inf = np.isposinf(cost_matrix).any()
    if min_inf and max_inf:
        raise ValueError("matrix contains both inf and -inf")

    if min_inf or max_inf:
        values = cost_matrix[~np.isinf(cost_matrix)]
        m = values.min()
        M = values.max()
        n = min(cost_matrix.shape)
        # strictly positive constant even when added
        # to elements of the cost matrix
        positive = n * (M - m + np.abs(M) + np.abs(m) + 1)
        if max_inf:
            place_holder = (M + (n - 1) * (M - m)) + positive
        if min_inf:
            place_holder = (m + (n - 1) * (m - M)) - positive

    cost_matrix[np.isinf(cost_matrix)] = place_holder
    return optimize.linear_sum_assignment(cost_matrix)

class GedBase(object):
    """ 
    Base class from Graph Edit Distance
    """
    
    def __init__(self, g1, g2, node_factor=None, greedy=False, verbose=False):
        """
        Class constructor
        """
        self.g1 = g1
        self.g2 = g2
        
        self.N = g1.number_of_nodes()
        self.M = g2.number_of_nodes()
        
        self.greedy = greedy
        self.verbose = verbose

        if node_factor is None:
            self.node_factor1 = np.ones(self.N)
            self.node_factor2 = np.ones(self.M)
        else:
            self.node_factor1 = node_factor[0]
            self.node_factor2 = node_factor[1]
        
    def make_cost_matrix(self):
        """
        Create the cost matrix to be optimised.
        
        This is a square matrix of size |n+m| x |n+m| 
        where n, m the number of nodes for the two graphs, respectively.
        It encodes all possible edit operation costs, considering all vertices
        of the two graphs.
        
        The cost matrix consists of four regions
        
        substitutions | deletions
        -----------------------------
        insertions    | zeros
        """        
        cost_matrix = np.zeros((self.N+self.M, self.N+self.M))

        A1 = self.node_factor1[None, :] * self.node_factor1[:, None] * nx.to_numpy_array(self.g1)
        A2 = self.node_factor2[None, :] * self.node_factor2[:, None] * nx.to_numpy_array(self.g2)

        # substitute cost
        for i in range(self.N):
            for j in range(self.M):
                cost_matrix[i, j] = np.abs(A1[i] - A2[j]).sum()

        # insert cost
        insert_cost_matrix = np.full((self.M, self.M), float('inf'))
        insert_cost_matrix[np.diag_indices(self.M)] = np.abs(A2).sum(axis = 0)
        cost_matrix[self.N:, :self.M] = insert_cost_matrix

        # delete cost
        delete_cost_matrix = np.full((self.N, self.N), float('inf'))
        delete_cost_matrix[np.diag_indices(self.N)] = np.abs(A1).sum(axis = 0)
        cost_matrix[:self.N, self.M:] = delete_cost_matrix
        
        return cost_matrix

    def distance(self):
        """
        Total distance between the two graphs
        """
        _, cols, costs = self.calculate_costs()
        self.Mindices = cols[:self.N]
        return np.sum(costs)
        
    def calculate_costs(self):
        """
        Return list of costs for all edit operations
        """
        cost_matrix = self.make_cost_matrix()
        
        if self.greedy:
            # Riesen et al., "Greedy Graph Edit Distance"
            costs = []
            psi = []
            row_ind = []
            col_ind = []
            for row in range(self.N):
                phi = self.M
                row_min = float('inf')
                for column in range(self.N+self.M):
                    if column not in psi:
                        if cost_matrix[row, column] < row_min:
                            row_min = cost_matrix[row, column]
                            phi = column
                    
                costs.append(row_min)
                row_ind.append(row)
                col_ind.append(phi)

                if phi < self.M:
                    psi.append(phi)
                    
            for row in range(self.N, self.N+self.M):
                if (row - self.N) not in psi:
                    costs.append(cost_matrix[row, row - self.N])
                    row_ind.append(row)
                    col_ind.append(row - self.N)
            row_ind = np.array(row_ind)
            col_ind = np.array(col_ind)
            costs = np.array(costs)
        else:
            # Riesen & Bunke, "Approximate graph edit distance computation by means of bipartite graph matching"
            row_ind, col_ind = linear_sum_assignment_with_inf(cost_matrix)
            
            if self.verbose:
                for row, column in (row_ind, col_ind):
                    value = cost_matrix[row, column]
                    print('%d, %d, %.4f' % (row, column, value))
            costs = np.array(cost_matrix[row_ind, col_ind])
        return row_ind, col_ind, costs
        
    def norm_distance(self):
        """
        Distance normalized on the size of the graphs
        """
        graph_size = self.N + self.M
        return self.distance() / (1. * graph_size)
        
    def print_matrix(self):
        print("Cost matrix:")
        for row in self.make_cost_matrix():
            for col in row:
                if col == float('inf'):
                    print("Inf\t", end='')
                else:
                    print("%.2f\t" % float(col), end='')
            print("\n", end='')