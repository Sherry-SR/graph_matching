from scipy import optimize
import numpy as np

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
    Implementation inspired by @haakondr
    """
    
    def __init__(self, g1, g2, greedy, verbose):
        """
        Class constructor
        """
        self.g1 = g1
        self.g2 = g2
        
        self.N = 0
        self.M = 0
        
        self.greedy = greedy
        self.verbose = verbose
        
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

        for i in range(self.N):
            for j in range(self.M):
                cost_matrix[i, j] = self.substitute_cost(i, j)
        
        for i in range(self.M):
            for j in range(self.M):
                cost_matrix[i+self.N, j] = self.insert_cost(i, j)
                
        for i in range(self.N):
            for j in range(self.N):
                cost_matrix[i, j+self.M] = self.delete_cost(i, j)
        
        return cost_matrix
                
    def insert_cost(self, i, j):
        """
        Cost of node j insertion
        """
        raise NotImplementedError
    
    def delete_cost(self, i, j):
        """
        Cost of node i deletion
        """
        raise NotImplementedError
        
    def substitute_cost(self, i, j):
        """
        Cost of substitution of node i from g1 with node j of g2
        """
        raise NotImplementedError
        
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
        
    def distance(self):
        """
        Total distance between the two graphs
        """
        _, _, costs = self.calculate_costs()
        return np.sum(costs)
        
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
                    print("Inf\t")
                else:
                    print("%.2f\t" % float(col))
            print("\n")
