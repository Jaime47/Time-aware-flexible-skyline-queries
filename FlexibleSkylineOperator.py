
import numpy as np
import operator
import pandas as pd
from datetime import datetime
import itertools

# Over L_1 family of function, minimum oriented

# Time series support

def generate_distribution(normalized, distribution):
    dist = []
    for value in normalized:
        dist.append((value,distribution(value)))
    return dist

def score_tuple_distribution(normalized, distribution):
    dist = []
    for value in normalized:
        dist.append((value,distribution(value)))
    return dist


### Time distribution test functions
def normal_standard_dist(x):
    prob_density = (np.pi) * np.exp(-0.5*((x))**2)
    return 1 + prob_density
def x_med(x):
    return 1 + x/2
def id_dist(x):
    return 1
###

class FlexibleSkylineOperator:
    dim = 0
    time_factors = []
    tuples = [] #List of tuples (tuple, date)

    def __init__(self, dim, time_factors, tuples) -> None:
        self.dim = dim
        self.time_factors = time_factors
        self.tuples = tuples

    def score_tuple(self, tuple, weights):
        new_WC = FlexibleSkylineOperator.re_compute_weights(self, tuple[1], [weights])
        vector = np.multiply(tuple[0], new_WC[0])
        return np.sum(vector)
    
    # Obtain WC using simplex algorithm
    # A: Equation system matrix
    # b: Equation system vector
    def compute_WC(self, A, ops):

        WC_set = set()
        condition_failed_flag = False
        # Generated n dimensional matrix
        default_matrix = np.diag(np.full(self.dim+1,1))
        default_matrix = default_matrix[:self.dim]
        A_mod = np.concatenate((A, default_matrix))
        combinations = list(itertools.combinations(A_mod, self.dim - 1))
    
        for subset in combinations:
            subset = np.array(subset[0:self.dim-1])           
            default = [np.append(np.ones(self.dim), 1)]
            subset = np.concatenate((subset, default))
            b_sub = subset[:,-1]
            A_sub = subset[:,:-1]
            if np.linalg.det(A_sub) == 0:
                continue
            vect = np.linalg.solve(A_sub, b_sub)
            condition_failed_flag = False
            for i in range(len(ops)):
                if not ops[i](np.sum(np.multiply(A[i][:-1],vect)) , A[i][-1]):
                    condition_failed_flag = True
                    break
            if not condition_failed_flag:
                WC_set.add(tuple(vect))
        return WC_set

    def re_compute_weights(self, time, WC):
        new_WC = []
        for point in WC:
            p = []
            for i in range(self.dim):
                p.append(point[i]*self.time_factors[i](time))
            norm = np.linalg.norm(p, ord=1)
            new_WC.append(tuple([element/norm for element in p]))
        return new_WC
    
    # Computes the centroid of WC for SVE1F
    def compute_centroid_WC(self, vectors):
        centroid = []
        vectors = list(vectors)
        for i in range(self.dim):
            sum = 0
            for j in range(len(vectors)):
                sum = sum + vectors[j][i]
            centroid.append(sum/self.dim)
        return centroid

    # Sorts r following some weight criteria    
    def sort_r(self, r, weights):
        sorted_r = []
        for tuple in r:
            score = FlexibleSkylineOperator.score_tuple(self, tuple, weights)
            sorted_r.append((score, tuple))
        ordered_r = sorted(sorted_r)
        return ordered_r #[element[1] for element in ordered_r]
        
    def calculate_dominance_region(self, tu, WC):
        dominance_region_vector = np.zeros(len(WC))
        new_WC = FlexibleSkylineOperator.re_compute_weights(self, tu[1], WC)
        for i in range(len(WC)):
            for j in range(self.dim):
                dominance_region_vector[i] = dominance_region_vector[i] + new_WC[i][j] *tu[0][j]
        dominance_region_vector = tuple(dominance_region_vector)
        return dominance_region_vector
    
    # Return true if a is in b dominance region
    def in_dominance_region_alt(self, a, b, WC):
        dra = FlexibleSkylineOperator.calculate_dominance_region(self, a, WC)
        drb = FlexibleSkylineOperator.calculate_dominance_region(self, b, WC)
        for i in range(len(dra)):
            if dra[i] < drb[i]:
                return False
        return True
    def in_dominance_region(self, a, b, WC):
        for i in range(len(a)):
            if a[i] < b[i]:
                return False
        return True
    
    def dominates(self, a, b):
        better = False
        for i in range(self.dim):
            a_i, b_i = a[i], b[i]
            # Worse in one dimension -> does not domiate
            # This is faster than computing `at least as good` in every dimension
            if a_i > b_i:
                return False
            # Better in at least one dimension
            if a_i < b_i:
                better = True
        return better

    def alg_SVE1F_ND(self, A, ops):
        WC = self.compute_WC(A , ops)
        sorted_r = self.sort_r(self.tuples,self.compute_centroid_WC(WC))
        nd = set()
        for s in sorted_r:
            s = s[1]
            flag_dominated_tuple = False
            s = (s, FlexibleSkylineOperator.calculate_dominance_region(self, s, list(WC)))
            for t in nd:
                if self.dominates(t[0][0], s[0][0]) or self.in_dominance_region(s[1], t[1], list(WC)):
                    flag_dominated_tuple = True
                    break
            if not flag_dominated_tuple:
                nd.add(s)
        return nd