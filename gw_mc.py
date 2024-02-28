import numpy as np
import cvxpy as cp
from scipy.linalg import sqrtm

def gw(n, edges):
    '''Goemans-Williamson algorithm for Max-Cut:
    Given a graph G(V=[n], E=edges), returns a vector x \in {-1, 1}^n
    that corresponds to the chosen subset of vertices S of V. '''
    ## SDP Relaxation
    X = cp.Variable((n, n), symmetric=True)
    constraints = [X >> 0]
    constraints += [
        X[i, i] == 1 for i in range(n)
    ]
    objective = sum( 0.5*(1 - X[i, j]) for (i, j) in edges )
    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve()

    ## Hyperplane Rounding
    Q = sqrtm(X.value).real
    r = np.random.randn(n)
    x = np.sign(Q @ r)

    return x

def cut(x, edges):
    '''Given a vector x \in {-1, 1}^n and edges of a graph G(V=[n], E=edges),
    returns the edges in cut(S) for the subset of vertices S of V represented by x.'''
    xcut = []
    for i, j in edges:
        if np.sign(x[i]*x[j]) < 0:
            xcut.append((i, j))
    return xcut


# def example():
#     ''' Cycle on n=5 vertices. '''
#     ## Define the graph
#     n = 4
#     edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
#     edges_02 = [(0,1), (0,2), (0,3), (1,3), (2,3)]
#     edges_01 = [(0,3), (0,5), (1,2), (1,3), (1,4), (1,5), (1,7), (2,3), (2,6), (3,5), (3,6), (4,7), (5,6), (5,7), (5,9), (6,9), (8,9)]


#     ## Define SDP Relaxation and solve it
#     x = gw(n, edges_02)

#     ## Find the edges of a cut
#     xcut = cut(x, edges_02)

#     ## Result
#     print('Chosen subset: %s' % np.where(x == 1))
#     print('Cut size: %i' % len(xcut) )
#     print('Edges of the cut: %s' % xcut )
# example()

# num_nodes = 22
# edges = [(6, 12), (6, 1), (6, 0), (12, 10), (12, 5), (5, 19), (5, 21), (19, 9), (19, 1), (8, 18), (8, 17), (8, 16), (18, 17), (18, 3), (1, 11), (17, 10), (10, 2), (11, 20), (11, 3), (20, 0), (20, 2), (0, 21), (2, 16), (3, 9), (9, 15), (4, 14), (4, 7), (4, 13), (14, 15), (14, 7), (15, 13), (21, 7), (16, 13)]
# num_edges = len(edges)

# sdp_relaxation_sols= gw(num_nodes, edges)
# # edges of the cut
# cut = cut(sdp_relaxation_sols, edges)
# # cut size
# cut_size = len(cut)
# # print(cut_size, num_edges)
# sdp_energy = (num_edges - 2 * cut_size)/num_nodes
# print(sdp_energy)