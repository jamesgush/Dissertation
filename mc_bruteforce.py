import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import random
import itertools
# import quimb as qu
# from quimb.tensor import TensorNetwork as qtn, Tensor as qt
import numpy as np
# import cvxpy as cp
from scipy.linalg import sqrtm
from tqdm import tqdm
from gw_mc import gw, cut
from personalised_steps import p_steps, exp_steps
import pandas as pd

def initialize_graph(num_nodes):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    return G

def add_random_edges(G, num_edges):
    nodes = list(G.nodes())
    for _ in range(num_edges):
        u, v = random.sample(nodes, 2)
        G.add_edge(u, v)

def add_fixed_edges(G, edges):
    for edge in edges:
        u = edge[0]
        v = edge[1]
        G.add_edge(u, v)

def visualize_graph(G):
    nx.draw(G, with_labels=True)
    plt.show()

def max_cut_brute_force(G):
    nodes = list(G.nodes)
    max_cut_value = 0
    max_cut = None

    for size in range(1, len(nodes) // 2 + 1):
        for subset in itertools.combinations(nodes, size):
            cut_value = sum((G.has_edge(u, v) for u in subset for v in G.nodes if v not in subset))
            if cut_value > max_cut_value:
                max_cut_value = cut_value
                max_cut = subset
    return max_cut, max_cut_value

def visualize_max_cut(G, max_cut):
    color_map = ['red' if node in max_cut else 'blue' for node in G.nodes()]
    nx.draw(G, node_color=color_map, with_labels=True, font_weight='bold')
    plt.show()

def find_non_zero_difference(points):
# Start from the end of the list and look for a non-zero difference
    for i in range(len(points) - 1, 0, -1):
        difference = points[i] - points[i-1]
        if difference != 0:
            return difference
    # If all differences are zero or the list is too short, return None or raise an error
    return None

def check_intersection(x, y, horizontal_line):
    for i in range(len(x) - 1):
        x1, x2 = x[i], x[i + 1]
        y1, y2 = y[i], y[i + 1]

        # Check if the horizontal line intersects the line segment
        if (y1 <= horizontal_line <= y2) or (y2 <= horizontal_line <= y1):
            # Calculate the x-coordinate of the intersection point
            intersection_x = x1 + (x2 - x1) * (horizontal_line - y1) / (y2 - y1)
            
            return True, intersection_x

    return False, None
# def compute_intersection(l1, l2):
    

# ----------------------------------------------------------------------------------------
# initialising graph 
# num_nodes = 15
# num_edges = 60
# # edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
# # edges_01 = [(0,3), (0,5), (1,2), (1,3), (1,4), (1,5), (1,7), (2,3), (2,6), (3,5), (3,6), (4,7), (5,6), (5,7), (5,9), (6,9), (8,9)]
# # edges_02 = [(0,1), (0,2), (0,3), (1,3), (2,3)]
# G = initialize_graph(num_nodes)
# add_random_edges(G, num_edges)
# print(G.edges())
# print(G.edges())
# nx.draw(G)
# plt.show()
# ----------------------------------------------------------------------------------------
# determining max cut with brute force
# max_cut, max_cut_value = max_cut_brute_force(G)
# print("Max Cut Brute Force:", max_cut)
# print("Max Cut Brute ForceValue :", max_cut_value)
# print("Numpy Array of G:", nx.to_numpy_array(G))
# visualize_max_cut(G, max_cut)

# [[0. 0. 0. 1. 0. 1. 0. 0. 0. 0.]
#  [0. 0. 1. 1. 1. 1. 0. 1. 0. 0.]
#  [0. 1. 0. 1. 0. 0. 1. 0. 0. 0.]
#  [1. 1. 1. 0. 0. 1. 1. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 1. 0. 0.]
#  [1. 1. 0. 1. 0. 0. 1. 1. 0. 1.]
#  [0. 0. 1. 1. 0. 1. 0. 0. 0. 1.]
#  [0. 1. 0. 0. 1. 1. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
#  [0. 0. 0. 0. 0. 1. 1. 0. 1. 0.]]

# ----------------------------------------------------------------------------------------
# TEST GRAPHS
# 4 Nodes and 5 Edges
# num_nodes = 4
# edges = [(0,1), (0,2), (0,3), (1,3), (2,3)]

# 8 Nodes and 15 Edges
# num_nodes = 8
# edges = [(0, 1), (0, 4), (0, 6), (1, 6), (1, 4), (1, 7), (2, 6), (2, 4), (2, 5), (3, 5), (4, 6), (5,7), (6,7),(1,6), (4,7)]

# 15 Nodes and 43 edges
# edges = [(0, 2), (0, 14), (0, 13), (0, 10), (0, 12), (0, 3), (0, 4), (1, 10), (1, 4), (1, 11), (1, 14), (1, 9), (2, 14), (2, 12), (2, 6), (2, 9), (2, 8), (2, 3), (2, 5), (3, 8), (3, 12), (3, 10), (3, 11), (3, 9), (3, 13), (4, 14), (4, 6), (4, 8), (4, 10), (5, 11), (5, 12), (6, 10), (6, 11), (7, 12), (7, 11), (7, 14), (8, 12), (8, 9), (8, 11), (9, 12), (10, 11), (10, 13), (11, 13)]

# 20 Nodes and 77 Edges
# edges = [(0, 2), (0, 4), (0, 1), (0, 3), (0, 6), (0, 5), (0, 17), (0, 18), (0, 15), (0, 11), (1, 10), (1, 11), (1, 19), (1, 4), (1, 15), (1, 9), (1, 13), (1, 14), (1, 18), (1, 17), (2, 17), (2, 9), (2, 5), (2, 11), (2, 3), (2, 15), (2, 16), (3, 11), (3, 8), (3, 5), (3, 4), (3, 17), (4, 14), (4, 8), (4, 10), (4, 6), (5, 11), (5, 7), (5, 9), (5, 15), (5, 19), (5, 10), (5, 6), (5, 14), (6, 10), (6, 14), (6, 9), (6, 19), (7, 17), (7, 15), (7, 19), (8, 19), (8, 14), (8, 10), (8, 18), (8, 16), (9, 12), (9, 18), (9, 14), (9, 19), (10, 11), (10, 19), (10, 15), (10, 13), (11, 12), (11, 19), (11, 16), (12, 17), (12, 18), (12, 13), (13, 15), (13, 16), (13, 18), (15, 18), (15, 17), (16, 19), (18, 19)]



# ----------------------------------------------------------------------------------------
# # Nature Method

# # Graph initialization
# num_nodes = 20
# edges = [(0, 2), (0, 4), (0, 1), (0, 3), (0, 6), (0, 5), (0, 17), (0, 18), (0, 15), (0, 11), (1, 10), (1, 11), (1, 19), (1, 4), (1, 15), (1, 9), (1, 13), (1, 14), (1, 18), (1, 17), (2, 17), (2, 9), (2, 5), (2, 11), (2, 3), (2, 15), (2, 16), (3, 11), (3, 8), (3, 5), (3, 4), (3, 17), (4, 14), (4, 8), (4, 10), (4, 6), (5, 11), (5, 7), (5, 9), (5, 15), (5, 19), (5, 10), (5, 6), (5, 14), (6, 10), (6, 14), (6, 9), (6, 19), (7, 17), (7, 15), (7, 19), (8, 19), (8, 14), (8, 10), (8, 18), (8, 16), (9, 12), (9, 18), (9, 14), (9, 19), (10, 11), (10, 19), (10, 15), (10, 13), (11, 12), (11, 19), (11, 16), (12, 17), (12, 18), (12, 13), (13, 15), (13, 16), (13, 18), (15, 18), (15, 17), (16, 19), (18, 19)]
# G = initialize_graph(num_nodes)
# add_fixed_edges(G, edges)

# ----------------------------------------------------------------------------------------
# # Random graph instantiation
num_nodes = 22
G = nx.random_regular_graph(3, num_nodes)
num_edges = G.number_of_edges()
edges = list(G.edges())

# ----------------------------------------------------------------------------------------


adj_mat = nx.to_numpy_array(G)
# print(adj_mat)
xbar = []

data_to_plot = []

steps = p_steps()

#for loop that loops over beta values, exponentially
for beta in tqdm(steps):
    # partition function
    Z_of_beta = 0
    # temporary energy value
    temp_e_of_beta= 0
    # loop through all partitions of the graph
    for xbar_inst in itertools.product([1,-1], repeat = num_nodes):
        xbar_list= list(xbar_inst)
        xbar = np.array(xbar_list)
        # compute hamiltonian 
        h_of_x = 1/2 * (xbar.T @ adj_mat @ xbar)
        Z_of_beta += np.exp(-beta*h_of_x)
        temp_e_of_beta += np.exp(-beta*h_of_x)*h_of_x
    # energy
    E_of_beta = temp_e_of_beta/Z_of_beta
    # entropy 
    S_of_beta_temp = np.log(Z_of_beta) + beta*E_of_beta
    S_of_beta = S_of_beta_temp/np.log(2)
    # append values (as densities)
    data_to_plot.append((S_of_beta/num_nodes, E_of_beta/num_nodes))
    

x_values, y_values = zip(*data_to_plot)
x_values = list(x_values)
y_values = list(y_values)
clean_array1 = np.array(x_values)[~np.isnan(x_values)]
x_values = clean_array1.tolist()
clean_array2 = np.array(y_values)[~np.isnan(y_values)]
y_values = clean_array2.tolist()

# ----------------------------------------------------------------------------------------
# Calculating Energy of SDP Solution

# subset of nodes (solution)
sdp_relaxation_sols= gw(num_nodes, edges)
# edges of the cut
cut = cut(sdp_relaxation_sols, edges)
# cut size
cut_size = len(cut)
# print(cut_size, num_edges)
sdp_energy = (num_edges - 2 * cut_size)/num_nodes
# ----------------------------------------------------------------------------------------
# Extending Straight Line
x_last = x_values[len(x_values)-1]
y_last = y_values[len(y_values)-1]
# print(x_values)
x_2last = x_values[len(x_values)-2]
y_2last = y_values[len(y_values)-2]
m = (find_non_zero_difference(y_values))/(find_non_zero_difference(x_values))
b = y_last - m * x_last
# ----------------------------------------------------------------------------------------
# Figuring out intersection
intersection_exists, intersection_x = check_intersection(x_values, y_values, sdp_energy)
# ----------------------------------------------------------------------------------------
# plotting potential for quantum advantage
fig, ax = plt.subplots()
ax.plot(x_values, y_values)
ax.hlines(sdp_energy, xmin = 0, xmax=1, color='r')
ax.plot([x_last, 0], [y_last, b], ':g')
if(intersection_exists):
    ax.axvline(x=intersection_x, linestyle='--', color='r')

print("Entropy Density Threshold : ",intersection_x)
#  print(intersection_exists)
# if intersection_exists:
#     ax.axvline(x=intersection_x, ymax=sdp_energy, ymin=b)
#     print(sdp_energy, b)
    

# ax.plot(x_space, y_space, color = 'y')

ax.set_xlabel('Entropy Density') 
ax.set_ylabel('Energy Density')  
ax.set_title('Energy Density vs Entropy Density') 
# fig.savefig('8-15.png')
print("done")
plt.show()

# ----------------------------------------------------------------------------------------
#  derive and plot the circuit size threshold above which the quantum advantage is out of reach

# # c is the entropy density threshold
# # c = 0.84 
# c = intersection_x
# p_2 = 0.01  

# # draft paper model for deriving the circuit size threshold
# def circuit_size_threshold(n, D):
#     return (n - 1) * D - (1/(2 * p_2)) * np.log(2**(n - 1) / 2**(n * (1 - c) - 1))

# # Space for plot of width and depth of model
# n_values = np.linspace(0, 100)  #number of qubits
# D_values = np.linspace(0, 40)  #depth of circuit

# n_mesh, D_mesh = np.meshgrid(n_values, D_values)

# # Calculate the inequality values for each combination of n and D
# inequality_values = circuit_size_threshold(n_mesh, D_mesh)

# # Plot the contour plot
# plt.contourf(n_mesh, D_mesh, inequality_values, levels=[-1e10, 0], cmap='viridis', alpha=0.3)
# plt.xlabel('Number of Qubits')
# plt.ylabel('Circuit Depth')
# plt.title('Circuit Size Threshold Above which quantum advantage is out of reach')
# # plt.colorbar(label='(n-1)D - (1/(2p_2)) ln(2^(n-1)/2^(n(1-c) - 1))')
# plt.show()

# # todo
# # run on different graphs, potentially bigger/more complex
# # confirm this equality
# # print a couple graphs for Raul (on bigger graphs?) send off ASAP
# # ask if the negative S (energy) can be expected
# # figure out how to plot the entropy and energy of the brute force solution 
# #       - try and do this before sending to Raul
# # organise meeting to discuss how we can do for tensor networks and potentially google paper graph

