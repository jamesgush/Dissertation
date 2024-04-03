import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp2d
import random
import itertools
# import quimb as qu
# from quimb.tensor import TensorNetwork as qtn, Tensor as qt
import numpy as np
# import cvxpy as cp
# from scipy.linalg import sqrtm
from tqdm import tqdm
from gw import calculate_sdp_energy
# import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
# from sklearn.gaussian_process.kernels import RationalQuadratic, Matern, WhiteKernel, ExpSineSquared

def initialize_graph(num_nodes):
    """
    Creates a NetworkX of certain size.

    Parameters:
    - num_nodes (Int): The number of nodes to initialize the graph with.
    - param2 (Type): Description of param2.

    Returns:
    - NetworkX Graph: The initialized graph

    """
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    return G

def add_random_edges(G, num_edges):
    """
    Randomly (i.e. between random nodes) adds a fixed number of edges to a pre-initialised NetworkX graph.

    Parameters:
    - G (NetworkX Graph): The graph to add adges to.
    - num_edges (Int): The number of edges to add.

    Returns:
    - Void: 

    """
    nodes = list(G.nodes())
    for _ in range(num_edges):
        u, v = random.sample(nodes, 2)
        G.add_edge(u, v)

def add_fixed_edges(G, edges):
    """
    Adds a fixed, predisclosed list of edges to a pre-initialised NetworkX graph.

    Parameters:
    - G (NetworkX Graph): The graph to add adges to.
    - edges (list): The list of edges to add.

    Returns:
    - Void: 

    """
    for edge in edges:
        u = edge[0]
        v = edge[1]
        G.add_edge(u, v)

def visualize_graph(G):
    """
    Visually plots a NetworkX graph.

    Parameters:
    - G (NetworkX Graph): The graph to plot.

    Returns:
    - Void: 

    """
    nx.draw(G, with_labels=True)
    plt.show()

def max_cut_brute_force(G):
    """
    Using brute force, calculates the max-cut and cut value of given graph.

    Parameters:
    - G (NetworkX Graph): The graph to calculate cut of.

    Returns:
    - list: max-cut
    - Int: value of the max cut

    """
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
    """
    Visually displays the 2 independent partitions of the graph of a cut solution

    Parameters:
    - G (NetworkX Graph): The graph to display.
    - max-cut (list): The cut.

    """
    color_map = ['red' if node in max_cut else 'blue' for node in G.nodes()]
    nx.draw(G, node_color=color_map, with_labels=True, font_weight='bold')
    plt.show()

def find_non_zero_difference(points):
    """
    Finds the first non-zero difference between consecutive elements in a list, searching in reverse order

    Parameters:
    - points (List[float|int]): A list of numeric values (either integers or floats).

    Returns:
    - float|int|None: The first non-zero difference. Returns None if all differences 
                        are zero or if the list has fewer than two elements.
    
    Notes:
    - Used in computing the gradient of the Gibbs Distribution, as the computational accuracy 
        is a limiting factor and the gradient cannot be positive or horizontal
    """
    for i in range(len(points) - 1, 0, -1):
        difference = points[i] - points[i-1]
        if difference != 0:
            return difference
    return None

def check_intersection(x, y, horizontal_line):
    """
    Checks if the any lines between a list of x and y coordinates intersects with a predefined horizontal line, and calculate the 
    x-coordinate of this intersection if there is one.


    Parameters:
    - x (List[float|int]): A list of x-coordinates of the polyline's points.
    - y (List[float|int]): A list of y-coordinates of the polyline's points.
    - horizontal_line (float|int): The y-coordinate of the horizontal line.

    Returns:
    - Tuple[bool, float|int|None]: A tuple where the first element is a bool representing if there is an intersection, 
      and the second element is the x-coordinate of the first intersection point if an intersection exists; 
    - None otherwise.
    """
    for i in range(len(x) - 1):
        x1, x2 = x[i], x[i + 1]
        y1, y2 = y[i], y[i + 1]

        if (y1 <= horizontal_line <= y2) or (y2 <= horizontal_line <= y1):
            intersection_x = x1 + (x2 - x1) * (horizontal_line - y1) / (y2 - y1)
            return True, intersection_x

    return False, None
    
def p_steps():
    """
    Customizable number space

    - Returns: a personalised number space used represent beta values in the Gibbs Distribution
    - Notes: Currently takes 100 steps, increases in a linear fashion for first 20%, then in a quadratic 
        fasion for next 20% and finally exponentially for the last 60%
    
    """
    num_steps = 100
    lin = 0.2
    quad = 0.2
    exp = 0.6

    linear_steps = int(num_steps * lin)
    quadratic_steps = int(num_steps * quad)
    exponential_steps = int(num_steps * exp)

    # Linear progression from 0 to 1 (normalized, we will scale later)
    linear_range = np.linspace(0, 1, linear_steps, endpoint=False)

    # Quadratic progression, making sure it starts from the end of the linear range
    quadratic_start = linear_range[-1] if linear_steps > 0 else 0
    quadratic_range = (np.linspace(0, 1, quadratic_steps, endpoint=False) ** 2) * (2 - quadratic_start) + quadratic_start

    # Exponential progression, ensuring it starts from the end of the quadratic range
    exponential_start = quadratic_range[-1] if quadratic_steps > 0 else 0
    # We aim for the exponential end to be around 10000, adjust base accordingly
    exponential_end = 10000
    exponential_range = np.logspace(np.log10(exponential_start + 1), np.log10(exponential_end), exponential_steps, endpoint=True) - 1

    # Combine all ranges
    steps = np.concatenate((linear_range, quadratic_range, exponential_range))

    # Scale linear and quadratic parts to fit into the desired progression
    # Since exponential part is already scaled towards 10000, we focus on scaling linear and quadratic parts proportionally
    max_linear_quadratic = np.max(np.concatenate((linear_range, quadratic_range)))
    scaling_factor = exponential_range[0] / max_linear_quadratic if max_linear_quadratic != 0 else 1
    steps[:linear_steps+quadratic_steps] *= scaling_factor

    # Ensure the sequence starts with 0 explicitly and ends with the desired final value
    steps = np.insert(steps, 0, 0)
    
    return steps


# ----------------------------------------------------------------------------------------
# Initialise the problem graph, which in this case has 22 nodes and order 3 
num_nodes = 22
# G = nx.random_regular_graph(3, num_nodes)
G = nx.complete_graph(num_nodes)
num_edges = G.number_of_edges()
edges = list(G.edges())
# visualize_graph(G)

# ----------------------------------------------------------------------------------------
# Calculate the Gibbs Distribution, by 'brute-forcing' the partition function,
# and calculating the energy and entropy densities (given by Helmholtz free energy theorem)

adj_mat = nx.to_numpy_array(G)
xbar = []
data_to_plot = []

# personalised step to loop through beta (inverse temperature)
# number space starts linear, then goes quadratic and then exponential
steps = p_steps()

#calculates energy and entropy density values for given beta
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
    # energy value
    E_of_beta = temp_e_of_beta/Z_of_beta
    # entropy (given by Helmholtz free energy theorem)
    S_of_beta_temp = np.log(Z_of_beta) + beta*E_of_beta
    S_of_beta = S_of_beta_temp/np.log(2)
    # append values (as densities i.e. divided by problem graph size)
    data_to_plot.append((S_of_beta/num_nodes, E_of_beta/num_nodes))
    
# append cleaned versions of energy densities and entropy densities to lists
x_values, y_values = zip(*data_to_plot)
x_values = list(x_values)
y_values = list(y_values)
clean_array1 = np.array(x_values)[~np.isnan(x_values)]
x_values = clean_array1.tolist()
clean_array2 = np.array(y_values)[~np.isnan(y_values)]
y_values = clean_array2.tolist()

# ----------------------------------------------------------------------------------------
# Calculating Energy of SDP Solution using Goemanns-Williamson Approximation Algorithm
sum_sdp_energies = 0
max_sdp_energy = 0
for i in range(100):
    sdp_energy = calculate_sdp_energy(num_nodes, edges)
    sum_sdp_energies += sdp_energy
    if sdp_energy < max_sdp_energy:
        max_sdp_energy = sdp_energy
average_sdp_energy = sum_sdp_energies/100
sdp_energy = average_sdp_energy
print("Avergae SDP:", average_sdp_energy)
print("Max SDP:",max_sdp_energy)

# ----------------------------------------------------------------------------------------
# Extend straight Line from Gibbs Distribution to Y-Axis, (which estimates quantum solution)
x_last = x_values[len(x_values)-1]
y_last = y_values[len(y_values)-1]
x_2last = x_values[len(x_values)-2]
y_2last = y_values[len(y_values)-2]
m = (find_non_zero_difference(y_values))/(find_non_zero_difference(x_values))
b = y_last - m * x_last

# ----------------------------------------------------------------------------------------
# interpolating with gaussian processes ****TODO******
# print(x_values)
# print(y_values)
# x_values_no_0  = [x for x in x_values if x != 0]
# y_values_no_0 = [x for x in y_values if x != 0]

x_data = np.array(x_values)
y_data = np.array(y_values)

# # ----------------------------------------------------------------------------------------
# Compute intersection of Gibbs Distribution and SDP Classical Solution
# This represents the Entropy Density threshold to the right of which the quantum advantage
#   is unattainable
intersection_exists, intersection_x = check_intersection(x_values, y_values, sdp_energy)
# ----------------------------------------------------------------------------------------
# Plot Gibbs Distribution, SDP Solution, and their intersection (and potentially interpolation TODO)
fig, ax = plt.subplots()
ax.plot(x_values, y_values)
# ax.plot(x_values_new, y_values_new, color='red')
# ax.plot(x_pred, y_pred, color='orange', label='Predicted Curve')
ax.hlines(sdp_energy, xmin = 0, xmax=1, linestyle = '--', color='goldenrod')
ax.plot([x_last, 0], [y_last, b], ':g')
# if(intersection_exists):
#     ax.axvline(x=intersection_x, linestyle='dashdot', color='slategrey')
#     trans = ax.get_xaxis_transform()
#     rounded = round(intersection_x,3)
#     str_round = "[" + str(rounded) + "]"
#     ax.text(intersection_x+0.01, 0, str_round, transform=trans, fontsize='small')

print("Entropy Density Threshold : ",intersection_x)

ax.set_xlabel('Entropy Density') 
ax.set_ylabel('Energy Density')  
ax.set_title('Energy Density vs Entropy Density') 
# ax.legend(['Gibbs Distribution', 'SDP Solver', 'Entropy Density Threshold'])
# fig.savefig('8-15.png')
print("done")
plt.show()


print(y_values)
# # ----------------------------------------------------------------------------------------
# Compute intersection of Gibbs Distribution and SDP Classical Solution
# This represents the Entropy Density threshold to the right of which the quantum advantage
#   is unattainable
intersection_exists, intersection_x = check_intersection(x_values, y_values, sdp_energy)
# ----------------------------------------------------------------------------------------
# Plot Gibbs Distribution, SDP Solution, and their intersection (and potentially interpolation TODO)
fig, ax = plt.subplots()
ax.plot(x_values, y_values)
# ax.plot(x_values_new, y_values_new, color='red')
# ax.plot(x_pred, y_pred, color='orange', label='Predicted Curve')
ax.hlines(sdp_energy, xmin = 0, xmax=1, linestyle = '--', color='goldenrod')
ax.plot([x_last, 0], [y_last, b], ':g')
# if(intersection_exists):
#     ax.axvline(x=intersection_x, linestyle='dashdot', color='slategrey')
#     trans = ax.get_xaxis_transform()
#     rounded = round(intersection_x,3)
#     str_round = "[" + str(rounded) + "]"
#     ax.text(intersection_x+0.01, 0, str_round, transform=trans, fontsize='small')

print("Entropy Density Threshold : ",intersection_x)

ax.set_xlabel('Entropy Density') 
ax.set_ylabel('Energy Density')  
ax.set_title('Energy Density vs Entropy Density') 
# ax.legend(['Gibbs Distribution', 'SDP Solver', 'Entropy Density Threshold'])
# fig.savefig('8-15.png')
print("done")
plt.show()


print(y_values)
# # ----------------------------------------------------------------------------------------
# Compute intersection of Gibbs Distribution and SDP Classical Solution
# This represents the Entropy Density threshold to the right of which the quantum advantage
#   is unattainable
intersection_exists, intersection_x = check_intersection(x_values, y_values, sdp_energy)
# ----------------------------------------------------------------------------------------
# Plot Gibbs Distribution, SDP Solution, and their intersection (and potentially interpolation TODO)
fig, ax = plt.subplots()
ax.plot(x_values, y_values)
# ax.plot(x_values_new, y_values_new, color='red')
# ax.plot(x_pred, y_pred, color='orange', label='Predicted Curve')
ax.hlines(sdp_energy, xmin = 0, xmax=1, linestyle = '--', color='goldenrod')
ax.plot([x_last, 0], [y_last, b], ':g')
# if(intersection_exists):
#     ax.axvline(x=intersection_x, linestyle='dashdot', color='slategrey')
#     trans = ax.get_xaxis_transform()
#     rounded = round(intersection_x,3)
#     str_round = "[" + str(rounded) + "]"
#     ax.text(intersection_x+0.01, 0, str_round, transform=trans, fontsize='small')

print("Entropy Density Threshold : ",intersection_x)

ax.set_xlabel('Entropy Density') 
ax.set_ylabel('Energy Density')  
ax.set_title('Energy Density vs Entropy Density') 
# ax.legend(['Gibbs Distribution', 'SDP Solver', 'Entropy Density Threshold'])
# fig.savefig('8-15.png')
print("done")
plt.show()


print(y_values)
# # ----------------------------------------------------------------------------------------
# Compute intersection of Gibbs Distribution and SDP Classical Solution
# This represents the Entropy Density threshold to the right of which the quantum advantage
#   is unattainable
intersection_exists, intersection_x = check_intersection(x_values, y_values, sdp_energy)
# ----------------------------------------------------------------------------------------
# Plot Gibbs Distribution, SDP Solution, and their intersection (and potentially interpolation TODO)
fig, ax = plt.subplots()
ax.plot(x_values, y_values)
# ax.plot(x_values_new, y_values_new, color='red')
# ax.plot(x_pred, y_pred, color='orange', label='Predicted Curve')
ax.hlines(sdp_energy, xmin = 0, xmax=1, linestyle = '--', color='goldenrod')
ax.plot([x_last, 0], [y_last, b], ':g')
# if(intersection_exists):
#     ax.axvline(x=intersection_x, linestyle='dashdot', color='slategrey')
#     trans = ax.get_xaxis_transform()
#     rounded = round(intersection_x,3)
#     str_round = "[" + str(rounded) + "]"
#     ax.text(intersection_x+0.01, 0, str_round, transform=trans, fontsize='small')

print("Entropy Density Threshold : ",intersection_x)

ax.set_xlabel('Entropy Density') 
ax.set_ylabel('Energy Density')  
ax.set_title('Energy Density vs Entropy Density') 
# ax.legend(['Gibbs Distribution', 'SDP Solver', 'Entropy Density Threshold'])
# fig.savefig('8-15.png')
print("done")
plt.show()


print(y_values)
# # ----------------------------------------------------------------------------------------
# Compute intersection of Gibbs Distribution and SDP Classical Solution
# This represents the Entropy Density threshold to the right of which the quantum advantage
#   is unattainable
intersection_exists, intersection_x = check_intersection(x_values, y_values, sdp_energy)
# ----------------------------------------------------------------------------------------
# Plot Gibbs Distribution, SDP Solution, and their intersection (and potentially interpolation TODO)
fig, ax = plt.subplots()
ax.plot(x_values, y_values)
# ax.plot(x_values_new, y_values_new, color='red')
# ax.plot(x_pred, y_pred, color='orange', label='Predicted Curve')
ax.hlines(sdp_energy, xmin = 0, xmax=1, linestyle = '--', color='goldenrod')
ax.plot([x_last, 0], [y_last, b], ':g')
# if(intersection_exists):
#     ax.axvline(x=intersection_x, linestyle='dashdot', color='slategrey')
#     trans = ax.get_xaxis_transform()
#     rounded = round(intersection_x,3)
#     str_round = "[" + str(rounded) + "]"
#     ax.text(intersection_x+0.01, 0, str_round, transform=trans, fontsize='small')

print("Entropy Density Threshold : ",intersection_x)

ax.set_xlabel('Entropy Density') 
ax.set_ylabel('Energy Density')  
ax.set_title('Energy Density vs Entropy Density') 
# ax.legend(['Gibbs Distribution', 'SDP Solver', 'Entropy Density Threshold'])
# fig.savefig('8-15.png')
print("done")
plt.show()


print(y_values)
# # ----------------------------------------------------------------------------------------
# Compute intersection of Gibbs Distribution and SDP Classical Solution
# This represents the Entropy Density threshold to the right of which the quantum advantage
#   is unattainable
intersection_exists, intersection_x = check_intersection(x_values, y_values, sdp_energy)
# ----------------------------------------------------------------------------------------
# Plot Gibbs Distribution, SDP Solution, and their intersection (and potentially interpolation TODO)
fig, ax = plt.subplots()
ax.plot(x_values, y_values)
# ax.plot(x_values_new, y_values_new, color='red')
# ax.plot(x_pred, y_pred, color='orange', label='Predicted Curve')
ax.hlines(sdp_energy, xmin = 0, xmax=1, linestyle = '--', color='goldenrod')
ax.plot([x_last, 0], [y_last, b], ':g')
# if(intersection_exists):
#     ax.axvline(x=intersection_x, linestyle='dashdot', color='slategrey')
#     trans = ax.get_xaxis_transform()
#     rounded = round(intersection_x,3)
#     str_round = "[" + str(rounded) + "]"
#     ax.text(intersection_x+0.01, 0, str_round, transform=trans, fontsize='small')

print("Entropy Density Threshold : ",intersection_x)

ax.set_xlabel('Entropy Density') 
ax.set_ylabel('Energy Density')  
ax.set_title('Energy Density vs Entropy Density') 
# ax.legend(['Gibbs Distribution', 'SDP Solver', 'Entropy Density Threshold'])
# fig.savefig('8-15.png')
print("done")
plt.show()


print(y_values)
# # ----------------------------------------------------------------------------------------
# Compute intersection of Gibbs Distribution and SDP Classical Solution
# This represents the Entropy Density threshold to the right of which the quantum advantage
#   is unattainable
intersection_exists, intersection_x = check_intersection(x_values, y_values, sdp_energy)
# ----------------------------------------------------------------------------------------
# Plot Gibbs Distribution, SDP Solution, and their intersection (and potentially interpolation TODO)
fig, ax = plt.subplots()
ax.plot(x_values, y_values)
# ax.plot(x_values_new, y_values_new, color='red')
# ax.plot(x_pred, y_pred, color='orange', label='Predicted Curve')
ax.hlines(sdp_energy, xmin = 0, xmax=1, linestyle = '--', color='goldenrod')
ax.plot([x_last, 0], [y_last, b], ':g')
# if(intersection_exists):
#     ax.axvline(x=intersection_x, linestyle='dashdot', color='slategrey')
#     trans = ax.get_xaxis_transform()
#     rounded = round(intersection_x,3)
#     str_round = "[" + str(rounded) + "]"
#     ax.text(intersection_x+0.01, 0, str_round, transform=trans, fontsize='small')

print("Entropy Density Threshold : ",intersection_x)

ax.set_xlabel('Entropy Density') 
ax.set_ylabel('Energy Density')  
ax.set_title('Energy Density vs Entropy Density') 
# ax.legend(['Gibbs Distribution', 'SDP Solver', 'Entropy Density Threshold'])
# fig.savefig('8-15.png')
print("done")
plt.show()


print(y_values)
# # ----------------------------------------------------------------------------------------
# Compute intersection of Gibbs Distribution and SDP Classical Solution
# This represents the Entropy Density threshold to the right of which the quantum advantage
#   is unattainable
intersection_exists, intersection_x = check_intersection(x_values, y_values, sdp_energy)
# ----------------------------------------------------------------------------------------
# Plot Gibbs Distribution, SDP Solution, and their intersection (and potentially interpolation TODO)
fig, ax = plt.subplots()
ax.plot(x_values, y_values)
# ax.plot(x_values_new, y_values_new, color='red')
# ax.plot(x_pred, y_pred, color='orange', label='Predicted Curve')
ax.hlines(sdp_energy, xmin = 0, xmax=1, linestyle = '--', color='goldenrod')
ax.plot([x_last, 0], [y_last, b], ':g')
# if(intersection_exists):
#     ax.axvline(x=intersection_x, linestyle='dashdot', color='slategrey')
#     trans = ax.get_xaxis_transform()
#     rounded = round(intersection_x,3)
#     str_round = "[" + str(rounded) + "]"
#     ax.text(intersection_x+0.01, 0, str_round, transform=trans, fontsize='small')

print("Entropy Density Threshold : ",intersection_x)

ax.set_xlabel('Entropy Density') 
ax.set_ylabel('Energy Density')  
ax.set_title('Energy Density vs Entropy Density') 
# ax.legend(['Gibbs Distribution', 'SDP Solver', 'Entropy Density Threshold'])
# fig.savefig('8-15.png')
print("done")
plt.show()


print(y_values)
# # ----------------------------------------------------------------------------------------
# Compute intersection of Gibbs Distribution and SDP Classical Solution
# This represents the Entropy Density threshold to the right of which the quantum advantage
#   is unattainable
intersection_exists, intersection_x = check_intersection(x_values, y_values, sdp_energy)
# ----------------------------------------------------------------------------------------
# Plot Gibbs Distribution, SDP Solution, and their intersection (and potentially interpolation TODO)
fig, ax = plt.subplots()
ax.plot(x_values, y_values)
# ax.plot(x_values_new, y_values_new, color='red')
# ax.plot(x_pred, y_pred, color='orange', label='Predicted Curve')
ax.hlines(sdp_energy, xmin = 0, xmax=1, linestyle = '--', color='goldenrod')
ax.plot([x_last, 0], [y_last, b], ':g')
# if(intersection_exists):
#     ax.axvline(x=intersection_x, linestyle='dashdot', color='slategrey')
#     trans = ax.get_xaxis_transform()
#     rounded = round(intersection_x,3)
#     str_round = "[" + str(rounded) + "]"
#     ax.text(intersection_x+0.01, 0, str_round, transform=trans, fontsize='small')

print("Entropy Density Threshold : ",intersection_x)

ax.set_xlabel('Entropy Density') 
ax.set_ylabel('Energy Density')  
ax.set_title('Energy Density vs Entropy Density') 
# ax.legend(['Gibbs Distribution', 'SDP Solver', 'Entropy Density Threshold'])
# fig.savefig('8-15.png')
print("done")
plt.show()


print(y_values)
# # ----------------------------------------------------------------------------------------
# Compute intersection of Gibbs Distribution and SDP Classical Solution
# This represents the Entropy Density threshold to the right of which the quantum advantage
#   is unattainable
intersection_exists, intersection_x = check_intersection(x_values, y_values, sdp_energy)
# ----------------------------------------------------------------------------------------
# Plot Gibbs Distribution, SDP Solution, and their intersection (and potentially interpolation TODO)
fig, ax = plt.subplots()
ax.plot(x_values, y_values)
# ax.plot(x_values_new, y_values_new, color='red')
# ax.plot(x_pred, y_pred, color='orange', label='Predicted Curve')
ax.hlines(sdp_energy, xmin = 0, xmax=1, linestyle = '--', color='goldenrod')
ax.plot([x_last, 0], [y_last, b], ':g')
# if(intersection_exists):
#     ax.axvline(x=intersection_x, linestyle='dashdot', color='slategrey')
#     trans = ax.get_xaxis_transform()
#     rounded = round(intersection_x,3)
#     str_round = "[" + str(rounded) + "]"
#     ax.text(intersection_x+0.01, 0, str_round, transform=trans, fontsize='small')

print("Entropy Density Threshold : ",intersection_x)

ax.set_xlabel('Entropy Density') 
ax.set_ylabel('Energy Density')  
ax.set_title('Energy Density vs Entropy Density') 
# ax.legend(['Gibbs Distribution', 'SDP Solver', 'Entropy Density Threshold'])
# fig.savefig('8-15.png')
print("done")
plt.show()


