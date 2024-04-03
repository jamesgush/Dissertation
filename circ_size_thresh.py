import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec 



# The entropy density threshold
# e_d_thresh = 0.3560121008525872
# sk_8_nodes
e_d_thresh = 0.9526940575887386
c = 1 - e_d_thresh
# 2 qubit gate depolarizing noise 
# less than 1 is reasonable for current qpu
# test in order of magnitudes of 10 to look into the future
p_2_1 = 0.1
p_2_01 = 0.01 
p_2_001 = 0.001
p_2_0001 = 0.0001


# circuit size threshold
def circ_size_thresh(n, D, p2):
    return (n - 1) * D - (1/(2 * p2)) * np.log(2**(n - 1) / 2**(n * (1 - c) - 1))


# Space for plot of width and depth of model respectively
n_values = np.linspace(1,500)
D_values = np.linspace(0,1000)  

n_mesh, D_mesh = np.meshgrid(n_values, D_values)

# Calculate the circuit size threshold for each combination of n and D
inequality_values_1 = circ_size_thresh(n_mesh, D_mesh, p_2_1)
inequality_values_01 = circ_size_thresh(n_mesh, D_mesh, p_2_01)
inequality_values_001 = circ_size_thresh(n_mesh, D_mesh, p_2_001)
inequality_values_0001 = circ_size_thresh(n_mesh, D_mesh, p_2_0001)
# Plot the contour plot
contour_lines_1 = plt.contour(n_mesh, D_mesh, inequality_values_1, levels=[0], colors='r')
contour_lines_01 = plt.contour(n_mesh, D_mesh, inequality_values_01, levels=[0], colors='g')
contour_lines_001 = plt.contour(n_mesh, D_mesh, inequality_values_001, levels=[0], colors='y')
contour_lines_0001 = plt.contour(n_mesh, D_mesh, inequality_values_0001, levels=[0], colors='b')
plt.close()

# Extract the boundary points from the contour lines
boundary_points_1 = contour_lines_1.collections[0].get_paths()[0].vertices
boundary_points_01 = contour_lines_01.collections[0].get_paths()[0].vertices
boundary_points_001 = contour_lines_001.collections[0].get_paths()[0].vertices
boundary_points_0001 = contour_lines_0001.collections[0].get_paths()[0].vertices

boundary_x_1 = boundary_points_1[:, 0]
boundary_y_1 = boundary_points_1[:, 1]

boundary_x_01 = boundary_points_01[:, 0]
boundary_y_01 = boundary_points_01[:, 1]

boundary_x_001 = boundary_points_001[:, 0]
boundary_y_001 = boundary_points_001[:, 1]

boundary_x_0001 = boundary_points_0001[:, 0]
boundary_y_0001 = boundary_points_0001[:, 1]

# # calculating the limit
# tail_size = 5  # Adjust the size of the tail based on your data
# x_tail = boundary_x[-tail_size:].reshape(-1, 1)
# y_tail = boundary_y[-tail_size:].reshape(-1, 1)

# # Fit linear regression model
# model = LinearRegression().fit(x_tail, y_tail)

# # Get the slope of the regression line
# slope = model.coef_[0][0]

# # Extrapolate to estimate the y value as x approaches infinity
# y_at_infinity = model.predict(np.array([[np.inf]]))[0][0]
# print(y_at_infinity)


# If we were to simply plot pts, we'd lose most of the interesting
# details due to the outliers. So let's 'break' or 'cut-out' the x-axis
# into two portions - use the left (ax1) for the outliers, and the right
# (ax2) for the details of the majority of our data


fig = plt.figure(figsize=(10, 5))
gs = gridspec.GridSpec(1, 2, width_ratios=[7, 1])

# Create subplots using gridspec
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])

scaling_factor_01_3 = 1/ (3 * boundary_x_01)
scaling_factor_001_3 = 1/ (3 * boundary_x_001)
scaling_factor_0001_3 = 1/ (3 * boundary_x_0001)


scaled_y_01 = boundary_y_01 * scaling_factor_01_3
scaled_y_001 = boundary_y_001 * scaling_factor_001_3
scaled_y_0001 = boundary_y_0001 * scaling_factor_0001_3

# Plot the same data on both axes
# ax1.plot(boundary_x_1, boundary_y_1)
# ax1.plot(boundary_x_1, scaled_y_01)
# ax1.plot(boundary_x_01, scaled_y_01)
ax1.plot(boundary_x_0001, scaled_y_0001)
# ax2.plot(boundary_x_1, boundary_y_1)
# ax2.plot(boundary_x_01, scaled_y_01)
ax2.plot(boundary_x_0001, scaled_y_0001)

ax1.hlines(10, xmin = 0, xmax = 10, linestyle = '--', color='goldenrod')


ax1.set_yscale('log')
ax2.set_yscale('log')
# ax1.set_xscale('log')
# ax2.set_xscale('log')
# plotting 3n and sqrt(7n) lines NB*** may need work TODO
# Plot the 3n line
# ax1.plot(n_values, line_3n(n_values), label='3n', color='red')
# ax2.plot(n_values, line_3n(n_values), label='3n', color='red')
# Plot the sqrt(7n) line
# ax1.plot(n_values, line_sqrt_7n(n_values), label=r'$\sqrt{7n}$', color='blue')
# ax2.plot(n_values, line_sqrt_7n(n_values), label=r'$\sqrt{7n}$', color='blue')

# Zoom-in / limit the view to different portions of the data
ax1.set_xlim(0, 10)  # outliers only
ax2.set_xlim(19, 20)  # most of the data
ax1.set_ylim(0,1000)
ax2.set_ylim(0,1000)


# Hide the spines between ax and ax2
ax1.spines.right.set_visible(False)
ax2.spines.left.set_visible(False)
ax1.yaxis.tick_left()
ax1.tick_params()  # don't put tick labels at the left
ax2.yaxis.tick_right()
ax1.set_xlabel("Num Qubits")
ax1.set_ylabel("Num QAOA Layers")
ax1.set_title("Bounding Circuit Size")

# Now, let's turn towards the cut-out slanted lines.
# We create line objects in axes coordinates, in which (0,0), (0,1),
# (1,0), and (1,1) are the four corners of the axes.
# The slanted lines themselves are markers at those locations, such that the
# lines keep their angle and position, independent of the axes size or scale
# Finally, we need to disable clipping.

d = 0.5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-d, -1), (d, 1)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([1, 1], [0, 1], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 0], [1, 0], transform=ax2.transAxes, **kwargs)
ax1.legend(['0.01% Probability of Error', 'Min QAOA Layers'])
plt.show()


# xs = np.array(boundary_x)
# ys = np.array(boundary_y)

# # y_value = np.interp(5, boundary_x, boundary_y)

# num_qubits = [1, 2, 3, 4, 5, 6, 7, 10, 20]
# # 3-reg
# # 0.001: 57.14, 28.57, 15.94, 10.64, 7.98, 6.38, 5.32, 3.55, 1.68
# # 0.01: 57.14, 3.24, 1.62, 1.14, 0.88, 0.71, 0.59, 0.37, 0.17
# # 0.1: 57.14, 0.57, 0.37, 0.27, 0.21, 0.17, 0.14, 0.08, 0.02

# # sk
# # 0.001: 57.14, 2.35, 1.18, 0.86, 0.68, 0.56, 0.47, 0.31, 0.13
# # 0.01: 57.14, 0.57, 0.37, 0.26, 0.2, 0.16, 0.13, 0.08, 0.02
# # 0.1: 57.14, 0.56, 0.36, 0.25, 0.19, 0.15, 0.12, 0.07, 0.01


    
    

