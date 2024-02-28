import numpy as np
import matplotlib.pyplot as plt

c = 0.84 
p_2 = 0.01  

# draft paper model
def inequality_curve(n, D):
    return (n - 1) * D - (1/(2 * p_2)) * np.log(2**(n - 1) / 2**(n * (1 - c) - 1))

# Space for plot of width and depth of model
n_values = np.linspace(0, 100)  #number of qubits
D_values = np.linspace(0, 40)  #depth of circuit

n_mesh, D_mesh = np.meshgrid(n_values, D_values)

# Calculate the inequality values for each combination of n and D
inequality_values = inequality_curve(n_mesh, D_mesh)

# Plot the contour plot
plt.contourf(n_mesh, D_mesh, inequality_values, levels=[-1e10, 0], cmap='viridis', alpha=0.3)
plt.xlabel('n')
plt.ylabel('D')
plt.title('Inequality Curve Plot')
# plt.colorbar(label='(n-1)D - (1/(2p_2)) ln(2^(n-1)/2^(n(1-c) - 1))')
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# # Given values for c
# c = 0.84  # Replace with your actual value for c

# # Define the inequality function
# def inequality_curve(n, D, p_2):
#     return (n - 1) * D - (1/(2 * p_2)) * np.log2(2**(n - 1) / 2**(n * (1 - c) - 1))

# # Generate values for n and D
# n_values = np.linspace(0, 100, 100)  # Adjust the range and granularity as needed
# D_values = np.linspace(0, 40, 100)  # Adjust the range and granularity as needed

# # Generate different values for p_2
# p_2_values = [0.1, 0.01, 0.001, 0.0001, 0.00001]

# # Create subplots for each value of p_2
# fig, axes = plt.subplots(nrows=1, ncols=len(p_2_values), figsize=(15, 4), sharey=True)

# # Plot each curve with a different color
# for i, p_2 in enumerate(p_2_values):
#     # Create a meshgrid for n and D
#     n_mesh, D_mesh = np.meshgrid(n_values, D_values)

#     # Calculate the inequality values for each combination of n and D
#     inequality_values = inequality_curve(n_mesh, D_mesh, p_2)

#     # Plot the contour plot on the respective subplot
#     contour = axes[i].contourf(n_mesh, D_mesh, inequality_values, levels=[-1e10, 0], cmap='viridis', alpha=0.3)
#     axes[i].set_title(f'$p_2 = {p_2}$')
#     axes[i].set_xlabel('n')
#     if i == 0:
#         axes[i].set_ylabel('D')

# # Add a colorbar for the last subplot
# fig.colorbar(contour, ax=axes, orientation='vertical', label='(n-1)D - (1/(2p_2)) ln(2^(n-1)/2^(n(1-c) - 1))')

# plt.suptitle('Inequality Curve Plots for Different p_2 Values', y=1.02)
# plt.tight_layout()
# plt.show()
