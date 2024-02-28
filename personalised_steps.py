import numpy as np
import matplotlib.pyplot as plt

def p_steps():
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

def exp_steps():
    steps_01 = np.logspace(np.log(1), np.log10(10000), 50) - 1
    np.insert(steps_01, 0, 0) 
    return steps_01

