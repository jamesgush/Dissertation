# Is the Quantum Advantage reachable for the Max-Cut Problem
This repository contains files used for testing and evaluation for my dissertation. It benchmarks quantum algorithms against the classical like (I decided to use the Goemans Williamson algorithm, the best known classical approximation algorithm of the Max-Cut problem).

## Introduction

gibbs_dist.py contains implementation for the derivation of the Gibbs State Boundary, as well as the computation of the SDP Solution and the entropy density threshold 
gw.py implements the Goemans Williamson Approximation Algorithm. This was adapted from Lalovic (2022)
circ_size_thresh.py derive and plot the circuit size threshold above which the quantum advantage is out of reach. It is also used to decipher how many layers of the Quantum Approximate Optimisation Algorithm can be performed

## Prerequisites

pip install cvxpy networkx tqdm matplotlib numpy

### Installation:

1. **Step 1:** Clone the repository
   ```bash
   git clone https://github.com/your-username/your-project.git
2. **Step 2:** Install the packages (in prerequisites)

### Usage:
**Plotting Gibbs State Boundary**
Edit graph size, as  well as connectivity by adjusting the clearly labeled parameters.
**Bounding Circuit Size**
1. Using entropy density threshold output from gibbs_dist.py solution, input it as a the e_d_thresh parameter
2. Change the variable name p_2 to edit the probability of error of two qubit gates
3. Adjust the size of the plot, by changing N-values and D_values (representing the number of qubits and the depth of the circuit)

### Contact
James Gush - s2095346@ed.ac.uk.

