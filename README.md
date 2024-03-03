# Is the Quantum Advantage reachable for the Max-Cut Problem
This repository contains files used for testing and evaluation for my dissertation. It benchmarks quantum algorithms against the classical like (I decided to use the Goemans Williamson algorithm, the best known classical approximation algorithm of the Max-Cut problem).

## Introduction

mc_bruteforce.py contains the main implementation of my project. 
personalised_steps.py is used to generate a number space used to best visualise the several graphs I have generated
gw_mc.py implements the Goemans Williamson Approximation Algorithm. 
boundary_num_qubits_depth.py derive and plot the circuit size threshold above which the quantum advantage is out of reach

## How to Use

Outline the steps or instructions on how to use your project. Include prerequisites, installation steps, configuration, and any other important information users need to get started.

### Example:

1. **Step 1:** Clone the repository
   ```bash
   git clone https://github.com/your-username/your-project.git
