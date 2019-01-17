# Inv-Problem-Solver
Inverse Problem solver, written in Python using Tensorflow

These scripts are based on a research project of Michael Schütt and Dominik Schildknecht (domischi). Due to a lack of time, the scripts will not be developed any further. Hence, also the messy state of the scripts.

The main purpose of the scripts is to train a neural network to invert operations, for which the forward solution is easy to obtain, however, the inversion problem is difficult to solve. For this purpose we studied two examples, namely:
    - Inversion of the 1D heat equation. I.e. given a temperature profile at t=t0>0, how did the heat profile look like at t=0.
    - Extracting the real time Greens function from the Matsubara Greens function.
