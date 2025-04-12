import numpy as np

length_units = "um"
XI = 0.0062
LONDONL = 0.2
D = 0.01
MU0 = 4 * np.pi * 1e-7
RESISTIVITY = 2.5
SIGMA = 1 / RESISTIVITY
H = 6.62607015e-34
E = 1.602176634e-19
PHI0 = H / (2 * E)
TAU0 = MU0 * SIGMA * LONDONL**2
B0 = PHI0 / (2 * np.pi * XI**2)
A0 = XI * B0
J0 = (4 * XI * B0) / (MU0 * LONDONL**2)
K0 = J0 * D
V0 = XI * J0 / SIGMA
