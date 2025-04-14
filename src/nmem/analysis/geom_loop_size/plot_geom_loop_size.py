import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.cm import get_cmap

# Load all .mat files in the current directory
loop_sizes = np.arange(1.7, 5.2, 0.5)
def import_mat_list(path):
    files = sorted([f for f in os.listdir(path) if f.endswith(".mat")])
    data = [sio.loadmat(os.path.join(path, f), squeeze_me=True, struct_as_record=False) for f in files]
    return [d['data'] if 'data' in d else d for d in data]

# Load data
data = import_mat_list(os.getcwd())
N = len(data)
NMEAS=1000
# Define colors (similar to 'MIndexed10', using color cycle here)
cmap = get_cmap("tab10")

# First plot: Vch vs. ber_est
plt.figure(1)
for i in range(N):
    Vch = np.ravel(data[i]["Vch"])
    ber_est = np.ravel(data[i]["ber_est"])
    err = np.sqrt(ber_est * (1 - ber_est) / NMEAS)
    plt.errorbar(Vch, ber_est, yerr=err, fmt='o', color=cmap(i), label=f"_Loop Size {loop_sizes[i]:.1f} au")
    plt.plot(Vch, ber_est, color=cmap(i), label=f"Loop Size {loop_sizes[i]:.1f} $\mu$A")
plt.yscale("log")
plt.xlabel("channel voltage (mV)")
plt.ylabel("estimated BER")
plt.legend(loc="lower left")

# Second plot: best BER vs loop size
best_ber = [np.min(np.ravel(d["ber_est"])) for d in data]
plt.figure(2)
plt.plot(range(1, N + 1), best_ber, '-o')
plt.yscale("log")
plt.xlabel("loop size (au)")
plt.ylabel("best BER")

plt.show()
