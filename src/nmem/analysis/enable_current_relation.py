import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os

def find_peak(data: dict):
    x = data["x"][0][:, 1] * 1e6
    y = data["y"][0][:, 0] * 1e6

    w0r1 = 100 - data["write_0_read_1"][0].flatten()
    w1r0 = data["write_1_read_0"][0].flatten()
    z = w1r0 + w0r1
    ztotal = z.reshape((len(y), len(x)), order="F")

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Find the maximum critical current using np.diff
    diff = np.diff(ztotal, axis=0)
    mid_idx = np.where(diff == np.max(diff, axis=0))

    plt.scatter(x[mid_idx[1]], y[mid_idx[0]])

    # Plot a fit line to the scatter points
    z = np.polyfit(x[mid_idx[1][4:-4]], y[mid_idx[0][4:-4]], 1)
    p = np.poly1d(z)

    print(f"slope: {z[0]}, intercept: {z[1]}")
    plt.plot(x[mid_idx[1]], p(x[mid_idx[1]]), "r--")

    plt.imshow(
        diff,
        extent=[
            (-0.5 * dx + x[0]),
            (0.5 * dx + x[-1]),
            (-0.5 * dy + y[0]),
            (0.5 * dy + y[-1]),
        ],
        aspect="auto",
        origin="lower",
    )
    plt.xticks(np.linspace(x[0], x[-1], len(x)))
    plt.yticks(np.linspace(y[0], y[-1], len(y)))
    plt.xlabel("Enable Current [$\mu$A]")
    plt.ylabel("Channel Current [$\mu$A]")
    plt.title(f"Cell {data['cell']}")
    cbar = plt.colorbar()

    # Add fit slope and intercept to the plot
    print(p)
    plt.text(x[0], y[0], f"{p}", fontsize=12, color="red", backgroundcolor="white")
    return ztotal

def find_max_critical_current(data):
    x = data["x"][0][:, 1] * 1e6
    y = data["y"][0][:, 0] * 1e6
    w0r1 = 100 - data["write_0_read_1"][0].flatten()
    w1r0 = data["write_1_read_0"][0].flatten()
    z = w1r0 + w0r1
    ztotal = z.reshape((len(y), len(x)), order="F")
    ztotal = ztotal[:,1]
    
    # Find the maximum critical current using np.diff
    diff = np.diff(ztotal)
    mid_idx = np.where(diff == np.max(diff))

    return np.mean(y[mid_idx])


if __name__ == "__main__":
    files = os.listdir("data")
    for file in files:
        data = os.path.join("data", file)
        cell = file.split("_")[-2]
        data_dict = sio.loadmat(data)
        data_dict["cell"] = cell
        ztotal = find_peak(data_dict)
        plt.show()
        print(find_max_critical_current(data_dict))