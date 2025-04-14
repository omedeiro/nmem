import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.path import Path
import qnngds.geometry as qg


def make_device():
    mem = qg.memory_v4()
    p = mem.polygons[0].polygons[0]
    pout = np.vstack((p[:149], p[-2:], p[:1]))   # outer boundary (closed)
    pin = np.vstack((p[150:-2], p[150:151]))     # inner cutout (closed)
    return pout, pin


def preprocess_griddata_inputs(x, y, T):
    coords = np.vstack((x, y)).T
    unique_coords, unique_indices = np.unique(coords, axis=0, return_index=True)
    return x[unique_indices], y[unique_indices], T[unique_indices]


def compound_mask(X, Y, outer, inner):
    points = np.vstack((X.ravel(), Y.ravel())).T
    path_outer = Path(outer, closed=True)
    path_inner = Path(inner, closed=True)
    in_outer = path_outer.contains_points(points)
    in_inner = path_inner.contains_points(points)
    return (in_outer & ~in_inner).reshape(X.shape)


def interpolate_comsol_data(x, y, T, N=1000, method='cubic'):
    x, y, T = preprocess_griddata_inputs(x, y, T)
    xi = np.linspace(x.min(), x.max(), N)
    yi = np.linspace(y.min(), y.max(), N)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((x, y), T, (X, Y), method=method)
    return X, Y, Z


def plot_field(ax, X, Y, Z, pout=None, pin=None, cmap='plasma', title='', colorbar=False):
    # Optional mask
    if pout is not None and pin is not None:
        mask = compound_mask(X, Y, outer=pout, inner=pin)
        Z = np.ma.array(Z, mask=~mask)

    mesh = ax.pcolormesh(X, Y, Z, shading='auto', cmap=cmap)
    if pout is not None:
        ax.plot(pout[:, 0], pout[:, 1], color='white', linewidth=1.2)
    if pin is not None:
        ax.plot(pin[:, 0], pin[:, 1], color='white', linewidth=1.2)
    if colorbar:
        ax.figure.colorbar(mesh, ax=ax, label='Temperature (K)')

    ax.set_xlabel('x (µm)')
    ax.set_ylabel('y (µm)')
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')


if __name__ == "__main__":
    # Load and prepare unmasked file
    data1 = np.loadtxt("Last_nMem_temp_output.txt", skiprows=9)
    x1, y1, T1 = data1[:, 2], data1[:, 3], data1[:, 4]
    valid1 = ~np.isnan(T1)
    x1, y1, T1 = x1[valid1], y1[valid1], T1[valid1]

    # Load and prepare masked file
    data2 = np.loadtxt("Last_nMem_suppression_output.txt", skiprows=9)
    x2, y2, T2 = data2[:, 2], data2[:, 3], data2[:, 4]
    valid2 = ~np.isnan(T2)
    x2, y2, T2 = x2[valid2], y2[valid2], T2[valid2]

    # Get geometry outlines
    pout, pin = make_device()

    # Create side-by-side plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    # Plot raw temperature slice (no mask)
    X1, Y1, Z1 = interpolate_comsol_data(x1, y1, T1)
    plot_field(axs[0], X1, Y1, Z1, cmap='plasma', title='Unmasked Temperature')

    # Plot masked suppression slice
    X2, Y2, Z2 = interpolate_comsol_data(x2, y2, T2)
    plot_field(axs[1], X2, Y2, Z2, pout, pin, cmap='plasma', title='Masked Suppression', colorbar=True)

    fig.tight_layout()
    plt.show()
