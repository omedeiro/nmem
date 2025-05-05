import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.path import Path
import qnngds.geometry as qg
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from nmem.analysis.analysis import set_plot_style
from matplotlib import ticker as mticker

set_plot_style()


def make_device():
    mem = qg.memory_v4()
    p = mem.polygons[0].polygons[0]
    pout = np.vstack((p[:149], p[-2:], p[:1]))  # outer boundary (closed)
    pin = np.vstack((p[150:-2], p[150:151]))  # inner cutout (closed)
    return pout, pin


def plot_device(ax, pout, pin):
    """
    Plot the device outline on the given axis.
    """
    ax.plot(pout[:, 0], pout[:, 1], color="black", linewidth=1.2)
    ax.plot(pin[:, 0], pin[:, 1], color="black", linewidth=1.2)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return ax


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


def interpolate_comsol_data(x, y, T, N=1000, method="nearest"):
    x, y, T = preprocess_griddata_inputs(x, y, T)
    xi = np.linspace(x.min(), x.max(), N)
    yi = np.linspace(y.min(), y.max(), N)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((x, y), T, (X, Y), method=method)
    return X, Y, Z


def plot_field(
    ax, X, Y, Z, pout=None, pin=None, cmap="plasma", title="", colorbar=False
):
    """
    Plot field with optional masking and tight inset colorbar.
    """
    # Optional mask
    if pout is not None and pin is not None:
        mask = compound_mask(X, Y, outer=pout, inner=pin)
        Z = np.ma.array(Z, mask=~mask)

    vmin = np.nanmin(Z)
    vmax = np.nanmax(Z)
    mesh = ax.pcolormesh(X, Y, Z, shading="auto", cmap=cmap, rasterized=True)
    if pout is not None:
        ax.plot(pout[:, 0], pout[:, 1], color="white", linewidth=1.2)
    if pin is not None:
        ax.plot(pin[:, 0], pin[:, 1], color="white", linewidth=1.2)

    if title == "Temperature":
        clabel = r"$T$ (K)"
    elif title == "Suppression":
        clabel = r"$\Delta(T)/\Delta(0)$"
    else:
        clabel = ""

    if colorbar:
        # Add inset colorbar matching height of axes
        cax = inset_axes(
            ax,
            width="3%",
            height="100%",
            loc="right",
            bbox_to_anchor=(0.05, 0.0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cbar = ax.figure.colorbar(
            mesh, cax=cax, label=clabel, ticks=np.linspace(vmin, vmax, 5)
        )
        cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_aspect("equal", adjustable="box")


if __name__ == "__main__":
    # Load and prepare unmasked file
    data1 = np.loadtxt("Last_nMem_temp_output.txt", skiprows=9)
    x1, y1, T1 = data1[:, 0], data1[:, 1], data1[:, 2]
    valid1 = ~np.isnan(T1)
    x1, y1, T1 = x1[valid1], y1[valid1], T1[valid1]

    # Load and prepare masked file
    data2 = np.loadtxt("Last_nMem_suppression_output.txt", skiprows=9)
    x2, y2, T2 = data2[:, 0], data2[:, 1], data2[:, 2]
    valid2 = ~np.isnan(T2)
    x2, y2, T2 = x2[valid2], y2[valid2], T2[valid2]

    # Get geometry outlines
    pout, pin = make_device()

    # Create side-by-side plot
    fig, ax = plt.subplots(figsize=(4, 4))

    # Plot raw temperature slice (no mask)
    X1, Y1, Z1 = interpolate_comsol_data(x1, y1, T1)
    plot_field(ax, X1, Y1, Z1, cmap="plasma", title="Temperature", colorbar=True)

    fig.patch.set_alpha(0.0)  # Set the figure background to transparent
    plt.savefig("comsol_temp_raw.png", dpi=300, bbox_inches="tight", transparent=True)
    plt.show()

    fig, ax = plt.subplots(figsize=(4, 4))
    # Plot masked suppression slice
    X2, Y2, Z2 = interpolate_comsol_data(x2, y2, T2)
    plot_field(
        ax,
        X2,
        Y2,
        Z2,
        pout,
        pin,
        cmap="viridis",
        title="Suppression",
        colorbar=True,
    )

    fig.subplots_adjust(wspace=0.4)

    fig.patch.set_alpha(0.0)  # Set the figure background to transparent
    fig.savefig(
        "comsol_temp_suppression.png",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    plt.show()
