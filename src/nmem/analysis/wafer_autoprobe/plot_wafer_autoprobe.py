import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib.colors import LogNorm

def load_autoprobe_data(filepath):
    """Load autoprobe data from a parsed .mat file."""
    mat = loadmat(filepath, squeeze_me=True)

    die_name = mat['die_name']
    device_name = mat['device_name']
    data = mat['data']

    Rmean_raw = data['Rmean'].flatten()[0]
    Rmse_raw = data['Rmse'].flatten()[0]

    Rmean = np.array([r.item() if np.size(r) == 1 else np.nan for r in Rmean_raw])
    Rmse = np.array([r.item() if np.size(r) == 1 else np.nan for r in Rmse_raw])

    return die_name, device_name, Rmean, Rmse


def build_resistance_map(die_name, device_name, Rmean, grid_size=56):
    """Construct the 2D resistance map based on die/device names."""
    Rmeas = np.zeros((grid_size, grid_size))
    N = len(device_name)

    for i in range(N):
        DieName = die_name[i]
        DeviceName = device_name[i]

        xdie = ord(DieName[0]) - 64
        ydie = 8 - int(DieName[1])

        xdev = ord(DeviceName[0]) - 65
        ydev = 8 - int(DeviceName[1])

        xloc = (xdie - 1) * 8 + xdev
        yloc = (ydie - 1) * 8 + ydev

        Rval = Rmean[i]
        Rmeas[xloc, yloc] = 0 if Rval < 0 else Rval

    Rmeas[Rmeas == 0] = np.nanmax(Rmeas)
    return Rmeas


def plot_resistance_map(ax, Rmeas):
    """Plot the 2D resistance heatmap with die grid overlay."""
    vmin = np.nanmin(Rmeas[Rmeas > 0])
    vmax = np.nanmax(Rmeas)

    if vmin <= 0 or vmax <= vmin:
        raise ValueError(f"Invalid log scale bounds: vmin={vmin}, vmax={vmax}")

    im = ax.imshow(Rmeas.T, origin='lower', extent=[0, 56, 0, 56],
                   cmap='turbo', norm=LogNorm(vmin=vmin, vmax=vmax))
    cb = plt.colorbar(im, ax=ax, label='Ω')
    ax.set_xticks(np.linspace(3.5, 52.5, 7))
    ax.set_yticks(np.linspace(3.5, 52.5, 7))
    ax.set_xticklabels([str(i) for i in range(1, 8)])
    ax.set_yticklabels(list('GFEDCBA'))
    ax.set_xlim([-0.5, 56.5])
    ax.set_ylim([-0.5, 56.5])
    ax.set_aspect('equal')
    ax.set_title("Resistance Map (log scale)")

    diespace = np.linspace(-0.5, 56.5, 8)
    for x in diespace:
        ax.plot([x, x], [-0.5, 56.5], 'k-', linewidth=1.5)
    for y in diespace:
        ax.plot([-0.5, 56.5], [y, y], 'k-', linewidth=1.5)

    return im


def plot_rmean_rmse(ax, Rmeas, Rmse):
    """Plot Rmean (flattened) and Rmse on twin y-axes."""
    ax.plot(Rmeas.flatten(), '.', label='Rmean')
    ax.set_yscale('log')
    ax.set_ylabel("Resistance (Ω)")

    ax2 = ax.twinx()
    ax2.plot(Rmse, '.', color='orange', label='Rmse')
    ax2.set_ylim([0, 200])
    ax2.set_ylabel("RMSE")

    ax.set_title("Rmean and RMSE Scatter")
    return ax, ax2


if __name__ == "__main__":
    # Load data
    die_name, device_name, Rmean, Rmse = load_autoprobe_data("autoprobe_parsed.mat")

    # Build resistance map
    Rmeas = build_resistance_map(die_name, device_name, Rmean)

    # Plot heatmap
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    plot_resistance_map(ax1, Rmeas)

    # Plot scatter
    fig2, ax2 = plt.subplots()
    plot_rmean_rmse(ax2, Rmeas, Rmse)

    plt.tight_layout()
    plt.show()
