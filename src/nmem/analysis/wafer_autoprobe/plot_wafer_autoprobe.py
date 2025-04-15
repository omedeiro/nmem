import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib.colors import LogNorm


def load_autoprobe_data(filepath):
    """Load autoprobe data from a parsed .mat file."""
    mat = loadmat(filepath, squeeze_me=True)

    die_name = mat["die_name"]
    device_name = mat["device_name"]
    data = mat["data"]

    Rmean_raw = data["Rmean"].flatten()[0]
    Rmse_raw = data["Rmse"].flatten()[0]

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


def plot_import(ax) -> None:
    die_name, device_name, Rmean, Rmse = load_autoprobe_data("autoprobe_parsed.mat")

    # Build resistance map
    Rmeas = build_resistance_map(die_name, device_name, Rmean)

    # Plot heatmap
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    plot_resistance_map(ax1, Rmeas)

    # Plot scatter
    fig2, ax2 = plt.subplots()
    plot_rmean_rmse(ax2, Rmeas, Rmse)

    return None


def plot_resistance_map(ax, Rmeas):
    """Plot the 2D resistance heatmap with die grid overlay."""
    vmin = np.nanmin(Rmeas[Rmeas > 0])
    vmax = np.nanmax(Rmeas)

    if vmin <= 0 or vmax <= vmin:
        raise ValueError(f"Invalid log scale bounds: vmin={vmin}, vmax={vmax}")

    im = ax.imshow(
        Rmeas.T,
        origin="lower",
        extent=[0, 56, 0, 56],
        cmap="turbo",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    cb = plt.colorbar(im, ax=ax, label="Ω")
    ax.set_xticks(np.linspace(3.5, 52.5, 7))
    ax.set_yticks(np.linspace(3.5, 52.5, 7))
    ax.set_xticklabels([str(i) for i in range(1, 8)])
    ax.set_yticklabels(list("GFEDCBA"))
    ax.set_xlim([-0.5, 56.5])
    ax.set_ylim([-0.5, 56.5])
    ax.set_aspect("equal")
    ax.set_title("Resistance Map (log scale)")

    diespace = np.linspace(-0.5, 56.5, 8)
    for x in diespace:
        ax.plot([x, x], [-0.5, 56.5], "k-", linewidth=1.5)
    for y in diespace:
        ax.plot([-0.5, 56.5], [y, y], "k-", linewidth=1.5)

    return im


def plot_rmean_rmse(ax, Rmeas, Rmse):
    """Plot Rmean (flattened) and Rmse on twin y-axes."""
    ax.plot(Rmeas.flatten(), ".", label="Rmean")
    ax.set_yscale("log")
    ax.set_ylabel("Resistance (Ω)")

    ax2 = ax.twinx()
    ax2.plot(Rmse, ".", color="orange", label="Rmse")
    ax2.set_ylim([0, 200])
    ax2.set_ylabel("RMSE")

    ax.set_title("Rmean and RMSE Scatter")
    return ax, ax2


def plot_die_row(
    ax, Rmeas, row_letter, resistance_threshold=None, stats=None, bounds=None
):
    row_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
    if row_letter.upper() not in row_map:
        raise ValueError(
            f"Invalid row '{row_letter}'. Must be one of: {list(row_map.keys())}"
        )

    row_index = row_map[row_letter.upper()]
    y_start = row_index * 8
    y_end = y_start + 8

    die_row = Rmeas[:, y_start:y_end].copy()
    if resistance_threshold is not None:
        die_row[die_row > resistance_threshold] = np.nan
    valid_values = die_row[np.isfinite(die_row)]

    if valid_values.size == 0:
        print(f"[{row_letter}] ⚠️ No valid data after filtering.")
        ax.set_title(f"Row {row_letter.upper()} — No valid data")
        ax.axis("off")
        return None

    if stats:
        try:
            vmin_stat = stats["q1"] * 0.5
            vmax_stat = stats["q3"] * 2.0
        except KeyError:
            vmin_stat, vmax_stat = None, None
        vmin = (
            max(vmin_stat, np.nanmin(valid_values))
            if vmin_stat
            else np.nanmin(valid_values)
        )
        vmax = (
            min(vmax_stat, np.nanmax(valid_values))
            if vmax_stat
            else np.nanmax(valid_values)
        )
    else:
        vmin = np.nanmin(valid_values[valid_values > 0])
        vmax = np.nanmax(valid_values)

    if bounds:
        vmin, vmax = bounds.get(row_letter.upper(), (vmin, vmax))
    # ✅ Validate limits
    if vmin <= 0 or vmax <= vmin:
        print(f"[{row_letter}] ⚠️ Invalid colormap limits: vmin={vmin}, vmax={vmax}")
        ax.set_title(f"Row {row_letter.upper()} — Invalid limits")
        ax.axis("off")
        return None

    im = ax.imshow(
        die_row.T,
        origin="lower",
        aspect="equal",
        extent=[0, 56, 0, 8],
        cmap="turbo",
        interpolation="none",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(f"Resistance Map: Die Row {row_letter.upper()}")
    ax.set_xticks(np.linspace(3.5, 52.5, 7))
    ax.set_xticklabels([str(i) for i in range(1, 8)])
    ax.set_yticks(np.arange(0.5, 8.5, 1))
    ax.set_yticklabels([str(i) for i in range(8)])
    ax.set_xlim(0, 56)
    ax.set_ylim(0, 8)
    ax.set_xlabel("Die Column")
    ax.set_ylabel("Device Index in Row")
    cbar = plt.colorbar(im, ax=ax, label="Resistance (Ω)", orientation="horizontal", pad=0.2)
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')

    return im


def plot_row_line(
    ax,
    Rmeas,
    row_letter,
    resistance_threshold=None,
    filter_percentile=None,
    bounds=None,
):
    """
    Plot all resistances in a die row as a line plot with boxplot lines.
    Also returns key statistics: mean, Q1, median, Q3.

    Parameters:
    - ax: matplotlib Axes object
    - Rmeas: full resistance map (56x56 array)
    - row_letter: one of 'A' to 'G'
    - resistance_threshold: optional float, values above this are masked
    - filter_percentile: optional float, values above this percentile are masked

    Returns:
    - dict with {'mean': ..., 'q1': ..., 'median': ..., 'q3': ...}
    """
    row_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
    if row_letter.upper() not in row_map:
        raise ValueError(
            f"Invalid row '{row_letter}'. Must be one of: {list(row_map.keys())}"
        )

    row_index = row_map[row_letter.upper()]
    y_start = row_index * 8
    y_end = y_start + 8

    # Extract and flatten 1D row values
    row_values = Rmeas[:, y_start:y_end].flatten()

    # Filter out open circuits and outliers
    if resistance_threshold is not None:
        row_values[row_values > resistance_threshold] = np.nan
    if filter_percentile is not None:
        p = np.nanpercentile(row_values, filter_percentile)
        row_values[row_values > p] = np.nan

    # Compute stats
    valid = row_values[~np.isnan(row_values)]
    mean_val = np.nanmean(valid)
    q1 = np.nanpercentile(valid, 25)
    median = np.nanpercentile(valid, 50)
    q3 = np.nanpercentile(valid, 75)

    # Plot
    x = np.arange(len(row_values))
    ax.plot(x, row_values, ".", label=f"Die Row {row_letter.upper()}")
    ax.axhline(mean_val, color="gray", linestyle="--", linewidth=1, label="Mean")
    ax.axhline(q1, color="blue", linestyle=":", linewidth=1, label="Q1")
    ax.axhline(median, color="green", linestyle="-", linewidth=1.5, label="Median")
    ax.axhline(q3, color="blue", linestyle=":", linewidth=1, label="Q3")
    if bounds:
        ax.set_ylim(
            bounds.get(
                row_letter.upper(), (np.nanmin(row_values), np.nanmax(row_values))
            )
        )
    ax.set_title(f"Resistance Line Plot: Die Row {row_letter.upper()}")
    ax.set_xlabel("Device Index")
    ax.set_ylabel("Resistance (Ω)")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()

    return {"mean": mean_val, "q1": q1, "median": median, "q3": q3}


if __name__ == "__main__":
    # Load data
    die_name, device_name, Rmean, Rmse = load_autoprobe_data("autoprobe_parsed.mat")

    # Build resistance map
    Rmeas = build_resistance_map(die_name, device_name, Rmean)

    rows = [
        "B",
        "D",
    ]
    bounds = {
        "A": (5e4, 5e6),
        "B": (9.5e5, 10.2e5),
        "C": (1e5, 1e7),
        "D": (3e4, 4e4),
        "E": (1e3, 1e9),
        "F": (5e5, 5e6),
        "G": (8e4, 1e6),
    }

    fig, axs = plt.subplot_mosaic(
        """
        AB
        CD
        """,
        layout="constrained",
        width_ratios=[1, 1],
        height_ratios=[1, 1],
    )

    plot_die_row(axs["A"], Rmeas, "B", resistance_threshold=10e6, bounds=bounds)

    stats = plot_row_line(
        axs["C"],
        Rmeas,
        "B",
        resistance_threshold=10e6,
        filter_percentile=95,
        bounds=bounds,
    )

    plot_die_row(axs["B"], Rmeas, "D", resistance_threshold=10e6, bounds=bounds)
    stats = plot_row_line(
        axs["D"],
        Rmeas,
        "D",
        resistance_threshold=10e6,
        filter_percentile=95,
        bounds=bounds,
    )
    plt.show()
