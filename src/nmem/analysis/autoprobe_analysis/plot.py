import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from nmem.analysis.autoprobe_analysis.utils import create_rmeas_matrix, get_log_norm_limits, annotate_matrix


def plot_die_resistance_map(ax, df, die_name, cmap="turbo", logscale=True, annotate=False):
    die_df = df[df["die"] == die_name]
    if die_df.empty:
        raise ValueError(f"No data found for die '{die_name}'")

    Rmeas = np.full((8, 8), np.nan)
    for _, row in die_df.iterrows():
        x, y = int(row["x_dev"]), int(row["y_dev"])
        y_plot = 7 - y  # flip vertical
        Rmeas[y_plot, x] = row["Rmean"] if row["Rmean"] > 0 else np.nan

    vmin, vmax = get_log_norm_limits(Rmeas)
    if vmin is None:
        raise ValueError(f"Die {die_name} contains no valid (R > 0) data.")

    im = ax.imshow(Rmeas, cmap=cmap, origin="upper",
                   norm=LogNorm(vmin=vmin, vmax=vmax) if logscale else None)

    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(8))
    ax.set_xticklabels(list("ABCDEFGH"))
    ax.set_yticklabels(np.arange(1, 9))
    ax.set_xlabel("Device Column")
    ax.set_ylabel("Device Row")
    ax.set_title(f"Resistance Map for Die {die_name}")
    ax.set_aspect("equal")

    if annotate:
        annotate_matrix(ax, Rmeas)

    plt.colorbar(im, ax=ax, label="Resistance (Ω)")
    return ax


def plot_resistance_map(ax, df, grid_size=56, cmap="turbo", logscale=True, annotate=False):
    Rmeas = create_rmeas_matrix(df, "x_abs", "y_abs", "Rmean", (grid_size, grid_size))
    if np.any(Rmeas == 0):
        Rmeas[Rmeas == 0] = np.nanmax(Rmeas)

    vmin, vmax = get_log_norm_limits(Rmeas)
    im = ax.imshow(
        Rmeas,
        origin="lower",
        extent=[0, grid_size, 0, grid_size],
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax) if logscale else None,
    )

    ax.set_xticks(np.linspace(3.5, 52.5, 7))
    ax.set_yticks(np.linspace(3.5, 52.5, 7))
    ax.set_xticklabels(list("ABCDEFG"))
    ax.set_yticklabels([str(i) for i in range(7, 0, -1)])
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect("equal")
    ax.set_title("Autoprobe Resistance Map")

    for line in np.linspace(0, grid_size, 8):
        ax.axhline(line, color='k', lw=1.5)
        ax.axvline(line, color='k', lw=1.5)

    if annotate:
        annotate_matrix(ax, Rmeas.T)

    plt.colorbar(im, ax=ax, label="Resistance (Ω)")
    return ax
