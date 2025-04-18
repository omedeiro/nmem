from nmem.analysis.autoprobe_analysis.data import (
    load_autoprobe_data,
    build_resistance_map,
    normalize_row_by_squares,
)
from nmem.analysis.autoprobe_analysis.plot import (
    plot_resistance_map,
    plot_die_resistance_map,
    plot_die_row,
    scatter_die_row_resistance,
    scatter_die_resistance,
)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np


def main(data_path="autoprobe_parsed.mat"):
    df = load_autoprobe_data(data_path)

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_resistance_map(ax, df)
    plt.show()

    fig, ax = plt.subplots(figsize=(4, 4))
    plot_die_resistance_map(ax, df, "G4", annotate=True)
    plt.show()

    fig, axs = plt.subplots(1, 7, figsize=(12, 8))
    plot_die_row(axs, df, 6, annotate=True)
    plt.tight_layout()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax = scatter_die_row_resistance(ax, df, 7, logscale=False)
    plt.show()


def main2(data_path="autoprobe_parsed.mat"):
    df = load_autoprobe_data(data_path)
    N = 7
    fig, axs = plt.subplots(4, N, figsize=(12, 8), constrained_layout=True)

    wafer_rows = [0, 3, 5, 6]  # A, D, F, G for example

    for i, row in enumerate(wafer_rows):  # For each row of dies
        print(f"Row {row + 1}")

        # Get all resistance values from this row for percentile scaling
        row_char = chr(65 + row)
        row_df = df[df["die"].str.startswith(row_char)]
        valid_vals = row_df["Rmean"] / 1e3
        valid_vals = valid_vals[(valid_vals > 0) & (valid_vals < 10000)]
        vmin, vmax = np.percentile(valid_vals, [1, 99])
        print(f"{row_char}-row color scale: vmin={vmin:.2f}, vmax={vmax:.2f}")

        # Plot the row with fixed vmin/vmax
        _, im_list = plot_die_row(
            axs[i, :], df, row + 1, annotate=True, vmin=vmin, vmax=vmax
        )

        for j, (ax, im) in enumerate(zip(axs[i, :], im_list)):
            die_label = f"{chr(65 + row)}{j + 1}"
            ax.set_title(die_label, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        # Add one colorbar to the right of the row
        ax_right = axs[i, -1]
        cax = inset_axes(
            ax_right,
            width="3%",
            height="100%",
            loc="center left",
            bbox_to_anchor=(1.05, 0.0, 1, 1),
            bbox_transform=ax_right.transAxes,
            borderpad=0,
        )
        cbar = fig.colorbar(im_list[0], cax=cax, label="Resistance (kΩ)")
        # cbar.ax.set_yticks(np.linspace(vmin, vmax, 5))
    plt.show()


if __name__ == "__main__":
    df = load_autoprobe_data("autoprobe_parsed.mat")

    main2()
