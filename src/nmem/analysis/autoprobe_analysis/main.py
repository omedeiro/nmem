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

    wafer_rows = [0, 3, 5, 6]  # A, D, F, G for example
    limit_dict = {
        "A": [20, 100],
        # "B": [0, 100],
        # "C": [0, 100],
        "D": [20, 100],
        # "E": [30, 160],
        "F": [900, 1100],
        "G": [20, 100],
    }
    plot_row_histograms(df, limit_dict, wafer_rows=wafer_rows)
    
    
    fig, axs = plt.subplots(4, N, figsize=(7, 4), constrained_layout=True)

    for i, row in enumerate(wafer_rows):  # For each row of dies
        print(f"Row {row + 1}")

        # Get all resistance values from this row for percentile scaling
        row_char = chr(65 + row)
        row_df = df[df["die"].str.startswith(row_char)]
        valid_vals = row_df["Rmean"] / 1e3
        valid_vals = valid_vals[(valid_vals > 0) & np.isfinite(valid_vals)]
        
        # Get the min and max values for the color scale
        if len(valid_vals) == 0:
            print(f"Row {row + 1} contains no valid (R > 0) data.")
            continue
        vmin, vmax = limit_dict[row_char]

        print(f"Row {row + 1} min: {valid_vals.min()}, max: {valid_vals.max()}")
        # Plot the row with fixed vmin/vmax
        _, im_list = plot_die_row(
            axs[i, :], df, row + 1, annotate=False, vmin=vmin, vmax=vmax
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
        cbar = fig.colorbar(im_list[0], cax=cax, label="[kΩ]")
        cax_lims = cbar.ax.get_ylim()
        cbar.set_ticks(np.linspace(cax_lims[0], cax_lims[1], 5))
        cbar.ax.set_yticklabels([f"{tick:.1f}" for tick in np.linspace(cax_lims[0], cax_lims[1], 5)])
    plt.savefig("wafer_row_resistance_maps.pdf", dpi=300)
    plt.show()


def plot_row_histograms(df, limit_dict, wafer_rows=[0, 3, 5, 6]):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.close('all')  # Reset state

    fig, axs = plt.subplots(len(wafer_rows), 1, figsize=(3, 4), sharex=True)

    for i, row in enumerate(wafer_rows):
        row_char = chr(65 + row)
        ax = axs[i]
        row_df = df[df["die"].str.startswith(row_char)]
        valid_vals = row_df["Rmean"] / 1e3
        valid_vals = valid_vals[(valid_vals > 0) & np.isfinite(valid_vals)]


        if len(valid_vals) == 0:
            ax.text(0.5, 0.5, f"No data for row {row_char}", ha="center", va="center")
            continue

        # After filtering
        valid_vals = valid_vals[(valid_vals > 0) & np.isfinite(valid_vals)]

        # Optional: clip to match map scale
        valid_vals = valid_vals[valid_vals < 5000]  # or whatever matches your map scale

        # Log bins across selected range
        log_bins = np.logspace(np.log10(valid_vals.min()), np.log10(valid_vals.max()), 200)

        # Plot
        counts, _, _ = ax.hist(valid_vals, bins=log_bins, color="gray", edgecolor="black", alpha=0.7)

        # Log-spaced bins
        log_bins = np.logspace(np.log10(valid_vals.min()), np.log10(valid_vals.max()), 200)
        ax.hist(valid_vals, bins=log_bins, color="gray", edgecolor="black", alpha=0.7)

        # vmin/vmax lines
        if row_char in limit_dict:
            vmin, vmax = limit_dict[row_char]
            ax.axvline(vmin, color="blue", linestyle="--", label="vmin")
            ax.axvline(vmax, color="red", linestyle="--", label="vmax")

        ax.set_ylim(0, 100)
        ax.set_xscale("log")
        ax.set_ylabel(f"Row {row_char}")
        ax.grid(True, linestyle=":", linewidth=0.5)

        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
    

    axs[-1].set_xlabel("Resistance (kΩ)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("wafer_row_histograms.pdf", dpi=300)
    plt.show()


if __name__ == "__main__":
    df = load_autoprobe_data("autoprobe_parsed.mat")

    main2()
