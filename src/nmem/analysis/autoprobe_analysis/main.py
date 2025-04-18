import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from nmem.analysis.autoprobe_analysis.data import load_autoprobe_data
from nmem.analysis.autoprobe_analysis.plot import plot_die_row
from nmem.analysis.analysis import set_plot_style
set_plot_style()  # Optional

# Custom colormap with NaNs as black (optional)
cmap = plt.colormaps["viridis"].copy()
# cmap.set_bad(color="black")

def plot_histogram(ax, vals, row_char, vmin=None, vmax=None):
    if len(vals) == 0:
        ax.text(0.5, 0.5, f"No data\nfor row {row_char}", ha="center", va="center", fontsize=8)
        ax.set_axis_off()
        return
    vals = vals[~np.isnan(vals)]
    log_bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), 100)
    ax.hist(vals, bins=log_bins, color="#888", edgecolor="black", alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlim(10, 5000)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.set_ylabel(f"{row_char}", rotation=0, ha="right", va="center", fontsize=9)
    ax.tick_params(axis='both', which='both', labelsize=6)
    if vmin and vmax:
        ax.axvline(vmin, color="blue", linestyle="--", linewidth=1)
        ax.axvline(vmax, color="red", linestyle="--", linewidth=1)

def combined_histogram_and_die_maps(df, wafer_row_numbers, limit_dict, N=7):
    fig, axs = plt.subplots(
        len(wafer_row_numbers), N + 2,
        figsize=(2 * (N + 2), 2.5 * len(wafer_row_numbers)),
        dpi=300,
        gridspec_kw={'width_ratios': [1] + [1]*N + [0.1]},
        constrained_layout=True
    )

    for i, row_number in enumerate(wafer_row_numbers):
        # Filter dies like A1, B1, ..., G1
        row_df = df[df["die"].str.endswith(str(row_number))].copy()
        valid_vals = row_df["Rmean"] / 1e3
        valid_vals = valid_vals[(valid_vals > 0) & np.isfinite(valid_vals) & (valid_vals < 50000)]

        n_nan = len(row_df) - len(valid_vals)
        if n_nan > 0:
            print(f"Row {row_number} has {n_nan} NaN values.")

        vmin, vmax = limit_dict.get(str(row_number), (valid_vals.min(), valid_vals.max()))
        plot_histogram(axs[i, 0], valid_vals, str(row_number), vmin, vmax)

        # Plot dies A1, B1, ..., G1
        im_list = []
        for j in range(N):
            die_name = f"{chr(65 + j)}{row_number}"
            die_df = df[df["die"] == die_name].copy()
            ax = axs[i, 1 + j]

            if die_df.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8)
                im_list.append(None)
                continue

            die_df["Rplot"] = die_df["Rmean"] / 1e3
            Rgrid = np.full((8, 8), np.nan)
            labels = np.full((8, 8), "", dtype=object)

            for _, row in die_df.iterrows():
                x, y = int(row["x_dev"]), int(row["y_dev"])
                if 0 <= x < 8 and 0 <= y < 8:
                    Rgrid[x, y] = row["Rplot"]
                    labels[x, y] = row["device"]

            im = ax.imshow(Rgrid.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            im_list.append(im)

            # Add device labels
            for x in range(8):
                for y in range(8):
                    label = labels[x, y]
                    if label:
                        ax.text(x, y, label, ha="center", va="center", fontsize=6, color="white")


            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(die_name, fontsize=7)
            ax.set_aspect("equal")

        # Colorbar
        axs[i, -1].set_xticks([])
        axs[i, -1].set_yticks([])
        axs[i, -1].set_frame_on(False)

        if any(im is not None for im in im_list):
            cax = axs[i, -1]
            first_valid_im = next(im for im in im_list if im is not None)
            cbar = fig.colorbar(first_valid_im, cax=cax)
            cbar.set_label("[kΩ]", fontsize=7)
            cbar.ax.tick_params(labelsize=6)
            cbar.set_ticks(np.linspace(vmin, vmax, 5))
            cbar.ax.set_yticklabels([f"{int(t)}" for t in np.linspace(vmin, vmax, 5)])
            if hasattr(cbar, "solids") and hasattr(cbar.solids, "set_rasterized"):
                cbar.solids.set_rasterized(True)
                cbar.solids.set_edgecolor("face")

    axs[-1, 0].set_xlabel("Resistance (kΩ)", fontsize=8)
    fig.patch.set_visible(False)
    fig.savefig("combined_wafer_map_and_histograms.pdf", bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    df = load_autoprobe_data("autoprobe_parsed.mat")
    wafer_rows = ["1", "4", "6",  "7"]  # Now using die *suffixes*
    limit_dict = {
        "1": [20, 100],
        "4": [20, 100],
        "6": [900, 1100],
        "7": [20, 100],
    }
    combined_histogram_and_die_maps(df, wafer_rows, limit_dict)
