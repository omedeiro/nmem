import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from nmem.analysis.autoprobe_analysis.data import load_autoprobe_data
from nmem.analysis.autoprobe_analysis.plot import plot_die_row
from nmem.analysis.analysis import set_plot_style
set_plot_style()  # Optional, comment out if unavailable

def plot_histogram(ax, vals, row_char, vmin=None, vmax=None):
    if len(vals) == 0:
        ax.text(0.5, 0.5, f"No data\nfor row {row_char}", ha="center", va="center", fontsize=8)
        ax.set_axis_off()
        return
    # Remove NaN values
    vals = vals[~np.isnan(vals)]
    #find values between 900 and 1100
    new_vals = vals[(vals > 900) & (vals < 1100)]
    if len(new_vals) > 0:
        print(f"Row {row_char} has {len(new_vals)} values between 900 and 1100.")
    log_bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), 100)
    ax.hist(vals, bins=log_bins, color="gray", edgecolor="black", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlim(10, 5000)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.set_ylabel(f"{row_char}", rotation=0, ha="right", va="center", fontsize=9)
    if vmin and vmax:
        ax.axvline(vmin, color="blue", linestyle="--")
        ax.axvline(vmax, color="red", linestyle="--")
    ax.tick_params(axis='both', which='both', labelsize=6)


def combined_histogram_and_die_maps(df, wafer_rows, limit_dict, N=7):
    fig, axs = plt.subplots(len(wafer_rows), N + 1, figsize=(7, 4), constrained_layout=True)
        
    for i, row in enumerate(wafer_rows):
        row_char = chr(65 + row)
        row_df = df[df["die"].str.startswith(row_char)]
        valid_vals = row_df["Rmean"] / 1e3
        valid_vals = valid_vals[(valid_vals > 0) & np.isfinite(valid_vals) & (valid_vals < 5000)]
        # find number of nan values 
        n_nan = len(row_df) - len(valid_vals)
        if n_nan > 0:
            print(f"Row {row_char} has {n_nan} NaN values.")
        vmin, vmax = limit_dict.get(row_char, (valid_vals.min(), valid_vals.max()))

        # Histogram in first column
        plot_histogram(axs[i, 0], valid_vals, row_char, vmin, vmax)

        # Die row plots
        _, im_list = plot_die_row(
            axs[i, 1:], df, row + 1, annotate=False, vmin=vmin, vmax=vmax
        )
        for j, (ax, im) in enumerate(zip(axs[i, 1:], im_list)):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{row_char}{j + 1}", fontsize=6)
            for spine in ax.spines.values():
                spine.set_visible(False)

        # Colorbar
        cax = inset_axes(axs[i, -1], width="3%", height="100%", loc="center left",
                         bbox_to_anchor=(1.05, 0.0, 1, 1), bbox_transform=axs[i, -1].transAxes, borderpad=0)
        cbar = fig.colorbar(im_list[0], cax=cax)
        cbar.set_label("[kΩ]", fontsize=7)
        cbar.ax.tick_params(labelsize=6)
        cbar.ax.set_yticks(np.linspace(vmin, vmax, 5))
        cbar.ax.set_yticklabels([f"{int(tick)}" for tick in np.linspace(vmin, vmax, 5)], fontsize=6)

    axs[-1, 0].set_xlabel("Resistance (kΩ)", fontsize=8)
    plt.savefig("combined_wafer_map_and_histograms.pdf", dpi=300)
    plt.show()


if __name__ == "__main__":
    df = load_autoprobe_data("autoprobe_parsed.mat")
    wafer_rows = [0, 3, 5, 6]  # A, D, F, G
    limit_dict = {
        "A": [20, 100],
        "D": [20, 100],
        "F": [900, 1100],
        "G": [20, 100],
    }
    combined_histogram_and_die_maps(df, wafer_rows, limit_dict)
