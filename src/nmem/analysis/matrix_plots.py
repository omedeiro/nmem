import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm

from nmem.analysis.core_analysis import (
    initialize_dict,
    process_cell,
)
from nmem.analysis.histogram_utils import plot_histogram
from nmem.analysis.plot_utils import get_log_norm_limits
from nmem.analysis.styles import CMAP
from nmem.analysis.utils import (
    convert_cell_to_coordinates,
    create_rmeas_matrix,
)
from nmem.measurement.cells import CELLS
from nmem.analysis.bit_error import get_bit_error_rate
from nmem.analysis.currents import get_write_currents, get_enable_current_sweep
from nmem.analysis.data_import import import_directory

def annotate_matrix(ax, R, fmt="{:.2g}", color="white"):
    """Add text annotations to matrix cells."""
    for y in range(R.shape[0]):
        for x in range(R.shape[1]):
            val = R[y, x]
            if not np.isnan(val):
                ax.text(
                    x,
                    y,
                    fmt.format(val),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=color,
                )


def plot_ber_grid(ax: plt.Axes):
    ARRAY_SIZE = (4, 4)
    param_dict = initialize_dict(ARRAY_SIZE)
    xloc_list = []
    yloc_list = []
    for c in CELLS:
        xloc, yloc = convert_cell_to_coordinates(c)
        param_dict = process_cell(CELLS[c], param_dict, xloc, yloc)
        xloc_list.append(xloc)
        yloc_list.append(yloc)

    plot_parameter_array(
        ax,
        xloc_list,
        yloc_list,
        param_dict["bit_error_rate"],
        log=True,
        cmap=plt.get_cmap("Blues").reversed(),
    )

    ax.xaxis.set_label_position("bottom")
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    cax = ax.inset_axes([1.10, 0, 0.1, 1])
    fig = ax.get_figure()
    cbar = fig.colorbar(
        ax.get_children()[0], cax=cax, orientation="vertical", label="minimum BER"
    )

    return ax


def plot_resistance_map(
    ax, df, grid_size=56, cmap="turbo", logscale=True, annotate=False
):
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
        ax.axhline(line, color="k", lw=1.5)
        ax.axvline(line, color="k", lw=1.5)

    if annotate:
        annotate_matrix(ax, Rmeas.T)

    plt.colorbar(im, ax=ax, label="Resistance (Ω)")
    return ax


def plot_die_row(
    axes, df, row_number, cmap="turbo", annotate=False, vmin=None, vmax=None
):
    """
    Plot all dies in a given wafer row.
    row_number: 1 (top) to 7 (bottom)
    columns: 'A' (left) to 'G' (right)
    """
    if not (1 <= row_number <= 7):
        raise ValueError("row_number must be between 1 and 7")

    die_names = [f"{col}{row_number}" for col in "ABCDEFG"]
    im_list = []
    for ax, die_name in zip(axes, die_names):
        try:
            ax, im = plot_die_resistance_map(
                ax, df, die_name, cmap=cmap, annotate=annotate, vmin=vmin, vmax=vmax
            )
            im_list.append(im)
        except Exception as e:
            ax.set_title(f"{die_name} (Error)")
            ax.axis("off")
            print(f"Skipping {die_name}: {e}")

    return axes, im_list


def plot_combined_histogram_and_die_maps(df, wafer_row_numbers, limit_dict, N=7):
    fig, axs = plt.subplots(
        len(wafer_row_numbers),
        N + 2,
        figsize=(5, 4),
        dpi=300,
        gridspec_kw={"width_ratios": [1] + [1] * N + [0.1]},
        constrained_layout=True,
    )

    for i, row_number in enumerate(wafer_row_numbers):
        # Filter dies like A1, B1, ..., G1
        row_df = df[df["die"].str.endswith(str(row_number))].copy()
        valid_vals = row_df["Rmean"] / 1e3
        valid_vals = valid_vals[
            (valid_vals > 0) & np.isfinite(valid_vals) & (valid_vals < 50000)
        ]

        n_nan = len(row_df) - len(valid_vals)
        vmin, vmax = limit_dict.get(
            str(row_number), (valid_vals.min(), valid_vals.max())
        )
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

            im = ax.imshow(Rgrid.T, origin="lower", cmap=CMAP, vmin=vmin, vmax=vmax)
            im_list.append(im)

            # # Add device labels
            # for x in range(8):
            #     for y in range(8):
            #         label = labels[x, y]
            #         if label:
            #             ax.text(x, y, label, ha="center", va="center", fontsize=6, color="white")

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

    axs[2, 0].set_xlim(500, 1500)

    return fig, axs


def plot_parameter_array(
    xloc: np.ndarray,
    yloc: np.ndarray,
    parameter_array: np.ndarray,
    title: str = None,
    log: bool = False,
    reverse: bool = False,
    cmap: plt.cm = None,
    ax: Axes = None,
) -> Axes:
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if cmap is None:
        cmap = plt.get_cmap("viridis")
    if reverse:
        cmap = cmap.reversed()

    if log:
        ax.matshow(
            parameter_array,
            cmap=cmap,
            norm=LogNorm(vmin=np.min(parameter_array), vmax=np.max(parameter_array)),
        )
    else:
        ax.matshow(parameter_array, cmap=cmap)

    if title:
        ax.set_title(title)
    ax.set_xticks(range(4), ["A", "B", "C", "D"])
    ax.set_yticks(range(4), ["1", "2", "3", "4"])
    ax.tick_params(axis="both", length=0)

    return ax


def plot_cell_parameter(ax: Axes, param: str) -> Axes:
    param_array = np.array([CELLS[cell][param] for cell in CELLS]).reshape(4, 4)
    plot_parameter_array(
        ax,
        np.arange(4),
        np.arange(4),
        param_array * 1e6,
        f"Cell {param}",
        log=False,
        norm=False,
        reverse=False,
    )
    return ax


def plot_die_resistance_map(
    ax, df, die_name, cmap="turbo", logscale=True, annotate=False, vmin=None, vmax=None
):
    die_df = df[df["die"] == die_name]
    if die_df.empty:
        raise ValueError(f"No data found for die '{die_name}'")

    Rmeas = np.full((8, 8), np.nan)
    for _, row in die_df.iterrows():
        x, y = int(row["x_dev"]), int(row["y_dev"])
        y_dev = 7 - y  # Invert y-axis for display
        Rmeas[y_dev, x] = row["Rmean"] if row["Rmean"] > 0 else np.nan

    # Robust color limits using percentiles
    valid_vals = Rmeas[np.isfinite(Rmeas) & (Rmeas > 0)] / 1e3

    if valid_vals.size == 0:
        raise ValueError(f"Die {die_name} contains no valid (R > 0) data.")

    im = ax.imshow(
        Rmeas / 1e3,
        cmap=cmap,
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )

    if annotate:
        for y in range(8):
            for x in range(8):
                val = Rmeas[y, x]
                if np.isfinite(val):
                    ax.text(
                        x,
                        y,
                        f"{val:.0f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white",
                    )

    ax.set_xticks([])
    ax.set_yticks([])
    return ax, im

    # ax.set_xticks(np.arange(8))
    # ax.set_yticks(np.arange(8))
    # ax.set_xticklabels(list("ABCDEFGH"))
    # ax.set_yticklabels(np.arange(1, 9))
    # set_axis_labels(
    #     ax, "Device Column", "Device Row", f"Resistance Map for Die {die_name}"
    # )
    # ax.set_aspect("equal")

    # if annotate:
    #     annotate_matrix(ax, Rmeas)

    # return ax, im


def plot_ber_array(ax):
    """Plot BER array map."""
    ARRAY_SIZE = (4, 4)
    param_dict = initialize_dict(ARRAY_SIZE)
    xloc_list = []
    yloc_list = []
    for c in CELLS:
        xloc, yloc = convert_cell_to_coordinates(c)
        param_dict = process_cell(CELLS[c], param_dict, xloc, yloc)
        xloc_list.append(xloc)
        yloc_list.append(yloc)
    plot_parameter_array(
        ax,
        xloc_list,
        yloc_list,
        param_dict["bit_error_rate"],
        log=True,
        cmap=plt.get_cmap("Blues").reversed(),
    )
    cax = ax.inset_axes([1.10, 0, 0.1, 1])
    cbar = ax.figure.colorbar(
        ax.get_children()[0], cax=cax, orientation="vertical", label="minimum BER"
    )
    # cbar.set_ticks([1e-5, 1e-4, 1e-3, 1e-2])


def plot_wafer_maps(maps, titles, cmaps, grid_x, grid_y, radius, annotate_points=False):
    fig, axes = plt.subplots(1, 3, figsize=(7, 3.5), dpi=300)  # 7.2" ≈ 2-column width
    for ax, title, (grid_z, pts, vals), cmap in zip(axes, titles, maps, cmaps):
        circle = plt.Circle((0, 0), radius, color="k", lw=0.5, fill=False)
        contour = ax.contourf(grid_x, grid_y, grid_z, levels=30, cmap=cmap)
        # ax.scatter(pts[:, 0], pts[:, 1], c='k', s=8, zorder=10)
        if annotate_points:
            for (x, y), v in zip(pts, vals):
                ax.text(
                    x,
                    y,
                    f"{v:.1f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white",
                    zorder=11,
                )
        ax.add_artist(circle)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        cbar = fig.colorbar(
            contour, ax=ax, orientation="vertical", fraction=0.046, pad=0.04
        )
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Thickness (nm)", fontsize=9)
    plt.tight_layout()
    return fig


def plot_state_current_matrix(dict_list, ax=None, vmin=None, vmax=None):
    """
    Plot BER as a matrix with write current vs enable current.
    
    Args:
        dict_list: List of measurement dictionaries
        ax: Matplotlib axis to plot on
        vmin, vmax: Color scale limits
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    
    # Collect data from all measurements
    write_currents = []
    enable_currents = []
    ber_data = []
    
    for dict_item in dict_list:
        write_current = get_write_currents(dict_item)[0]
        enable_current = get_enable_current_sweep(dict_item)
        bit_error_rate = get_bit_error_rate(dict_item)
        
        write_currents.append(write_current)
        enable_currents.append(enable_current)
        ber_data.append(bit_error_rate)
    
    # Get unique write currents and enable current array (should be same for all)
    unique_write_currents = sorted(set(write_currents))
    enable_current_array = enable_currents[0]  # All should be the same
    
    # Create matrix: rows = write currents, cols = enable currents
    ber_matrix = np.full((len(unique_write_currents), len(enable_current_array)), np.nan)
    
    # Fill the matrix
    for i, dict_item in enumerate(dict_list):
        write_current = write_currents[i]
        ber_values = ber_data[i]
        
        # Find row index for this write current
        row_idx = unique_write_currents.index(write_current)
        ber_matrix[row_idx, :] = ber_values
    
    # Set color limits if not provided
    if vmin is None:
        vmin = np.nanmin(ber_matrix)
    if vmax is None:
        vmax = np.nanmax(ber_matrix)
    
    # Create the plot
    im = ax.imshow(
        ber_matrix,
        cmap=plt.get_cmap("Reds").reversed(),
        norm=LogNorm(vmin=vmin, vmax=vmax) if vmin > 0 else None,
        origin="lower",
        aspect="auto",
        extent=[
            enable_current_array.min(),
            enable_current_array.max(),
            min(unique_write_currents),
            max(unique_write_currents),
        ],
    )
    
    # Set labels and title
    ax.set_xlabel("Enable Current [µA]")
    ax.set_ylabel("Write Current [µA]")
    ax.set_title("Bit Error Rate Matrix")
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Bit Error Rate")
    
    return ax


def plot_ber_heatmap_detailed(dict_list, ax=None, annotate=False):
    """
    Plot detailed BER heatmap with discrete tick marks.
    
    Args:
        dict_list: List of measurement dictionaries
        ax: Matplotlib axis to plot on
        annotate: Whether to add text annotations
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure
    
    # Collect and organize data
    write_currents = []
    enable_currents = []
    ber_data = []
    
    for dict_item in dict_list:
        write_current = get_write_currents(dict_item)[0]
        enable_current = get_enable_current_sweep(dict_item)
        bit_error_rate = get_bit_error_rate(dict_item)
        
        write_currents.append(write_current)
        enable_currents.append(enable_current)
        ber_data.append(bit_error_rate)
    
    # Get unique values
    unique_write_currents = sorted(set(write_currents))
    enable_current_array = enable_currents[0]
    
    # Create matrix
    ber_matrix = np.full((len(unique_write_currents), len(enable_current_array)), np.nan)
    
    for i, dict_item in enumerate(dict_list):
        write_current = write_currents[i]
        ber_values = ber_data[i]
        row_idx = unique_write_currents.index(write_current)
        ber_matrix[row_idx, :] = ber_values
    
    # Plot with proper indexing for discrete ticks
    im = ax.imshow(
        ber_matrix,
        cmap=plt.get_cmap("Reds").reversed(),
        origin="lower",
        aspect="auto",
    )
    
    # Set discrete ticks
    ax.set_xticks(range(0, len(enable_current_array), 5))  # Every 5th enable current
    ax.set_xticklabels([f"{enable_current_array[i]:.0f}" for i in range(0, len(enable_current_array), 5)])
    
    ax.set_yticks(range(len(unique_write_currents)))
    ax.set_yticklabels([f"{wc:.0f}" for wc in unique_write_currents])
    
    ax.set_xlabel("Enable Current [µA]")
    ax.set_ylabel("Write Current [µA]")
    ax.set_title("Bit Error Rate Heatmap")
    
    # Add annotations if requested
    if annotate:
        for i in range(len(unique_write_currents)):
            for j in range(len(enable_current_array)):
                if not np.isnan(ber_matrix[i, j]):
                    text = f"{ber_matrix[i, j]:.3f}"
                    ax.text(j, i, text, ha="center", va="center", 
                           fontsize=6, color="white" if ber_matrix[i, j] > 0.5 else "black")
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Bit Error Rate")
    
    return ax


if __name__ == "__main__":
    # Example usage
    dict_list = import_directory("../data/ber_sweep_enable_write_current/data1")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Continuous color mapping
    plot_state_current_matrix(dict_list, ax=ax1)
    ax1.set_title("BER Matrix (Continuous)")
    
    # Plot 2: Detailed heatmap with discrete ticks
    plot_ber_heatmap_detailed(dict_list, ax=ax2, annotate=False)
    ax2.set_title("BER Heatmap (Detailed)")
    
    plt.tight_layout()
    plt.show()
