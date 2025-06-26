#!/usr/bin/env python3
"""
Generate array parameter matrix visualization for various memory cell parameters.

Creates spatial heatmaps displaying different parameter distributions across memory cells
in the array, including write current, read current, enable currents, and other
characteristics. Enables identification of process variations and optimization
of array-level performance characteristics.

Usage:
    python plot_array_parameter_matrix.py --parameter write_current
    python plot_array_parameter_matrix.py --parameter read_current --save-dir ./output
    python plot_array_parameter_matrix.py --parameter enable_write_current
"""

import matplotlib.pyplot as plt

from nmem.analysis.core_analysis import (
    process_array_parameter_data,  # newly refactored function
)
from nmem.analysis.matrix_plots import plot_parameter_array
from nmem.analysis.styles import apply_global_style
from nmem.measurement.cells import CELLS

# Apply global plot styling
apply_global_style()


def main(save_dir=None, parameter="write_current", show_colorbar=True, log_scale=False):
    """
    Process cell data and plot parameter arrays for the given array size.

    Args:
        save_dir (str, optional): Directory to save the plot. Defaults to None.
        parameter (str): Parameter to plot. Options include:
            - "write_current": Write Current [μA]
            - "read_current": Read Current [μA]
            - "enable_write_current": Enable Write Current [μA]
            - "enable_read_current": Enable Read Current [μA]
            - "slope": Slope
            - "y_intercept": Y-Intercept
            - "x_intercept": X-Intercept
            - "resistance": Resistance [Ω]
            - "bit_error_rate": Bit Error Rate
            - "max_critical_current": Max Critical Current [μA]
            - "enable_write_power": Enable Write Power [W]
            - "enable_read_power": Enable Read Power [W]
        show_colorbar (bool): Whether to show a colorbar legend. Defaults to True.
        log_scale (bool): Whether to use logarithmic scale for the colormap. Defaults to False.
    """
    # Parameter labels and units mapping
    parameter_labels = {
        "write_current": "Write Current [$\\mu$A]",
        "read_current": "Read Current [$\\mu$A]",
        "enable_write_current": "Enable Write Current [$\\mu$A]",
        "enable_read_current": "Enable Read Current [$\\mu$A]",
        "slope": "Slope",
        "y_intercept": "Y-Intercept [$\\mu$A]",
        "x_intercept": "X-Intercept [$\\mu$A]",
        "resistance": "Resistance [$\\Omega$]",
        "bit_error_rate": "Bit Error Rate",
        "max_critical_current": "Max Critical Current [$\\mu$A]",
        "enable_write_power": "Enable Write Power [W]",
        "enable_read_power": "Enable Read Power [W]",
    }

    if parameter not in parameter_labels:
        raise ValueError(
            f"Parameter '{parameter}' not supported. Available parameters: {list(parameter_labels.keys())}"
        )

    xloc_list, yloc_list, param_dict, yintercept_list, slope_list = (
        process_array_parameter_data(CELLS)
    )
    fig, ax = plt.subplots()

    # Get the parameter data and label
    param_data = param_dict[parameter]
    param_label = parameter_labels[parameter]

    # Create filename based on parameter
    filename = f"array_{parameter}_matrix.png"

    # Plot the parameter array
    plot_parameter_array(
        xloc_list,
        yloc_list,
        param_data,
        param_label,
        log=log_scale,
        ax=ax,
    )

    # Add colorbar if requested
    if show_colorbar:
        # Get the mappable object from the most recent matshow call
        mappable = ax.get_images()[0]  # matshow creates an AxesImage
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.8, pad=0.05)
        cbar.set_label(param_label, rotation=270, labelpad=20)

        # Set colorbar ticks to show min and max values from the data
        vmin, vmax = param_data.min(), param_data.max()
        cbar.set_ticks([vmin, vmax])
        if log_scale:
            # Use scientific notation for log scale
            cbar.set_ticklabels([f"{vmin:.2e}", f"{vmax:.2e}"])
        else:
            cbar.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])
    if save_dir:
        plt.savefig(
            f"{save_dir}/{filename}",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate array parameter matrix visualization"
    )
    parser.add_argument(
        "--parameter",
        default="write_current",
        choices=[
            "write_current",
            "read_current",
            "enable_write_current",
            "enable_read_current",
            "slope",
            "y_intercept",
            "x_intercept",
            "resistance",
            "bit_error_rate",
            "max_critical_current",
            "enable_write_power",
            "enable_read_power",
        ],
        help="Parameter to plot (default: write_current)",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Directory to save plots (default: None - displays plot instead of saving)",
    )
    parser.add_argument(
        "--no-colorbar",
        action="store_true",
        help="Disable colorbar legend (default: colorbar is shown)",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use logarithmic scale for colormap (useful for parameters with wide ranges like bit_error_rate)",
    )

    args = parser.parse_args()

    # Generate plot for specified parameter
    main(
        save_dir=args.save_dir,
        parameter=args.parameter,
        show_colorbar=not args.no_colorbar,
        log_scale=args.log_scale,
    )

    # Example: Generate plots for multiple common parameters
    # Uncomment the lines below to generate multiple plots at once
    # common_parameters = ["write_current", "read_current", "enable_write_current", "enable_read_current"]
    # for param in common_parameters:
    #     print(f"Generating plot for {param}...")
    #     main(save_dir=args.save_dir, parameter=param)
