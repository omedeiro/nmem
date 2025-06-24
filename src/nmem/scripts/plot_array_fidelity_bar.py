import logging

from nmem.analysis.bar_plots import (
    plot_ber_3d_bar,
    plot_fidelity_clean_bar,
)
from nmem.analysis.core_analysis import process_ber_data
from nmem.analysis.styles import (
    set_inter_font,
    set_pres_style,
)

# Set plot styles
set_pres_style()
set_inter_font()

# Set up logger for better traceability
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def generate_plots(ber_array, save_dir=None):
    """
    Generate the plots and save them to the specified directory.
    """
    if save_dir:
        # Save the plots with a path provided
        plot_ber_3d_bar(ber_array, save_path=f"{save_dir}/ber_3d_bar.png")
        plot_fidelity_clean_bar(
            ber_array, save_path=f"{save_dir}/fidelity_clean_bar.png"
        )
    else:
        # Display the plots if no save path is provided
        plot_ber_3d_bar(ber_array)
        plot_fidelity_clean_bar(ber_array)


def main(save_dir=None):
    """
    Main function to process data and generate plots.
    """
    ber_array = process_ber_data(logger=logger)
    generate_plots(ber_array, save_dir)


if __name__ == "__main__":
    # Call the main function
    main()
