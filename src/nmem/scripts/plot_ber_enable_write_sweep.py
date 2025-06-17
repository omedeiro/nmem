import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.sweep_plots import (
    plot_enable_write_sweep2,
    plot_state_current_markers2,
)


def main(
    data_dir="../data/ber_sweep_enable_write_current/data1", save_sweep_fig=False, sweep_output_path="enable_write_sweep.pdf"
):
    """
    Main function to plot enable write sweep and state current markers.
    """
    dict_list = import_directory(data_dir)
    plot_enable_write_sweep2(
        dict_list, save_fig=save_sweep_fig, output_path=sweep_output_path
    )
    plt.show()
    plot_state_current_markers2(dict_list)
    plt.show()


if __name__ == "__main__":
    main()
