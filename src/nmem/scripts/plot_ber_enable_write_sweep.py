import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.sweep_plots import (
    plot_enable_write_sweep2,
    plot_state_current_markers2,
)


def main(data_dir="../data/ber_sweep_enable_write_current/data1", save_dir=None):
    """
    Main function to plot enable write sweep and state current markers.
    """
    dict_list = import_directory(data_dir)
    
    # Plot enable write sweep
    fig1, ax1 = plot_enable_write_sweep2(dict_list)
    if save_dir:
        plt.savefig(
            f"{save_dir}/ber_enable_write_sweep.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
    
    # Plot state current markers
    fig2, ax2 = plot_state_current_markers2(dict_list)
    if save_dir:
        plt.savefig(
            f"{save_dir}/ber_enable_write_sweep_state_current_markers.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
