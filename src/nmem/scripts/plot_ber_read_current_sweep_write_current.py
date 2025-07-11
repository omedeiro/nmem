
import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.sweep_plots import (
    plot_read_sweep_write_current,
)


def main(save_dir=None):
    """
    Main function to generate BER read current sweep write current plots.
    """
    # Process first dataset
    data_list = import_directory("../data/ber_sweep_read_current/write_current/data1")
    plot_read_sweep_write_current(data_list)

    # Process second dataset
    data_list2 = import_directory("../data/ber_sweep_read_current/write_current/data2")
    plot_read_sweep_write_current(data_list2)

    if save_dir:
        plt.savefig(
            f"{save_dir}/ber_read_current_sweep_write_current.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
