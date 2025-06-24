import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_read_current_sweep_enable_write_data
from nmem.analysis.sweep_plots import plot_read_current_sweep_enable_write


def main(save_dir=None):
    data_list, data_list2, colors = import_read_current_sweep_enable_write_data()
    plot_read_current_sweep_enable_write(data_list, data_list2, colors)

    if save_dir:
        plt.savefig(
            f"{save_dir}/ber_read_current_sweep_enable_write.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
