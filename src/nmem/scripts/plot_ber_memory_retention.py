import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_delay_data
from nmem.analysis.matrix_plots import plot_ber_array
from nmem.analysis.sweep_plots import plot_retention


def main(save_dir=None):
    delay_list, bit_error_rate_list, _ = import_delay_data()
    fig, axs = plot_retention(delay_list, bit_error_rate_list)
    plot_ber_array(axs["B"])

    if save_dir:
        plt.savefig(
            f"{save_dir}/ber_memory_retention.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
