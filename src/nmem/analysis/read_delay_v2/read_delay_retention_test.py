import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_delay_data
from nmem.analysis.matrix_plots import plot_ber_array
from nmem.analysis.sweep_plots import plot_retention


def main():
    delay_list, bit_error_rate_list, _ = import_delay_data()
    fig, axs = plot_retention(delay_list, bit_error_rate_list)
    plot_ber_array(axs["B"])
    save_fig = False
    if save_fig:
        plt.savefig("read_delay_retention_test.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
