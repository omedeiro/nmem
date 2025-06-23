import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.histogram_utils import (
    plot_voltage_hist,
)


def main(save_dir=None):
    dict_list = import_directory("../data/voltage_trace_averaged")
    fig, ax = plt.subplots()
    plot_voltage_hist(ax, dict_list[-2])

    if save_dir:
        plt.savefig(
            f"{save_dir}/voltage_read_histogram.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
