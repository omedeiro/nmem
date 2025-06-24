import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.trace_plots import (
    plot_voltage_pulse_avg,
)


def main(save_dir=None):
    dict_list = import_directory("../data/voltage_trace_averaged")
    plot_voltage_pulse_avg(dict_list)

    if save_dir:
        plt.savefig(
            f"{save_dir}/voltage_trace_averaged.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
