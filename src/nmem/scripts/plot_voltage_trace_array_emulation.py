import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.trace_plots import (
    plot_time_concatenated_traces,
)


def main():
    dict_list = import_directory("../data/voltage_trace_emulate_array")

    fig, axs = plt.subplots(3, 1, figsize=(6, 3), sharex=True)
    plot_time_concatenated_traces(axs, dict_list[:5])
    save_fig = False
    if save_fig:
        plt.savefig("voltage_trace_emulate_slow.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
