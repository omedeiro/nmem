import matplotlib.pyplot as plt

from nmem.analysis.histogram_utils import (
    plot_voltage_hist,
)
from nmem.analysis.data_import import import_directory


def main():
    dict_list = import_directory("../data/voltage_trace_averaged")
    fig, ax = plt.subplots()
    plot_voltage_hist(ax, dict_list[-2])


if __name__ == "__main__":
    main()
