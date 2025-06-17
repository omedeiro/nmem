
from nmem.analysis.data_import import import_directory
from nmem.analysis.trace_plots import (
    plot_voltage_pulse_avg,
)


def main():
    dict_list = import_directory("../data/voltage_trace_averaged")
    plot_voltage_pulse_avg(dict_list)


if __name__ == "__main__":
    main()
