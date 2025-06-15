
from nmem.analysis.data_import import import_directory
from nmem.analysis.plotting import (
    plot_voltage_pulse_avg,
)


def main():
    dict_list = import_directory("data")
    plot_voltage_pulse_avg(dict_list)


if __name__ == "__main__":
    main()
