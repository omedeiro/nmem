import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.plotting import (
    plot_read_sweep_write_current,
)


if __name__ == "__main__":
    data_list = import_directory("data")
    plot_read_sweep_write_current(data_list)

    data_list2 = import_directory("data2")
    plot_read_sweep_write_current(data_list2)
