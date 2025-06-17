

from nmem.analysis.data_import import import_read_current_sweep_enable_write_data
from nmem.analysis.sweep_plots import plot_read_current_sweep_enable_write


def main():
    data_list, data_list2, colors = import_read_current_sweep_enable_write_data()
    plot_read_current_sweep_enable_write(data_list, data_list2, colors)


if __name__ == "__main__":
    main()
