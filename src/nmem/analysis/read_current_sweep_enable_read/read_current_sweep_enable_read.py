
from nmem.analysis.data_import import import_read_current_sweep_data
from nmem.analysis.plotting import (
    plot_read_current_sweep_enable_read,
)


def main():
    dict_list, data_list, data_list2 = import_read_current_sweep_data()
    plot_read_current_sweep_enable_read(dict_list, data_list, data_list2)


if __name__ == "__main__":
    main()
