from nmem.analysis.data_import import import_read_current_sweep_three_data
from nmem.analysis.plotting import plot_read_current_sweep_three


def main():
    dict_list = import_read_current_sweep_three_data()
    plot_read_current_sweep_three(dict_list)


if __name__ == "__main__":
    main()
