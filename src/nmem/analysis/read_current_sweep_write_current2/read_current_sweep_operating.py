
from nmem.analysis.data_import import import_directory
from nmem.analysis.plotting import (
    plot_read_current_operating
)



def main():
    dict_list = import_directory("write_current_sweep_C3")
    plot_read_current_operating(dict_list)


if __name__ == "__main__":
    main()
