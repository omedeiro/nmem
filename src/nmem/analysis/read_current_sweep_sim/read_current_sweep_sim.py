

from nmem.analysis.data_import import load_current_sweep_data
from nmem.analysis.plotting import plot_current_sweep_results

if __name__ == "__main__":
    files, ltsp_data_dict, dict_list, write_current_list = load_current_sweep_data()
    plot_current_sweep_results(files, ltsp_data_dict, dict_list, write_current_list)
