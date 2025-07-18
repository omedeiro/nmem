

import matplotlib.pyplot as plt

from nmem.analysis.data_import import load_current_sweep_data
from nmem.analysis.styles import apply_global_style
from nmem.analysis.sweep_plots import plot_current_sweep_results

# Apply global plot styling
apply_global_style()



def main(save_dir="../plots"):
    
    files, ltsp_data_dict, dict_list, write_current_list = load_current_sweep_data()
    plot_current_sweep_results(files, ltsp_data_dict, dict_list, write_current_list)

    if save_dir:
        plt.savefig(
            f"{save_dir}/ber_read_current_sweep_sim.pdf", dpi=300, bbox_inches="tight"
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
