import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_delay_data
from nmem.analysis.styles import apply_global_style
from nmem.analysis.sweep_plots import plot_retention

# Apply global plot styling
apply_global_style()



def main(save_dir=None):
    delay_list, bit_error_rate_list, _ = import_delay_data()
    plot_retention(delay_list, bit_error_rate_list)

    if save_dir:
        plt.savefig(
            f"{save_dir}/ber_memory_retention.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
