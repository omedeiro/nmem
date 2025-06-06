import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.core_analysis import (
    convert_cell_to_coordinates,
    initialize_dict,
    process_cell,
)
from nmem.analysis.plotting import (
    plot_ber_3d_bar,
    plot_fidelity_clean_bar,
    set_inter_font,
    set_pres_style,
)
from nmem.measurement.cells import CELLS

C0 = "#1b9e77"
C1 = "#d95f02"
RBCOLORS = plt.get_cmap("coolwarm")(np.linspace(0, 1, 4))
CMAP2 = plt.get_cmap("viridis")
set_pres_style()
set_inter_font()


if __name__ == "__main__":
    param_dict = initialize_dict((4, 4))
    xloc_list = []
    yloc_list = []
    for c in CELLS:
        xloc, yloc = convert_cell_to_coordinates(c)
        param_dict = process_cell(CELLS[c], param_dict, xloc, yloc)

    ber_array = param_dict["bit_error_rate"]
    valid_ber = ber_array[np.isfinite(ber_array) & (ber_array < 5.5e-2)]

    average_ber = np.mean(valid_ber)
    std_ber = np.std(valid_ber)
    min_ber = np.min(valid_ber)
    max_ber = np.max(valid_ber)
    print(len(valid_ber))
    print("=== Array BER Statistics ===")
    print(f"Average BER: {average_ber:.2e}")
    print(f"Std Dev BER: {std_ber:.2e}")
    print(f"Min BER: {min_ber:.2e}")
    print(f"Max BER: {max_ber:.2e}")
    print("=============================")

    # Plot the 3D bar chart
    plot_ber_3d_bar(ber_array)

    plot_fidelity_clean_bar(ber_array)
