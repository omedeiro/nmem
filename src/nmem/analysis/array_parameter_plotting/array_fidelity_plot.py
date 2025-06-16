import numpy as np

from nmem.analysis.bar_plots import (
    plot_ber_3d_bar,
    plot_fidelity_clean_bar,
)
from nmem.analysis.core_analysis import (
    initialize_dict,
    process_cell,
)
from nmem.analysis.styles import (
    set_inter_font,
    set_pres_style,
)
from nmem.analysis.utils import convert_cell_to_coordinates
from nmem.measurement.cells import CELLS

set_pres_style()
set_inter_font()


def main():
    param_dict = initialize_dict((4, 4))
    for c in CELLS:
        xloc, yloc = convert_cell_to_coordinates(c)
        param_dict = process_cell(CELLS[c], param_dict, xloc, yloc)

    ber_array = param_dict["bit_error_rate"]
    valid_ber = ber_array[np.isfinite(ber_array) & (ber_array < 5.5e-2)]

    average_ber = np.mean(valid_ber)
    std_ber = np.std(valid_ber)
    min_ber = np.min(valid_ber)
    max_ber = np.max(valid_ber)
    print("=== Array BER Statistics ===")
    print(f"Average BER: {average_ber:.2e}")
    print(f"Std Dev BER: {std_ber:.2e}")
    print(f"Min BER: {min_ber:.2e}")
    print(f"Max BER: {max_ber:.2e}")
    print("=============================")

    plot_ber_3d_bar(ber_array)
    plot_fidelity_clean_bar(ber_array)

if __name__ == "__main__":
    main()