import matplotlib.pyplot as plt

from nmem.analysis.analysis_cmos_comparison.memory_data import (
    colors,
    energies_fj,
    energies_labels,
)
from nmem.analysis.bar_plots import plot_extruded_bar
from nmem.analysis.styles import set_inter_font, set_pres_style

set_inter_font()
set_pres_style()


plot_extruded_bar(
    energies_labels, energies_fj, colors
)
plt.show()
