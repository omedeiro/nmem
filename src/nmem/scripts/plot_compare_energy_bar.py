"""
Energy Comparison Bar Chart

This script generates a 3D extruded bar chart comparing energy consumption
across different memory technologies. The plot visualizes energy per bit
operation for various superconducting and semiconducting memory types,
providing a clear comparison of their energy efficiency characteristics.
"""

import matplotlib.pyplot as plt

from nmem.analysis.bar_extrusion import plot_extruded_bar
from nmem.analysis.memory_data import (
    colors,
    energies_fj,
    energies_labels,
)
from nmem.analysis.styles import apply_global_style

# Apply the globally configured style
apply_global_style()


def main(save_dir=None):
    plot_extruded_bar(
        energies_labels,
        energies_fj,
        colors,
    )

    if save_dir:
        plt.savefig(f"{save_dir}/compare_energy_bar.png", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
