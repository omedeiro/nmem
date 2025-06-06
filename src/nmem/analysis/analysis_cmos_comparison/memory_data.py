import numpy as np

from nmem.analysis.spider_plots import (
    normalize_log,
    normalize_log_inverse,
)

energies_labels = ["Write (W)", "Read (R)", "Enable-Write (EW)", "Enable-Read (ER)"]
energies_fj = [46, 31, 1256, 202]
colors = ["gray", "gray", "darkred", "darkred"]


tech_cap = [
    "CMOS NAND Flash",
    "CMOS DRAM",
    "CMOS SRAM",
    "Josephson RAM",
]  # , "This Work (SNM)"]
cap_vals = np.log10([1e12, 24e9, 64e6, 4e3])  # , 64])
cap_labels = ["~1 Tb", "~24 Gb", "~64 Mb", "4 kb"]  # , "64 b"]
cap_colors = ["gray", "gray", "gray", "darkred"]  # , "royalblue"]

tech_den = [
    "CMOS NAND Flash",
    "CMOS DRAM",
    "CMOS SRAM",
    "Josephson RAM",
]  # , "This Work (SNM)"]
den_vals = np.log10([96e9, 50e9, 8e9, 1e6])  # , 2.6e6])
den_labels = ["~96 Gb/cm²", "~50 Gb/cm²", "~8 Gb/cm²", "1 Mb/cm²"]  # , "2.6 Mb/cm²"]
den_colors = ["gray", "gray", "gray", "darkred"]  # , "royalblue"]


# --- Data ---


values_snm_projected = [10.0, 1e4, 1e-6, 1e-15, 1e-9]
values_snm_new = [2.6, 64, 1e-5, 1.3e-12, 20e-9]
values_snm_old = [2, 1, 1e-3, 10e-15, 10e-9]
jsram = [4, 1, 1, 50e-16, 1e-12]
vt2ram = [
    0.9,
    72,
    np.nan, 
    
]

datasets = [values_snm_new, values_snm_old, values_snm_projected]
labels = ["This work", "Previous work", "Advanced Process Node", "JSRAM"]
styles = [
    {"color": "#BD342D", "linewidth": 2.0, "linestyle": "-"},
    {"color": "#586563", "linewidth": 1.2, "linestyle": "--"},
    {"color": "#658DDC", "linewidth": 1.2, "linestyle": "-."},
    {"color": "black", "linewidth": 1.2, "linestyle": ":"},
]
metrics = ["Density", "Capacity", "BER", "Access energy", "Access time"]
units = ["Mb/cm²", "bits", "", "J/bit", "s"]
axis_min = [0.1, 0.1, 1e-6, 1e-16, 1e-12]
axis_max = [10.0, 1e6, 1e-2, 1e-9, 1e-3]
normalizers = [
    normalize_log,
    normalize_log,
    normalize_log_inverse,
    normalize_log_inverse,
    normalize_log_inverse,
]
