import matplotlib.pyplot as plt
from nmem.analysis.analysis import set_pres_style, set_inter_font

set_inter_font()
set_pres_style()
# Energy values in femtojoules
labels = ["Write (W)", "Read (R)", "Enable-Write (EW)", "Enable-Read (ER)"]
energies_fj = [46, 31, 1256, 202]
colors = ["royalblue", "mediumseagreen", "darkorange", "orchid"]

plt.figure(figsize=(6, 4))
bars = plt.bar(labels, energies_fj, color=colors)

# Annotate the bars with energy values
for bar, energy in zip(bars, energies_fj):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 30, f"{energy} fJ",
             ha='center', va='bottom', fontsize=10)

plt.ylabel("Energy per Operation (fJ)")
plt.xticks(rotation=20, ha='right')

plt.title("Measured Energy of SNM Pulses")
plt.yscale("log")  # Optional
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.ylim(top=max(energies_fj) * 1.5)
plt.tight_layout()
plt.savefig("snm_pulse_energy_bar_chart.png", dpi=600)
plt.show()
