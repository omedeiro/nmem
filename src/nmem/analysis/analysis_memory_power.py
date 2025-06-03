import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Polygon, Rectangle

from nmem.analysis.analysis import set_inter_font, set_pres_style

# Apply custom styles
set_inter_font()
set_pres_style()

# ---------------------- Data ----------------------
labels = ["Write (W)", "Read (R)", "Enable-Write (EW)", "Enable-Read (ER)"]
energies_fj = [46, 31, 1256, 202]
colors = ["gray", "gray", "darkred", "darkred"]
bar_positions = np.arange(len(labels))

# ---------------------- Shading Helpers ----------------------
def darken(color, factor=0.6):
    return tuple(np.clip(factor * np.array(to_rgb(color)), 0, 1))

def lighten(color, factor=1.1):
    return tuple(np.clip(factor * np.array(to_rgb(color)), 0, 1))

# ---------------------- Plot ----------------------
fig, ax = plt.subplots(figsize=(6.5, 4.5))
bar_width = 0.6
depth = 50  # Depth in fJ units for visual extrusion

for i, (val, label, base_color) in enumerate(zip(energies_fj, labels, colors)):
    x = bar_positions[i]

    front_color = base_color
    top_color = lighten(base_color)
    side_color = darken(base_color)

    # Front face
    rect = Rectangle((x - bar_width / 2, 0), bar_width, val,
                     facecolor=front_color, edgecolor='none')
    ax.add_patch(rect)

    # Top face
    top = Polygon([
        (x - bar_width / 2, val),
        (x - bar_width / 2 + depth / 1000, val + depth),
        (x + bar_width / 2 + depth / 1000, val + depth),
        (x + bar_width / 2, val)
    ], closed=True, facecolor=top_color, edgecolor='none')
    ax.add_patch(top)

    # Side face
    side = Polygon([
        (x + bar_width / 2, 0),
        (x + bar_width / 2, val),
        (x + bar_width / 2 + depth / 1000, val + depth),
        (x + bar_width / 2 + depth / 1000, 0 + depth)
    ], closed=True, facecolor=side_color, edgecolor='none')
    ax.add_patch(side)

    # Text annotation
    ax.text(x, val + 80, f"{val} fJ", ha='center', va='bottom', fontsize=10)

# ---------------------- Axes ----------------------
ax.set_xlim(-0.5, len(labels) - 0.5)
ax.set_ylim(0, 1500)
ax.set_xticks(bar_positions)
ax.set_xticklabels(labels, rotation=20, ha='right')
ax.set_ylabel("Energy per Operation [fJ]")
ax.set_title("Measured Energy of SNM Pulses", weight='bold')
ax.grid(True, axis='y', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig("snm_energy_extrudedbar_linear.png", dpi=600)
plt.show()
