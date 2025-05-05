import matplotlib.pyplot as plt
import numpy as np


# --- Radar Plot Function with scaled radial tick labels at cardinal directions ---
def plot_radar(
    metrics, units, axis_min, axis_max, normalizers, datasets, labels, styles, filename
):
    def normalize(val, vmin, vmax, fn):
        try:
            return np.clip(fn(val, vmin, vmax), 0, 1)
        except:
            return 0

    def normalize_all(data):
        return [
            normalize(data[i], axis_min[i], axis_max[i], normalizers[i])
            for i in range(num_vars)
        ] + [normalize(data[0], axis_min[0], axis_max[0], normalizers[0])]

    def format_label(metric, val, unit):
        if val is None or val <= 0 or not np.isfinite(val):
            return f"{metric}\n0 {unit}".strip()
        elif val < 1e-3 or val > 1e3:
            try:
                log_val = int(np.round(np.log10(val)))
                return f"{metric}\n$10^{{{log_val}}}$ {unit}".strip()
            except Exception:
                return f"{metric}\n{val:.1e} {unit}".strip()
        elif unit == "bits" and val >= 1000:
            return f"{metric}\n{int(val / 1000)}k"
        else:
            return f"{metric}\n{val:g} {unit}".strip()

    # Precompute angles and outer values to avoid recalculating them multiple times
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    outer_vals = [
        axis_max[i] if fn in [normalize_linear, normalize_log] else axis_min[i]
        for i, fn in enumerate(normalizers)
    ]

    # Precompute xtick labels
    xtick_labels = [
        format_label(metrics[i], outer_vals[i], units[i]) for i in range(num_vars)
    ]

    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300, subplot_kw=dict(polar=True))

    # Precompute normalized datasets to avoid redundant calculations
    normalized_datasets = [normalize_all(data) for data in datasets]

    for norm, label, style in zip(normalized_datasets, labels, styles):
        ax.plot(angles, norm, **style, label=label)
        ax.fill(angles, norm, alpha=style.get("alpha", 0.2), color=style.get("color"))

    ax.set_ylim(0, 1)
    from math import degrees

    ax.set_xticks([])  # Remove default labels
    label_radius = 1.15

    # Optimize label placement by precomputing alignment
    for angle, label in zip(angles[:-1], xtick_labels):
        angle_deg = degrees(angle)
        ha = "center"
        if 90 < angle_deg < 270:
            ha = "right"
        elif angle_deg < 90 or angle_deg > 270:
            ha = "left"
        ax.text(angle, label_radius, label, ha=ha, va="center", fontsize=7)

    ax.set_yticklabels([])
    ax.grid(True, lw=0.5, ls="--", alpha=0.4)
    ax.spines["polar"].set_visible(False)

    # Custom radial tick labels for each metric at cardinal directions
    num_vars = len(metrics)  # 5 if you've added "Access Time"
    cardinal_angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    cardinal_axes = [0, 1, 2, 3, 4]  # Metric indices

    # --- Custom radial ticks per metric ---
    custom_ticks = {
        "Density": [1, 5, 10],
        "Capacity": [1e3, 1e4, 1e5, 1e6],
        "BER": [1e-2, 1e-3, 1e-4, 1e-5],
        "Access energy": [1e-15, 1e-13, 1e-11],
        "Access time": [1e-12, 1e-10, 1e-8],
    }

    # Precompute custom radial ticks for each metric
    for angle, i in zip(angles[:-1], range(num_vars)):
        vmin, vmax = axis_min[i], axis_max[i]
        fn = normalizers[i]
        unit = units[i]
        metric = metrics[i]

        raw_vals = custom_ticks.get(metric, np.linspace(vmin, vmax, num=5))
        tick_pairs = [(val, normalize(val, vmin, vmax, fn)) for val in raw_vals]
        tick_pairs = [(val, r) for val, r in tick_pairs if np.isfinite(r)]
        tick_pairs.sort(key=lambda x: x[1])

        for val, r in tick_pairs[1:]:
            if not np.isfinite(val) or not np.isfinite(r):
                continue
            if val < 1e-2 or val > 1e3:
                exp = int(np.floor(np.log10(val)))
                label = f"$10^{{{exp}}}$"
            elif unit == "bits" and val >= 1000:
                label = f"{int(val / 1000)}k"
            else:
                label = f"{val:g}"

            ax.text(angle, r, label, fontsize=6, ha="center", va="center", color="gray")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        fontsize=7,
        frameon=False,
        ncol=len(labels),
    )
    plt.tight_layout()
    plt.savefig(filename, transparent=True)
    plt.show()


# --- Normalization functions ---
normalize_linear = lambda val, vmin, vmax: (val - vmin) / (vmax - vmin)
normalize_inverse = lambda val, vmin, vmax: (vmax - val) / (vmax - vmin)
normalize_log = lambda val, vmin, vmax: (np.log10(val) - np.log10(vmin)) / (
    np.log10(vmax) - np.log10(vmin)
)
normalize_log_inverse = lambda val, vmin, vmax: (np.log10(vmax) - np.log10(val)) / (
    np.log10(vmax) - np.log10(vmin)
)

# --- Metric definitions ---
metrics = ["Density", "Capacity", "BER", "Access energy", "Access time"]
units = ["Mb/cm²", "bits", "", "J/bit", "s"]
axis_min = [0.1, 0.1, 1e-6, 1e-16, 1e-12]
axis_max = [10.0, 1e6, 1e-2, 1e-1, 1e-3]
normalizers = [
    normalize_linear,
    normalize_log,
    normalize_log_inverse,
    normalize_log_inverse,
    normalize_log_inverse,
]

# --- Data ---
values_snm_projected = [5.0, 1e4, 1e-5, 1e-13, 1e-9]
values_snm_new = [2.6, 64, 1e-5, 100e-13, 20e-9]
values_snm_old = [2.0, 1, 1e-3, 1300e-13, 10e-9]
values_sce = [0.9, 4e3, 1e-5, 50e-16, 10e-12]

datasets = [values_snm_new, values_snm_old, values_sce, values_snm_projected]
labels = ["SNM", "SNM_old", "SCE", "SNM_projected"]
styles = [
    {"color": "black", "linewidth": 1.2},
    {"color": "blue", "linewidth": 1.2, "linestyle": "--", "alpha": 0.15},
    {"color": "red", "linewidth": 1.2, "linestyle": "--", "alpha": 0.15},
    {"color": "green", "linewidth": 1.2, "linestyle": "--", "alpha": 0.15},
]

# --- Call the updated plot function ---
plot_radar(
    metrics,
    units,
    axis_min,
    axis_max,
    normalizers,
    datasets,
    labels,
    styles,
    "radar_snm_cmos_normalized.pdf",
)
