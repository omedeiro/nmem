import numpy as np
import matplotlib.pyplot as plt

# --- Radar Plot Function with scaled radial tick labels at cardinal directions ---
def plot_radar(metrics, units, axis_min, axis_max, normalizers, datasets, labels, styles, filename):
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    def normalize(val, vmin, vmax, fn):
        try:
            return np.clip(fn(val, vmin, vmax), 0, 1)
        except:
            return 0

    def normalize_all(data):
        return [normalize(data[i], axis_min[i], axis_max[i], normalizers[i])
                for i in range(num_vars)] + [normalize(data[0], axis_min[0], axis_max[0], normalizers[0])]

    def format_label(metric, val, unit):
        if val is None or val <= 0 or not np.isfinite(val):
            return f"{metric}\n0 {unit}".strip()
        elif val < 1e-3 or val > 1e3:
            try:
                log_val = int(np.round(np.log10(val)))
                return f"{metric}\n$10^{{{log_val}}}$ {unit}".strip()
            except Exception:
                return f"{metric}\n{val:.1e} {unit}".strip()
        elif unit == 'bits' and val >= 1000:
            return f"{metric}\n{int(val / 1000)}k"
        else:
            return f"{metric}\n{val:g} {unit}".strip()

    # Outer labels
    outer_vals = [axis_max[i] if fn in [normalize_linear, normalize_log] else axis_min[i] for i, fn in enumerate(normalizers)]
    xtick_labels = [format_label(metrics[i], outer_vals[i], units[i]) for i in range(num_vars)]

    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300, subplot_kw=dict(polar=True))

    for data, label, style in zip(datasets, labels, styles):
        norm = normalize_all(data)
        ax.plot(angles, norm, **style, label=label)
        ax.fill(angles, norm, alpha=style.get('alpha', 0.2), color=style.get('color'))

    ax.set_ylim(0, 1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(xtick_labels, fontsize=7)
    ax.set_yticklabels([])
    ax.grid(True, lw=0.5, ls='--', alpha=0.4)
    ax.spines['polar'].set_visible(False)

    # Custom radial tick labels for each metric at cardinal directions
    num_vars = len(metrics)  # 5 if you've added "Access Time"
    cardinal_angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    cardinal_axes = [0, 1, 2, 3, 4]  # Metric indices

    for angle, i in zip(cardinal_angles, cardinal_axes):
        vmin, vmax = axis_min[i], axis_max[i]
        fn = normalizers[i]
        unit = units[i]

        # Generate raw values
        if fn in [normalize_log, normalize_log_inverse]:
            raw_vals = np.logspace(np.log10(vmin), np.log10(vmax), num=5)
        else:
            raw_vals = np.linspace(vmin, vmax, num=5)

        # Compute (val, radius) pairs and sort by radius
        tick_pairs = [(val, normalize(val, vmin, vmax, fn)) for val in raw_vals]
        tick_pairs = [(val, r) for val, r in tick_pairs if np.isfinite(r)]
        tick_pairs.sort(key=lambda x: x[1])  # sort by radius

        for val, r in tick_pairs[1:]:  # skip smallest radius tick (center-most)
            if val < 1e-3 or val > 1e3:
                label = f"$10^{{{int(np.log10(val))}}}$"
            elif unit == 'bits' and val >= 1000:
                label = f"{int(val / 1000)}k"
            else:
                label = f"{val:g}"

            ax.text(angle, r, label, fontsize=6, ha='center', va='center', color='gray')

    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=7, frameon=False)
    plt.tight_layout()
    plt.savefig(filename, transparent=True)
    plt.show()

# --- Normalization functions ---
normalize_linear = lambda val, vmin, vmax: (val - vmin) / (vmax - vmin)
normalize_inverse = lambda val, vmin, vmax: (vmax - val) / (vmax - vmin)
normalize_log = lambda val, vmin, vmax: (np.log10(val) - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))
normalize_log_inverse = lambda val, vmin, vmax: (np.log10(vmax) - np.log10(val)) / (np.log10(vmax) - np.log10(vmin))

# --- Metric definitions ---
metrics = ['Density', 'Capacity', 'BER', 'Access energy', 'Access time']
units = ['Mb/cm²', 'bits', '', 'J/bit', 's']
axis_min = [0.1, 0.1, 1e-6, 1e-16, 1e-12]
axis_max = [10.0, 1e6, 1e-2, 1e-1, 1e-3]
normalizers = [normalize_linear, normalize_log, normalize_log_inverse, normalize_log_inverse, normalize_log_inverse]

# --- Data ---
values_snm_new = [2.6, 64, 1e-5, 100e-13, 20e-9]
values_snm_old = [2.0, 1, 1e-3, 1300e-13, 10e-9]
values_sce = [1, 4e3, 1e-5, 50e-16, 10e-12]

datasets = [values_snm_new, values_snm_old, values_sce]
labels = ['SNM', 'SNM_old', 'SCE']
styles = [
    {'color': 'black', 'linewidth': 1.2},
    {'color': 'blue', 'linewidth': 1.2, 'linestyle': '--', 'alpha': 0.15},
    {'color': 'red', 'linewidth': 1.2, 'linestyle': '--', 'alpha': 0.15}
]

# --- Call the updated plot function ---
plot_radar(metrics, units, axis_min, axis_max, normalizers, datasets, labels, styles, "radar_snm_cmos_normalized.pdf")
