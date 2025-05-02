import matplotlib.pyplot as plt
import numpy as np
import os
import ltspice
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cycler import cycler
from nmem.simulation.spice_circuits.plotting import apply_snm_style
from nmem.analysis.analysis import set_pres_style
from matplotlib.ticker import FuncFormatter

set_pres_style()
plt.rcParams.update({
    "axes.prop_cycle": cycler(color=['#1f77b4', '#d62728'])
})

def load_ltspice_data(raw_dir: str) -> dict:
    data_dict = {}
    for fname in os.listdir(raw_dir):
        if not fname.endswith(".raw") or fname.endswith(".op.raw"):
            continue
        raw_path = os.path.join(raw_dir, fname)
        ltsp = ltspice.Ltspice(raw_path)
        ltsp.parse()
        try:
            _ = ltsp.get_time()
        except Exception as e:
            print(f"Skipping {fname}: {type(e).__name__} — {e}")
            continue
        data_dict[fname] = ltsp
    return data_dict

def extract_trace_data(ltsp, time_scale=1, voltage_scale=1e6):
    return {
        "time": ltsp.get_time() * time_scale,
        "left_current": ltsp.get_data("Ix(HL:drain)") * 1e6,
        "right_current": ltsp.get_data("Ix(HR:drain)") * 1e6,
        "ichl": ltsp.get_data("I(ichl)") * 1e6,
        "ichr": ltsp.get_data("I(ichr)") * 1e6,
        "enable_bias": ltsp.get_data("I(R1)") * 1e6,
        "input_bias": ltsp.get_data("I(R2)") * 1e6,
        "voltage": ltsp.get_data("V(out)") * voltage_scale,
    }

def plot_trace_slice(ax, data_dict, start_time, end_time, invert_ic=False, add_inverted=False):
    for key, ltsp in data_dict.items():
        try:
            amp_uA = int(key.split("_")[-1].replace("u.raw", ""))
        except ValueError:
            continue
        if not (35 <= amp_uA <= 45):
            continue
        data = extract_trace_data(ltsp, time_scale=1e9)
        mask = (data["time"] >= start_time) & (data["time"] <= end_time)
        if data["time"][mask][0] > 1600:
            data["time"][mask] -= 1600
        ax.plot(data["time"][mask], data["left_current"][mask], color="C0")
        ax.plot(data["time"][mask], data["right_current"][mask], color="C1")
        ichl = -data["ichl"][mask] if invert_ic else data["ichl"][mask]
        ichr = -data["ichr"][mask] if invert_ic else data["ichr"][mask]
        ax.plot(data["time"][mask], ichl, linestyle="--", color="C0", dashes=(1, 1))
        ax.plot(data["time"][mask], ichr, linestyle="--", color="C1", dashes=(1, 1))
        if add_inverted:
            ax.plot(data["time"][mask], -ichl, linestyle="--", color="C0", dashes=(1, 1))
            ax.plot(data["time"][mask], -ichr, linestyle="--", color="C1", dashes=(1, 1))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_ylim(-40, 70)

if __name__ == "__main__":
    raw_dir = "data/write_amp_sweep"
    data_dict = load_ltspice_data(raw_dir)

    fig, axs = plt.subplot_mosaic(
        [["X", "X", "X", "X"],
         ["A", "B", "C", "D"],
         ["Y", "Y", "Y", "Y"],
         ["E", "F", "G", "H"]],
        width_ratios=[1, 1, 1, 1],
        height_ratios=[1, 1, 1, 1],
        figsize=(7, 5),
    )

    axs["X"].set_axis_off()
    axs["Y"].set_axis_off()
    plot_trace_slice(axs["A"], data_dict, 20, 180)
    plot_trace_slice(axs["B"], data_dict, 200, 400)
    plot_trace_slice(axs["C"], data_dict, 480, 520)
    plot_trace_slice(axs["D"], data_dict, 1220, 1400)
    axs["D"].set_ylim(-500, 800)
    plot_trace_slice(axs["E"], data_dict, 1620, 1780, add_inverted=True)
    plot_trace_slice(axs["F"], data_dict, 1820, 1980, invert_ic=True)
    plot_trace_slice(axs["G"], data_dict, 2600, 2800)
    plot_trace_slice(axs["H"], data_dict, 2820, 3100)
    axs["A"].set_ylim(-50, 50)
    axs["B"].set_ylim(-30, 50)
    axs["C"].set_ylim(-30, 50)
    axs["E"].set_ylim(-50, 50)
    axs["F"].set_ylim(-50, 30)
    axs["G"].set_ylim(-50, 800)
    axs["H"].set_ylim(-500, 800)
    axs["E"].set_xlabel("Time (ns)")
    axs["F"].set_xlabel("Time (ns)")
    axs["G"].set_xlabel("Time (ns)")
    axs["H"].set_xlabel("Time (ns)")
    axs["E"].set_ylabel("Current (uA)")
    axs["A"].set_ylabel("Current (uA)")

    inset_ax_h = inset_axes(axs["H"], width="45%", height="30%", loc="lower left")
    plot_trace_slice(inset_ax_h, data_dict, 2100, 2950)
    inset_ax_h.set_xlim(1298, 1305)
    inset_ax_h.set_ylim(610, 630)
    inset_ax_h.set_xticks([]); inset_ax_h.set_yticks([])
    inset_ax_h.yaxis.tick_right(); inset_ax_h.yaxis.set_major_locator(MaxNLocator(3))

    inset_ax_d = inset_axes(axs["D"], width="45%", height="30%", loc="lower left")
    plot_trace_slice(inset_ax_d, data_dict, 1220, 1400)
    inset_ax_d.set_xlim(1298, 1305)
    inset_ax_d.set_ylim(610, 630)
    inset_ax_d.set_xticks([]); inset_ax_d.set_yticks([])
    inset_ax_d.yaxis.tick_right(); inset_ax_d.yaxis.set_major_locator(MaxNLocator(3))

    fig.subplots_adjust(hspace=0.2, wspace=0.4)
    plt.savefig("data/write_amp_sweep/plot_write_persistent.pdf", bbox_inches="tight")

    # === Save subplots A–J ===
    output_dir = os.path.join(raw_dir, "individual_plots")
    os.makedirs(output_dir, exist_ok=True)
    label_map = {
        "A": axs["A"], "B": axs["B"], "C": axs["C"], "D": axs["D"],
        "E": axs["E"], "F": axs["F"], "G": axs["G"], "H": axs["H"],
        "I": inset_ax_d, "J": inset_ax_h
    }

    for label, ax in label_map.items():
        if not ax.has_data():
            continue

        fig_single, ax_main = plt.subplots(figsize=(3.2, 2.4), dpi=300)

        for line in ax.get_lines():
            ax_main.plot(
                line.get_xdata(), line.get_ydata(),
                color=line.get_color(),
                linestyle=line.get_linestyle(),
                linewidth=1.2,
                marker=line.get_marker()
            )

        # Ensure same view window and labels
        ax_main.set_xlim(ax.get_xlim())
        ax_main.set_ylim(ax.get_ylim())
        ax_main.set_xlabel("Time (ns)")
        ax_main.set_ylabel("Current (uA)")
        ax_main.yaxis.set_label_coords(-0.18, 0.5)  # lock x-position

        # Uniform ticks and spacing
        ax_main.yaxis.set_major_locator(MaxNLocator(3))
        ax_main.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:>4.0f}"))  # same width
        ax_main.tick_params(direction="out", length=3, width=0.8, labelsize=8)
        ax_main.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)

        # Manually fix axis position (left margin and spacing)
        ax_main.set_position([0.22, 0.2, 0.72, 0.75])  # [left, bottom, width, height]

        fig_single.savefig(
            os.path.join(output_dir, f"subplot_{label}.png"),
            bbox_inches=None,  # don't crop
            transparent=True
        )
        plt.close(fig_single)

    # === Full waveform overview plot ===
    for key, ltsp in data_dict.items():
        try:
            amp_uA = int(key.split("_")[-1].replace("u.raw", ""))
        except ValueError:
            continue
        if not (35 <= amp_uA <= 45):
            continue
        data = extract_trace_data(ltsp, time_scale=1e9)
        break

    fig_ov, ax_ov = plt.subplots(figsize=(12, 2.8), dpi=300)
    ax_ov.plot(data["time"], data["left_current"], label="Left Current", color="C0")
    ax_ov.plot(data["time"], data["right_current"], label="Right Current", color="C1")
    ax_ov.plot(data["time"], data["ichl"], linestyle="--", color="C0", alpha=0.5)
    ax_ov.plot(data["time"], data["ichr"], linestyle="--", color="C1", alpha=0.5)

    regions = {
        "A": (20, 180), "B": (200, 400), "C": (480, 520), "D": (1220, 1400),
        "E": (1620, 1780), "F": (1820, 1980), "G": (2600, 2800), "H": (2820, 3100)
    }

    for label, (start, end) in regions.items():
        ax_ov.axvspan(start, end, color="grey", alpha=0.2)
        ax_ov.text((start + end) / 2, ax_ov.get_ylim()[1]*0.8, label,
                   ha='center', va='top', fontsize=6,
                   bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

    ax_ov.set_xlabel("Time (ns)")
    ax_ov.set_ylabel("Current (uA)")
    ax_ov.yaxis.set_major_locator(MaxNLocator(4))
    ax_ov.grid(True, linestyle="--", linewidth=0.3)
    ax_ov.legend(loc="center", ncol=2, bbox_to_anchor=(0.5, 1.05))
    plt.tight_layout()
    plt.savefig("data/write_amp_sweep/full_waveform_overview.png", dpi=300)
    plt.close(fig_ov)
