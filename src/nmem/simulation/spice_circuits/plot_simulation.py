import matplotlib.pyplot as plt
import numpy as np
import os
import ltspice
from matplotlib.ticker import MaxNLocator, FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cycler import cycler
from nmem.simulation.spice_circuits.plotting import apply_snm_style
from nmem.analysis.analysis import set_pres_style

set_pres_style()
plt.rcParams.update({
    "axes.prop_cycle": cycler(color=['#1f77b4', '#d62728'])
})

# --- Load and extract ---
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

# --- Main Execution ---
if __name__ == "__main__":
    raw_dir = "data/write_amp_sweep"
    data_dict = load_ltspice_data(raw_dir)

    # --- Pull one trace for full waveform view ---
    for key, ltsp in data_dict.items():
        try:
            amp_uA = int(key.split("_")[-1].replace("u.raw", ""))
            if 35 <= amp_uA <= 45:
                data = extract_trace_data(ltsp, time_scale=1e9)
                break
        except ValueError:
            continue

    time = data["time"]
    left_current = data["left_current"]
    right_current = data["right_current"]
    ichl = data["ichl"]
    ichr = data["ichr"]

    regions = {
        "A": (20, 180), "B": (200, 400), "C": (480, 520), "D": (1220, 1400),
        "E": (1620, 1780), "F": (1820, 1980), "G": (2600, 2800), "H": (2820, 3100)
    }

    output_dir = os.path.join(raw_dir, "full_waveform_steps")
    os.makedirs(output_dir, exist_ok=True)

    # --- Save base waveform with no highlight ---
    fig, ax = plt.subplots(figsize=(12, 2.8), dpi=300)
    ax.plot(time, left_current, label="Left Current", color="C0", linewidth=0.8)
    ax.plot(time, right_current, label="Right Current", color="C1", linewidth=0.8)
    ax.plot(time, ichl, linestyle="--", color="C0", alpha=0.5)
    ax.plot(time, ichr, linestyle="--", color="C1", alpha=0.5)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Current (uA)")
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.grid(True, linestyle="--", linewidth=0.3)
    ax.legend(loc="upper right", fontsize=6)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "waveform_base.png"), transparent=True)
    plt.close(fig)

    # --- Save each overlay frame ---
    for label, (start, end) in regions.items():
        fig, ax = plt.subplots(figsize=(12, 2.8), dpi=300)
        ax.plot(time, left_current, label="Left Current", color="C0", linewidth=0.8)
        ax.plot(time, right_current, label="Right Current", color="C1", linewidth=0.8)
        ax.plot(time, ichl, linestyle="--", color="C0", alpha=0.5)
        ax.plot(time, ichr, linestyle="--", color="C1", alpha=0.5)
        ax.axvspan(start, end, color="grey", alpha=0.25)
        ax.text((start + end) / 2, ax.get_ylim()[1]*0.8, label,
                ha='center', va='top', fontsize=6,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Current (uA)")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.grid(True, linestyle="--", linewidth=0.3)
        ax.legend(loc="upper right", fontsize=6)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"waveform_{label}.png"), transparent=True)
        plt.close(fig)
