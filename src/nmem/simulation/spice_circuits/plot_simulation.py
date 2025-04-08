import matplotlib.pyplot as plt
import numpy as np
import os
import ltspice
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cycler import cycler
from nmem.simulation.spice_circuits.plotting import apply_snm_style

# apply_snm_style()
# Global Plot Settings
plt.rcParams.update({
    "font.size": 5,
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

def plot_inset(ax, data_dict, start_time, end_time, xlim, ylim):
    inset_ax = inset_axes(ax, width="30%", height="30%", loc="lower left")
    plot_trace_slice(inset_ax, data_dict, start_time, end_time)
    inset_ax.set_xlim(*xlim)
    inset_ax.set_ylim(*ylim)
    inset_ax.set_xticks([])
    inset_ax.yaxis.tick_right()
    inset_ax.yaxis.set_major_locator(MaxNLocator(3))

def plot_transient_traces(data_dict, axs=None):
    if axs is None:
        fig, axs = plt.subplots(4, 1, figsize=(10, 6), sharex=True,
                                gridspec_kw={"height_ratios": [0.3, 0.3, 0.3, 1]})
    else:
        fig = None

    for key, ltsp in data_dict.items():
        try:
            amp_uA = int(key.split("_")[-1].replace("u.raw", ""))
        except ValueError:
            continue
        if not (45 <= amp_uA <= 65):
            continue

        data = extract_trace_data(ltsp, time_scale=1e9, voltage_scale=1e3)

        axs[3].plot(data["time"], data["left_current"], color="C0")
        axs[3].plot(data["time"], data["right_current"], color="C1")
        axs[3].plot(data["time"], data["ichl"], linestyle="--", color="C0")
        axs[3].plot(data["time"], data["ichr"], linestyle="--", color="C1")

        axs[2].plot(data["time"], data["voltage"], color="C3")
        axs[0].plot(data["time"], data["input_bias"], color="C2")
        axs[1].plot(data["time"], data["enable_bias"], color="C4")

    for ax in axs:
        ax.grid()
        ax.yaxis.set_major_locator(MaxNLocator(3))

    axs[0].set_ylabel("Input Bias (uA)")
    axs[1].set_ylabel("Enable Bias (uA)")
    axs[2].set_ylabel("Voltage (mV)")
    axs[3].set_ylabel("Current (uA)")
    axs[3].set_xlabel("Time (s)")

    if fig:
        plt.tight_layout()
        plt.show()
    return axs

def plot_write_vs_persistent(write_list, persistent_list):
    fig, ax = plt.subplots(figsize=(7, 7))
    sorted_indices = np.argsort(write_list)
    ax.plot(np.array(write_list)[sorted_indices],
            np.array(persistent_list)[sorted_indices], "o-")
    ax.set_xlabel("Write Current (uA)")
    ax.set_ylabel("Persistent Current (uA)")
    ax.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    raw_dir = "data/write_amp_sweep"  # Update this path as needed
    data_dict = load_ltspice_data(raw_dir)

    fig, axs = plt.subplot_mosaic(
        [
            ["X", "X", "X", "X"],
            ["A", "B", "C", "D"],
            ["Y", "Y", "Y", "Y"],
            ["E", "F", "G", "H"],
        ],
        width_ratios=[1, 1, 1, 1],
        height_ratios=[1, 1, 1, 1],
        figsize=(5.6, 4.2),
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
    
    axs["E"].set_ylabel("Current (uA)")
    axs["E"].set_xlabel("Time (ns)")
    axs["A"].set_ylabel("Current (uA)")
    axs["A"].set_xlabel("Time (ns)")

    # Add inset to axs["H"]

    inset_ax: plt.Axes = inset_axes(axs["H"], width="45%", height="30%", loc="lower left")
    plot_trace_slice(inset_ax, data_dict, 2100, 2950)
    inset_ax.set_xlim(1298, 1305)
    inset_ax.set_ylim(610, 630)
    inset_ax.set_xticks([])
    inset_ax.yaxis.tick_right()
    inset_ax.yaxis.set_major_locator(MaxNLocator(3))
    inset_ax.set_yticks([])

    # Add inset to axs["D"]
    inset_ax2: plt.Axes = inset_axes(axs["D"], width="45%", height="30%", loc="lower left")
    plot_trace_slice(inset_ax2, data_dict, 1220, 1400)
    inset_ax2.set_xlim(1298, 1305)
    inset_ax2.set_ylim(610, 630)
    inset_ax2.set_xticks([])
    inset_ax2.yaxis.tick_right()
    inset_ax2.yaxis.set_major_locator(MaxNLocator(3))
    inset_ax2.set_yticks([])
    
    fig.subplots_adjust(hspace=0.2, wspace=0.4)
    plt.savefig(
        "data/write_amp_sweep/plot_write_persistent.pdf",
        bbox_inches="tight",
    )
