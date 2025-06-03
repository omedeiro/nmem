import os

import ltspice
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.ticker import MaxNLocator

from nmem.analysis.analysis import set_pres_style

# Apply consistent style
set_pres_style()
plt.rcParams.update({"axes.prop_cycle": cycler(color=["#1f77b4", "#d62728"])})


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


def extract_trace_data(ltsp, time_scale=1, voltage_scale=1e3):
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


def plot_full_waveform_with_biases(data, regions, output_dir, yaxis_limits):
    time = data["time"]
    left_current = data["left_current"]
    right_current = data["right_current"]
    ichl = data["ichl"]
    ichr = data["ichr"]
    input_bias = data["input_bias"]
    enable_bias = data["enable_bias"]
    voltage = data["voltage"]

    xlim = (0, 3000)
    os.makedirs(output_dir, exist_ok=True)

    # --- Base waveform with no highlight ---
    fig, axs = plt.subplots(
        3,
        1,
        figsize=(12, 6),
        dpi=300,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 2, 1]},
    )

    # Input signals
    axs[0].plot(time, input_bias, label="Input Bias", color="tab:green", linewidth=1.5)
    axs[0].plot(
        time, enable_bias, label="Enable Bias", color="tab:orange", linewidth=1.5
    )
    axs[0].set_ylabel("Input (uA)")
    axs[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    axs[0].grid(True, linestyle="--", linewidth=0.3)

    # Device currents
    axs[1].plot(time, left_current, label="Left Current", color="C0", linewidth=1.5)
    axs[1].plot(time, right_current, label="Right Current", color="C1", linewidth=1.5)
    axs[1].plot(time, ichl, linestyle="--", color="C0", alpha=0.5, linewidth=1.0)
    axs[1].plot(time, ichr, linestyle="--", color="C1", alpha=0.5, linewidth=1.0)
    axs[1].set_ylabel("Device Currents (uA)")
    axs[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    axs[1].grid(True, linestyle="--", linewidth=0.3)

    # Output signal
    axs[2].plot(
        time, voltage, label="Output Voltage", color="tab:purple", linewidth=1.5
    )
    axs[2].set_xlabel("Time (ns)")
    axs[2].set_ylabel("Voltage (mV)")
    axs[2].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    axs[2].grid(True, linestyle="--", linewidth=0.3)

    for ax in axs:
        ax.yaxis.set_major_locator(MaxNLocator(4))
    axs[0].set_xlim(xlim)

    plt.tight_layout(pad=1.5)
    fig.savefig(os.path.join(output_dir, "waveform_0_with_bias.png"), transparent=True)
    plt.close(fig)

    # --- Highlighted waveform for each region ---
    for label, (start, end) in regions.items():
        fig, ax = plt.subplots(figsize=(11, 2.8), dpi=300)

        ax.plot(time, left_current, label="Left Current", color="C0", linewidth=1.5)
        ax.plot(time, right_current, label="Right Current", color="C1", linewidth=1.5)
        ax.plot(
            time,
            ichl,
            linestyle="--",
            color="C0",
            alpha=0.5,
            linewidth=1.5,
            label="Left Critical Current",
        )
        ax.plot(
            time,
            ichr,
            linestyle="--",
            color="C1",
            alpha=0.5,
            linewidth=1.5,
            label="Right Critical Current",
        )
        ax.axvspan(start, end, color="grey", alpha=0.25)
        ax.text(
            (start + end) / 2,
            axs[1].get_ylim()[1] * 0.8,
            label,
            ha="center",
            va="top",
            fontsize=12,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
        )
        ax.set_ylabel("Currents (μA)")
        ax.grid(True, linestyle="--", linewidth=0.3)

        # axs[2].plot(time, voltage, label="Output Voltage", color="tab:purple", linewidth=1.5)
        # axs[2].axvspan(start, end, color="grey", alpha=0.25)
        ax.set_xlabel("Time (ns)")
        ax.legend(
            loc="lower right",
            bbox_to_anchor=(1.01, 1),
            fontsize=12,
            ncol=4,
            frameon=False,
        )

        # axs[2].grid(True, linestyle="--", linewidth=0.3)
        ax.set_xlim(xlim)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
        plt.tight_layout(pad=1.5)
        fig.savefig(
            os.path.join(output_dir, f"waveform_{label}_with_bias.png"),
            transparent=True,
        )
        plt.show()
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(3.0, 3.0), dpi=300)
        mask = (time >= start) & (time <= end)
        if label in ["F"]:
            sign_bool = -1
        else:
            sign_bool = 1
            

        ax.plot(time[mask], left_current[mask], label="Left Current", color="C0", linewidth=1.5)
        ax.plot(time[mask], right_current[mask], label="Right Current", color="C1", linewidth=1.5)
        ax.plot(
            time[mask],
            ichl[mask] * sign_bool,
            linestyle="--",
            color="C0",
            alpha=0.5,
            linewidth=1.5,
            label="Left Critical Current",
        )
        ax.plot(
            time[mask],
            ichr[mask] * sign_bool,
            linestyle="--",
            color="C1",
            alpha=0.5,
            linewidth=1.5,
            label="Right Critical Current",
        )
        ax.text(
            0.1,
            0.9,
            label,
            ha="center",
            va="top",
            fontsize=12,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
            transform=ax.transAxes,
        )
        ax.set_ylim(yaxis_limits[label])
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
        ax.set_ylabel("Currents (μA)")
        ax.set_xlabel("Time (ns)")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        if label == "A":
            ax_position = ax.get_position()
        else:
            ax.set_position(ax_position)

        fig.savefig(
            os.path.join(output_dir, f"waveform_{label}_with_bias_zoom.png"),
            transparent=True,
        )

        if label == "D":
            fig, ax = plt.subplots(figsize=(3.0, 3.0), dpi=300)
            ax.plot(time[mask], voltage[mask], label="Output Voltage", color="tab:purple", linewidth=1.5)
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("Voltage (mV)")
            fig.savefig(
                os.path.join(output_dir, f"waveform_{label}_voltage_zoom.png"),
                transparent=True,
            )

        if label == "H":
            fig, ax = plt.subplots(figsize=(3.0, 3.0), dpi=300)
            ax.plot(time[mask], ichl[mask], label="Left Critical Current", color="C0", linewidth=1.5, alpha=0.5, linestyle="--")
            ax.plot(time[mask], ichr[mask], label="Right Critical Current", color="C1", linewidth=1.5, alpha=0.5, linestyle="--")
            ax.plot(time[mask], left_current[mask], label="Left Current", color="C0", linewidth=1.5)
            ax.plot(time[mask], right_current[mask], label="Right Current", color="C1", linewidth=1.5)
            ax.set_ylim(660, 680)
            ax.set_xlim(2880, 2950)
            # ax.set_xlabel("Time (ns)")
            # ax.set_ylabel("Currents (μA)")
            fig.savefig(
                os.path.join(output_dir, f"waveform_{label}_ichl_ichr_zoom.png"),
                transparent=True,
            )
        plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    raw_dir = "data/write_amp_sweep_2"
    data_dict = load_ltspice_data(raw_dir)

    # --- Pull one trace for full waveform view ---
    for key, ltsp in data_dict.items():
        try:
            amp_uA = int(key.split("_")[-1].replace("u.raw", ""))
            if 705 <= amp_uA <= 745:
                data = extract_trace_data(ltsp, time_scale=1e9)
                break
        except ValueError:
            continue

    regions = {
        "A": (20, 180),
        "B": (200, 400),
        "C": (400, 600),
        "D": (1220, 1400),
        "E": (1620, 1780),
        "F": (1820, 1980),
        "G": (2600, 2800),
        "H": (2820, 3100),
    }
    yaxis_limits = {
        "A": (-150, 150),
        "B": (-150, 150),
        "C": (-100, 1500),
        "D": (-100, 1500),
        "E": (-150, 150),
        "F": (-150, 150),
        "G": (-100, 1500),
        "H": (-600, 1500),
    }


    output_dir = os.path.join(raw_dir, "full_waveform_steps")
    plot_full_waveform_with_biases(data, regions, output_dir, yaxis_limits)
