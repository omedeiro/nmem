import matplotlib.pyplot as plt
import numpy as np
import os
import ltspice
import logging
from matplotlib.ticker import MaxNLocator
from cycler import cycler
from nmem.analysis.analysis import set_pres_style

# Apply consistent style
set_pres_style()
plt.rcParams.update({"axes.prop_cycle": cycler(color=["#1f77b4", "#d62728"])})

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- Load and extract ---
def load_ltspice_data(raw_dir: str) -> dict:
    """Load LTspice data from .raw files in the specified directory."""
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
            logging.warning(f"Skipping {fname}: {type(e).__name__} — {e}")
            continue
        data_dict[fname] = ltsp
    logging.info(f"Loaded {len(data_dict)} LTspice files from {raw_dir}")
    return data_dict


def extract_trace_data(ltsp, time_scale=1, voltage_scale=1e3):
    """Extract relevant trace data from an LTspice object."""
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


def plot_waveform(data, regions, output_dir, label_prefix="waveform"):
    """Plot the waveform with optional highlighted regions."""
    time = data["time"]
    left_current = data["left_current"]
    right_current = data["right_current"]
    ichl = data["ichl"]
    ichr = data["ichr"]
    input_bias = data["input_bias"]
    enable_bias = data["enable_bias"]
    voltage = data["voltage"]

    xlim = (time[0], time[-1])
    os.makedirs(output_dir, exist_ok=True)

    # Base waveform plot
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
    plt.show()
    # fig.savefig(os.path.join(output_dir, f"{label_prefix}_base.png"), transparent=True)
    plt.close(fig)


def main():
    raw_dir = "data/write_amp_sweep"
    output_dir = os.path.join(raw_dir, "plots")
    regions = {
        "A": (20, 180),
        "B": (200, 400),
        "C": (480, 520),
        "D": (1220, 1400),
        "E": (1620, 1780),
        "F": (1820, 1980),
        "G": (2600, 2800),
        "H": (2820, 3100),
    }

    data_dict = load_ltspice_data(raw_dir)

    for key, ltsp in data_dict.items():
        try:
            amp_uA = int(key.split("_")[-1].replace("u.raw", ""))
            if 605 <= amp_uA <= 795:
                data = extract_trace_data(ltsp, time_scale=1e9)
                plot_waveform(
                    data, regions, output_dir, label_prefix=f"waveform_{amp_uA}"
                )
        except ValueError:
            continue


if __name__ == "__main__":
    main()
