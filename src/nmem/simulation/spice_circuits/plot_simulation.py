import matplotlib.pyplot as plt
import numpy as np
import os
import ltspice

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

def plot_transient_traces_broken_axes(data_dict: dict):
    fig, (ax4, ax3, ax1, ax2) = plt.subplots(4, 1, sharex=True, figsize=(10, 8), gridspec_kw={'height_ratios': [0.3, .3, .3, 1]})
    persistent_current_list = []
    write_current_list = []

    for key, ltsp in data_dict.items():
        try:
            amp_uA = int(key.split("_")[-1].replace("u.raw", ""))
        except ValueError:
            continue
        if amp_uA < 35 or amp_uA > 45:
            continue
        time = ltsp.get_time()
        left_current = ltsp.get_data("Ix(HL:drain)") * 1e6
        right_current = ltsp.get_data("Ix(HR:drain)") * 1e6
        ichl = ltsp.get_data("I(ichl)") * 1e6
        ichr = ltsp.get_data("I(ichr)") * 1e6
        enable_bias = ltsp.get_data("I(R1)") * 1e6
        voltage = ltsp.get_data("V(out)") * 1e6

        write_current = amp_uA
        persistent_current = right_current[-1]
        persistent_current_list.append(persistent_current)
        write_current_list.append(write_current)

        # Top and bottom current axes
        ax1.plot(time, left_current, color="C0")
        ax1.plot(time, right_current, color="C1")
        ax1.plot(time, ichl, linestyle="--", color="C0")
        ax1.plot(time, ichr, linestyle="--", color="C1")

        ax2.plot(time, left_current, color="C0")
        ax2.plot(time, right_current, color="C1")
        ax2.plot(time, ichl, linestyle="--", color="C0")
        ax2.plot(time, ichr, linestyle="--", color="C1")

        ax3.plot(time, enable_bias, color="C2")
        ax4.plot(time, voltage, color="C3")

    # Y-axis break setup
    ax1.set_ylim(300, 2000)
    ax2.set_ylim(-50, 100)
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.tick_params(labeltop=False)
    ax2.xaxis.tick_bottom()

    # Diagonal lines for axis break
    d = .015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # Labels
    ax1.set_ylabel("Current (uA)")
    ax2.set_ylabel("Current (uA)")
    ax2.set_xlabel("Time (s)")
    ax3.set_ylabel("Enable (uA)")
    ax4.set_ylabel("Voltage (uV)")

    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    plt.tight_layout()
    plt.show()

    return write_current_list, persistent_current_list

def plot_transient_traces(data_dict: dict):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    for key, ltsp in data_dict.items():
        try:
            amp_uA = int(key.split("_")[-1].replace("u.raw", ""))
        except ValueError:
            continue
        if amp_uA < 35 or amp_uA > 45:
            continue
        time = ltsp.get_time()
        left_current = ltsp.get_data("Ix(HL:drain)") * 1e6
        right_current = ltsp.get_data("Ix(HR:drain)") * 1e6
        ichl = ltsp.get_data("I(ichl)") * 1e6
        ichr = ltsp.get_data("I(ichr)") * 1e6
        enable_bias = ltsp.get_data("I(R1)") * 1e6
        voltage = ltsp.get_data("V(out)") * 1e6

        ax1.plot(time, left_current, color="C0")
        ax1.plot(time, right_current, color="C1")
        ax1.plot(time, ichl, linestyle="--", color="C0")
        ax1.plot(time, ichr, linestyle="--", color="C1")
        ax1.plot(time, enable_bias, color="C2")

        ax2.plot(time, voltage, color="C3")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Current (uA)")
    ax1.grid()
    plt.tight_layout()  
    plt.show()


def plot_write_persistent(write_current_list, persistent_current_list):
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_indices = np.argsort(write_current_list)
    wc_sorted = np.array(write_current_list)[sorted_indices]
    pc_sorted = np.array(persistent_current_list)[sorted_indices]
    ax.plot(wc_sorted, pc_sorted, "o-")
    ax.set_xlabel("Write Current (uA)")
    ax.set_ylabel("Persistent Current (uA)")
    ax.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    raw_dir = "data/write_amp_sweep"  # Update this path as needed
    data_dict = load_ltspice_data(raw_dir)
    plot_transient_traces(data_dict)
