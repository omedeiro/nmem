import matplotlib.pyplot as plt
import numpy as np
import scipy
import ltspice
from nmem.simulation.spice_circuits.functions import import_raw_dir, get_current_or_voltage, get_persistent_current, process_read_data
import os
if __name__ == "__main__":
    # Example usage
    raw_dir = "data\write_amp_sweep"  # Replace with your raw files directory

    data_dict = {}
    for fname in os.listdir(raw_dir):
        if fname.endswith(".op.raw"):
            continue  # skip .op.raw files
        if not fname.endswith(".raw"):
            continue  # optionally skip non-raw files

        raw_path = os.path.join(raw_dir, fname)
        ltsp = ltspice.Ltspice(raw_path)
        ltsp.parse()

        try:
            time = ltsp.get_time()
        except Exception as e:
            print(f"Skipping {fname}: {type(e).__name__} — {e}")
            continue
        data_dict[fname] = ltsp
        print(f"Loaded {fname} with time data.")

    persistent_current_list = []
    write_current_list = []
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, ltsp in list(data_dict.items()):  
        try:
            amp_uA = int(key.split("_")[-1].replace("u.raw", ""))
        except ValueError:
            continue
        if amp_uA < -600 or amp_uA > 600:
            continue  # only plot 20uA–80uA traces

        time = ltsp.get_time()
        left_current = ltsp.get_data("Ix(HL:drain)")*1e6
        ax.plot(time, left_current, label=f"Left branch current", color="C0")
        right_current = ltsp.get_data("Ix(HR:drain)")*1e6
        ax.plot(time, right_current, label=f"right branch current", color="C1")

        write_current = amp_uA
        persistent_current = right_current[-1] * 1e6
        persistent_current_list.append(persistent_current)
        write_current_list.append(write_current)
    ichl = ltsp.get_data("I(ichl)")*1e6
    ichr = ltsp.get_data("I(ichr)")*1e6
    time = ltsp.get_time()
    ax.plot(time, ichl, label="I(ichl)", linestyle="--", color="C0")
    ax.plot(time, -ichl, label="_I(ichl)", linestyle="--", color="C0")
    ax.plot(time, ichr, label="I(ichr)", linestyle="--", color="C1")
    ax.plot(time, -ichr, label="_I(ichr)", linestyle="--", color="C1")
    ax.set_ylim(-600, 600)
    ax.grid()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Current (uA)")

    fig, ax = plt.subplots(figsize=(10, 6))
    # sort the data for plotting
    sorted_indices = np.argsort(write_current_list)
    write_current_list = np.array(write_current_list)[sorted_indices]
    persistent_current_list = np.array(persistent_current_list)[sorted_indices]
    ax.plot(write_current_list, persistent_current_list, "o-")
    ax.set_xlabel("Write Current (uA)")
    ax.set_ylabel("Persistent Current (uA)")
    ax.grid()