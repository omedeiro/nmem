import ltspice
import numpy as np
import matplotlib.pyplot as plt


def plot_data(
    ax: plt.Axes, ltspice_data: ltspice.Ltspice, signal_name: str
) -> plt.Axes:
    time = ltspice_data.get_time()
    signal = ltspice_data.get_data(signal_name)
    ax.plot(time, signal, label=signal_name)
    return ax


# Path to your LTspice .raw file
file_path = "spice_simulation_raw/nmem_cell_write_read_clear.raw"

# Load the LTspice data
l = ltspice.Ltspice(file_path)
l.parse()  # Parse the raw file

# Extract specific signals
time = l.get_time()  # Time vector
voltage = l.get_data("Ix(HR:drain)")  # Example: Node voltage

# Plot the extracted data
fig, ax = plt.subplots()
ax = plot_data(ax, l, "Ix(HR:drain)")
# ax = plot_data(ax, l, "Ix(HL:drain)")
# ax = plot_data(ax, l, "V(ichl)")
ax = plot_data(ax, l, "V(ichr)")

ax.set_xlabel("Time (s)")
ax.set_ylabel("Voltage (V)")
ax.legend()
plt.show()
