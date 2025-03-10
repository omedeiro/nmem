import ltspice
import numpy as np
import matplotlib.pyplot as plt

# Path to your LTspice .raw file
file_path = "nmem_cell.raw"

# Load the LTspice data
l = ltspice.Ltspice(file_path)
l.parse()  # Parse the raw file

# Extract specific signals
time = l.get_time()  # Time vector
voltage = l.get_data('Ix(HR:drain)')  # Example: Node voltage

# Plot the extracted data
plt.figure(figsize=(8, 5))
plt.plot(time, voltage, label="V(n003)")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("LTspice Simulation Results")
plt.legend()
plt.grid()
plt.show()
