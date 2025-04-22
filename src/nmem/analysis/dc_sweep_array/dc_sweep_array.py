import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from nmem.analysis.analysis import plot_linear_fit

# Load the .mat file
data = sio.loadmat("SPG806_C5_array_ic_vs_ih_data.mat")
ic_vs_ih = data["ic_vs_ih_data"]

# Extract the fields from the MATLAB struct
heater_currents = ic_vs_ih["heater_currents"][0, 0]  # cell array (nested)
avg_current = ic_vs_ih["avg_current"][0, 0]
ystd = ic_vs_ih["ystd"][0, 0]
cell_names = ic_vs_ih["cell_names"][0, 0]  # Extract cell names

# Initialize plot
plt.figure(figsize=(8, 6))

# Initialize lists to store intercepts
x_intercepts = []
y_intercepts = []

# Iterate over each folder's data
for j in range(heater_currents.shape[1]):
    ih = np.squeeze(heater_currents[0, j]) * 1e6  # Convert to µA
    ic = np.squeeze(avg_current[0, j])
    err = np.squeeze(ystd[0, j])

    # Extract cell name from MATLAB string format
    cell_name = str(cell_names[0, j][0])

    # Plot error bars
    plt.errorbar(ih, ic, yerr=err, fmt="-o", label=f"Cell {cell_name}")

    # Filter heater currents to include only those between 200 µA and 400 µA
    valid_indices = (ih >= 200) & (ih <= 600)
    ih_filtered = ih[valid_indices]
    ic_filtered = ic[valid_indices]

    # Fit a linear line and extract intercepts
    z = np.polyfit(ih_filtered, ic_filtered, 1)
    p = np.poly1d(z)
    x_intercept = -z[1] / z[0]
    y_intercept = z[1]

    # Store intercepts
    x_intercepts.append(x_intercept)
    y_intercepts.append(y_intercept)

    # Define the range for plotting the fit line
    fit_range = np.linspace(0, 800, 100)

    # Plot the fit line over the specified range
    # plt.plot(fit_range, p(fit_range), "--", label=f"_Fit {cell_name}")

# Filter x_intercepts to include only positive values
valid_indices = (np.array(x_intercepts) > 0) & (np.array(x_intercepts) < 1e3)
filtered_x_intercepts = np.array(x_intercepts)[valid_indices]
filtered_y_intercepts = np.array(y_intercepts)[valid_indices]

# Calculate the average x and y intercepts using the filtered values
avg_x_intercept = np.mean(filtered_x_intercepts)
avg_y_intercept = np.mean(filtered_y_intercepts)

# Define the range for plotting the average line
avg_fit_range = np.linspace(0, 800, 100)


# Calculate the y-values for the average line
def avg_line(x):
    slope = avg_y_intercept / avg_x_intercept
    return -slope * x + avg_y_intercept

# Plot the average line
plt.plot(
    avg_fit_range,
    avg_line(avg_fit_range),
    "-.",
    label="Average Fit Line",
    color="black",
)

# Print intercepts for all devices
for idx, cell_name in enumerate(cell_names[0]):
    print(
        f"Cell {cell_name[0]}: x-intercept = {x_intercepts[idx]:.2f}, y-intercept = {y_intercepts[idx]:.2f}"
    )

plt.xlabel("I_h (µA)")
plt.ylabel("I_c (µA)")
plt.title("I_c vs. I_h with Error Bars")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
