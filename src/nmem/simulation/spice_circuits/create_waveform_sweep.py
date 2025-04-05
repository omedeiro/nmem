import numpy as np
import os
import pandas as pd

# Recreate output directory after state reset
output_dir = "data/write_amp_sweep/"
os.makedirs(output_dir, exist_ok=True)

# Core waveform generation function (uses flat-top Gaussian shape)
def flat_top_gaussian(t_center, sigma, hold_width, amp, dt):
    total_width = 8 * sigma + hold_width
    t_start = t_center - total_width / 2
    t_end = t_center + total_width / 2

    t_rise = np.arange(t_start, t_center - hold_width / 2, dt)
    t_hold = np.arange(t_center - hold_width / 2, t_center + hold_width / 2, dt)
    t_fall = np.arange(t_center + hold_width / 2, t_end, dt)

    i_rise = amp * np.exp(-0.5 * ((t_rise - (t_center - hold_width / 2)) / sigma) ** 2)
    i_hold = np.full_like(t_hold, amp)
    i_fall = amp * np.exp(-0.5 * ((t_fall - (t_center + hold_width / 2)) / sigma) ** 2)

    t = np.concatenate([t_rise, t_hold[1:], t_fall[1:]])
    i = np.concatenate([i_rise, i_hold[1:], i_fall[1:]])
    return t, i

# Save waveform to file
def save_pwl_file(filename, time, current):
    with open(filename, "w") as f:
        for t, i in zip(time, current):
            f.write(f"{t:.9e} {i:.9e}\n")

# Sweep write amplitudes from 0 to 100 µA in 10 µA steps
sweep_values = np.arange(0, 110e-6, 10e-6)  # 0 to 100 µA

# Signal parameters
t_center = 100e-9
pulse_sigma = 10e-9
hold_width = 20e-9
dt = 0.1e-9

# Store generated filenames
pwl_files = []

for amp in sweep_values:
    t, i = flat_top_gaussian(t_center, pulse_sigma, hold_width, amp, dt)
    filename = os.path.join(output_dir, f"write_amp_{int(amp * 1e6)}u.txt")
    save_pwl_file(filename, t, i)
    pwl_files.append(filename)

df = pd.DataFrame({
    "Write Amplitude (uA)": sweep_values * 1e6,
    "Waveform File": pwl_files
})
