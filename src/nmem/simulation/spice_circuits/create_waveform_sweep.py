import numpy as np
import os
import pandas as pd
from nmem.simulation.spice_circuits.waveform_utils import generate_memory_protocol_sequence, save_pwl_file

# Recreate output directory after state reset
output_dir = "data/write_amp_sweep"
os.makedirs(output_dir, exist_ok=True)

# Sweep write amplitudes from 0 to 100 µA in 10 µA steps
sweep_values = np.arange(715e-6, 730e-6, 5e-6)  # 0 to 100 µA


# Store generated filenames
pwl_files = []


def main():
    for amp in sweep_values:
        # Generate the waveform
        t_chan, i_chan, t_enab, i_enab, ops, enab_on = (
            generate_memory_protocol_sequence(
                cycle_time=1e-6,
                pulse_sigma=35e-9,
                hold_width_write=120e-9,
                hold_width_read=300e-9,
                hold_width_clear=5e-9,
                write_amplitude=80e-6,
                read_amplitude=amp,
                enab_write_amplitude=465e-6,
                enab_read_amplitude=300e-6,
                clear_amplitude=700e-6,
                dt=0.1e-9,
                seed=42,
            )
        )

        # Save to file
        filename = os.path.join(
            output_dir, f"write_amp_sweep_{int(amp * 1e6):+05g}u.txt"
        )
        save_pwl_file(filename, t_chan, i_chan)
        pwl_files.append(filename)

    df = pd.DataFrame(
        {"Write Amplitude (uA)": sweep_values * 1e6, "Waveform File": pwl_files}
    )

    print("Generated waveform sweep.")


if __name__ == "__main__":
    main()
