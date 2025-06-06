import matplotlib.pyplot as plt

from nmem.simulation.spice_circuits.waveform_utils import (
    generate_memory_protocol_sequence,
    save_pwl_file,
)

# --- Main execution ---
t_chan, i_chan, t_enab, i_enab, ops, enab_on = generate_memory_protocol_sequence(
    cycle_time=1e-6,
    pulse_sigma=35e-9,
    hold_width_write=120e-9,
    hold_width_read=300e-9,
    hold_width_clear=5e-9,
    write_amplitude=80e-6,
    read_amplitude=725e-6,
    enab_write_amplitude=465e-6,
    enab_read_amplitude=300e-6,
    clear_amplitude=700e-6,
    dt=0.1e-9,
    seed=42,
)

# Save for LTspice
save_pwl_file("chan.txt", t_chan, i_chan)
save_pwl_file("enab.txt", t_enab, i_enab)

# Summary print
print("Slot | Operation | Wordline Active")
for i in range(min(10, len(ops))):
    print(f"{i:>4} | {ops[i]:>9} | {'ON' if enab_on[i] else 'OFF'}")

# Plot
plt.figure(figsize=(10, 4))
plt.plot(t_chan * 1e9, i_chan * 1e6, label="I_chan (data)", color="tab:blue")
plt.plot(t_enab * 1e9, i_enab * 1e6, label="I_enab (word-line)", color="tab:orange")
plt.xlabel("Time (ns)")
plt.ylabel("Current (µA)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
