import numpy as np
import matplotlib.pyplot as plt


# Save waveform to file
def save_pwl_file(filename, time, current):
    with open(filename, "w") as f:
        for t, i in zip(time, current):
            f.write(f"{t:.9e} {i:.9e}\n")


# Flat-top Gaussian pulse generator
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

    # Avoid duplicate time steps at transitions
    t = np.concatenate([t_rise, t_hold[1:], t_fall[1:]])
    i = np.concatenate([i_rise, i_hold[1:], i_fall[1:]])
    return t, i


# Build waveform for a sequence of memory ops
def generate_memory_protocol_sequence(
    cycle_time=200e-9,
    pulse_sigma=10e-9,
    hold_width_write=20e-9,
    hold_width_read=50e-9,
    hold_width_clear=0,
    write_amplitude=180e-6,
    read_amplitude=700e-6,
    enab_write_amplitude=440e-6,
    enab_read_amplitude=330e-6,
    clear_amplitude=700e-6,
    enab_fraction=0.5,
    phase_offset=0,
    dt=0.1e-9,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)

    patterns = [
        ["write_1"],
    ]

    ops = []
    for pair in patterns:
        for op in pair:
            ops.append(op)
            # if op == "read":
            #     ops.append("clear")

    t_chan, i_chan = [], []
    t_enab, i_enab = [], []

    enab_on = np.ones(len(ops), dtype=bool)
    enab_on[1:2] = False  # Disable word-line select for read operations
    enab_on[5:6] = False  # Disable word-line select for read operations
    for i, op in enumerate(ops):
        t_center = i * cycle_time + cycle_time / 2

        # --- I_chan: data line ---
        if op == "write_1":
            amp = write_amplitude
            t_vec, i_vec = flat_top_gaussian(
                t_center, pulse_sigma, hold_width_write, amp, dt
            )
            t_chan.extend(t_vec)
            i_chan.extend(i_vec)
        elif op == "write_0":
            amp = -write_amplitude
            t_vec, i_vec = flat_top_gaussian(
                t_center, pulse_sigma, hold_width_write, amp, dt
            )
            t_chan.extend(t_vec)
            i_chan.extend(i_vec)
        elif op == "read":
            amp = read_amplitude
            t_vec, i_vec = flat_top_gaussian(
                t_center, pulse_sigma, hold_width_read, amp, dt
            )
            t_chan.extend(t_vec + 20e-9)
            i_chan.extend(i_vec)

        # --- I_enab: word-line select ---
        if op in ["write_1", "write_0", "enab"]:
            amp = enab_write_amplitude
            hold = 5e-9
        elif op == "read":
            amp = enab_read_amplitude
            hold = 5e-9
        elif op == "clear":
            amp = clear_amplitude
            hold = hold_width_clear
        else:
            amp = 0

        if amp > 0 and enab_on[i]:
            t_vec, i_vec = flat_top_gaussian(t_center, pulse_sigma, hold, amp, dt)
            t_enab.extend(t_vec + phase_offset)
            i_enab.extend(i_vec)

    return (
        np.array(t_chan),
        np.array(i_chan),
        np.array(t_enab),
        np.array(i_enab),
        ops,
        enab_on,
    )


# --- Main execution ---
t_chan, i_chan, t_enab, i_enab, ops, enab_on = generate_memory_protocol_sequence(
    cycle_time=200e-9,
    pulse_sigma=15e-9,
    hold_width_write=20e-9,
    hold_width_read=50e-9,
    hold_width_clear=0,
    write_amplitude=70e-6,
    read_amplitude=710e-6,
    enab_write_amplitude=485e-6,
    enab_read_amplitude=315e-6,
    clear_amplitude=700e-6,
    dt=0.1e-9,
    seed=42,
)

# Save for LTspice
save_pwl_file("/home/omedeiro/hTron-behavioral-model/chan.txt", t_chan, i_chan)
save_pwl_file("/home/omedeiro/hTron-behavioral-model/enab.txt", t_enab, i_enab)

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
