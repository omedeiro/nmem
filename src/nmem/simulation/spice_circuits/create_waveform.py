import numpy as np
import matplotlib.pyplot as plt


import numpy as np

def save_pwl_file(filename, time, current):
    with open(filename, "w") as f:
        for t, i in zip(time, current):
            f.write(f"{t:.9e} {i:.9e}\n")


def generate_memory_protocol_sequence(
    num_slots=50,
    cycle_time=20e-9,
    pulse_sigma=1e-9,
    write_amplitude=100e-6,
    read_amplitude=100e-6,
    enab_amplitude=50e-6,
    dt=0.1e-9,
    read_fraction=0.3,
    enab_fraction=0.5,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)

    # # Select random op per slot
    # ops = np.random.choice(["write_1", "write_0", "read", "idle"], size=num_slots)

    # # Select random enab on/off pattern
    # enab_on = np.random.rand(num_slots) < enab_fraction


    patterns = [
        ["write_0", "read", "write_0", "read"],
        ["write_0", "read", "write_1", "read"],
        ["write_1", "read", "write_0", "read"],
        ["write_1", "read", "write_1", "read"], 
    ]

    # Interleave each write pair with a read
    ops = []
    for pair in patterns:
        ops.extend(pair)

    num_slots = len(ops)
    enab_on = np.ones(num_slots, dtype=bool)  # Enable is ON for all ops


    t_chan = []
    i_chan = []
    t_enab = []
    i_enab = []

    for i, op in enumerate(ops):
        t_center = i * cycle_time + cycle_time / 2

        # Determine channel waveform
        if op == "write_1":
            amp = write_amplitude
        elif op == "write_0":
            amp = -write_amplitude
        elif op == "read":
            amp = read_amplitude
        else:
            amp = 0

        if amp != 0:
            t_vec = np.arange(t_center - 4*pulse_sigma, t_center + 4*pulse_sigma, dt)
            i_vec = amp * np.exp(-0.5 * ((t_vec - t_center) / pulse_sigma)**2)
            t_chan.extend(t_vec)
            i_chan.extend(i_vec)

        # Determine enable line
        if enab_on[i] and op in ["write_1", "write_0", "read"]:
            t_vec = np.arange(t_center - 4*pulse_sigma, t_center + 4*pulse_sigma, dt)
            i_vec = enab_amplitude * np.exp(-0.5 * ((t_vec - t_center) / pulse_sigma)**2)
            t_enab.extend(t_vec)
            i_enab.extend(i_vec)

    return np.array(t_chan), np.array(i_chan), np.array(t_enab), np.array(i_enab), ops, enab_on



# Generate waveforms
t_chan, i_chan, t_enab, i_enab, ops, enab_on = generate_memory_protocol_sequence(
    num_slots=40,
    cycle_time=100e-9,
    pulse_sigma=10e-9,
    write_amplitude=100e-6,
    read_amplitude=650e-6,
    enab_amplitude=450e-6,
    dt=0.1e-9,
    read_fraction=0.3,
    enab_fraction=0.5,
    seed=42
)

# Save to files for LTspice
save_pwl_file("/home/omedeiro/hTron-behavioral-model/chan.txt", t_chan, i_chan)
save_pwl_file("/home/omedeiro/hTron-behavioral-model/enab.txt", t_enab, i_enab)

# Print out operation schedule (first 10)
print("Slot | Operation | Wordline Active")
for i in range(10):
    print(f"{i:>4} | {ops[i]:>9} | {'ON' if enab_on[i] else 'OFF'}")

# Optional: plot waveforms for visual confirmation
plt.figure(figsize=(10, 4))
plt.plot(t_chan * 1e9, i_chan * 1e6, label="I_chan (data)", color="tab:blue")
plt.plot(t_enab * 1e9, i_enab * 1e6, label="I_enab (word-line)", color="tab:orange")
plt.xlabel("Time (ns)")
plt.ylabel("Current (µA)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()