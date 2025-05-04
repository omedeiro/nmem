import numpy as np


def generate_memory_protocol_sequence(
    cycle_time,
    pulse_sigma,
    hold_width_write,
    hold_width_read,
    hold_width_clear,
    write_amplitude,
    read_amplitude,
    enab_write_amplitude,
    enab_read_amplitude,
    clear_amplitude,
    dt,
    seed=None,
    phase_offset=0,
):
    if seed is not None:
        np.random.seed(seed)



    patterns = [
        ["null", "null", "null", "write_0", "read", "null", "null", "null", "write_1", "read"],
        ["null", "null", "write_0", "null", "read", "null", "null", "write_1", "null", "read"],
        ["null", "write_0", "null", "null", "read", "null", "write_1", "null", "null", "read"],
        ["null", "write_1", "enab", "read", "read", "write_0", "enab", "read", "read", "null"],
        ["null", "write_0", "enab", "read", "read", "write_1", "enab", "read", "read", "null"],
    ]

    ops = []
    for pair in patterns:
        for op in pair:
            ops.append(op)

    t_chan, i_chan = [], []
    t_enab, i_enab = [], []

    enab_on = np.ones(len(ops), dtype=bool)
    enab_on[33:34] = False  # Disable word-line select for read operations
    enab_on[37:38] = False  # Disable word-line select for read operations
    enab_on[43:44] = False  # Disable word-line select for read operations
    enab_on[47:48] = False  # Disable word-line select for read operations
    hold_width_enab = 20e-9
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
        elif op == "null":
            amp = 0
            t_vec, i_vec = flat_top_gaussian(
                t_center, pulse_sigma, hold_width_enab, amp, dt
            )
            t_chan.extend(t_vec)
            i_chan.extend(i_vec)

        # --- I_enab: word-line select ---
        if op in ["write_1", "write_0", "enab"]:
            amp = enab_write_amplitude
            hold = hold_width_enab
        elif op == "read":
            amp = enab_read_amplitude
            hold = hold_width_enab
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


def save_pwl_file(filename, t, i):
    with open(filename, "w") as f:
        for time, current in zip(t, i):
            f.write(f"{time:.12e}\t{current:.12e}\n")
