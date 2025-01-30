from nmem.analysis.analysis import import_directory, get_bit_error_rate
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    dict_list = import_directory("data2")
    delay_list = []
    bit_error_rate_list = []
    for data_dict in dict_list:
        delay = data_dict.get("delay").flatten()
        bit_error_rate = get_bit_error_rate(data_dict)

        delay_list.append(delay)
        bit_error_rate_list.append(bit_error_rate)

        print(f"delay: {delay}, bit_error_rate: {bit_error_rate}")
    fidelity = 1 - np.array(bit_error_rate_list)
    fig, ax = plt.subplots()
    ax.plot(delay_list, fidelity, marker="o")
    ax.plot(delay_list[10], fidelity[10], marker="d", color="red")
    ax.plot(delay_list[9], fidelity[9], marker="s", color="red")
    ax.set_xscale("log")

    ax.set_xlabel("Memory Retention Time (s)")
    ax.set_ylabel("Fidelity")
