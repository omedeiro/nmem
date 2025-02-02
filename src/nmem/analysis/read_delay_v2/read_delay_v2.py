import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.analysis import get_bit_error_rate, import_directory

if __name__ == "__main__":
    dict_list = import_directory("data3")
    # dict_list.extend(import_directory("data3"))
    delay_list = []
    bit_error_rate_list = []
    for data_dict in dict_list:
        delay = data_dict.get("delay").flatten()[0]*1e-3
        bit_error_rate = get_bit_error_rate(data_dict)

        delay_list.append(delay)
        bit_error_rate_list.append(bit_error_rate)

        print(f"delay: {delay}, bit_error_rate: {bit_error_rate}, num_measurements: {data_dict['num_meas'].flatten()[0]:.0g}")
    fidelity = 1 - np.array(bit_error_rate_list)
    
    
    
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sort_index = np.argsort(delay_list)
    delay_list = np.array(delay_list)[sort_index]
    bit_error_rate_list = np.array(bit_error_rate_list)[sort_index]
    ax.plot(delay_list, bit_error_rate_list, marker="o", color="black")
    # ax.plot(delay_list[10], fidelity[10], marker="d", color="red")
    # ax.plot(delay_list[9], fidelity[9], marker="s", color="red")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("Bit Error Rate")

    ax.set_xlabel("Memory Retention Time (s)")
    # ax.set_ylabel("Fidelity")
    ax.set_ylim(1e-4, 1)
    ax.set_xbound(lower=1e-6)
    
    plt.savefig("read_delay_v2.pdf", bbox_inches="tight")
    plt.show()
