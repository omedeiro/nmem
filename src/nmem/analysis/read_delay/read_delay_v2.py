from nmem.analysis.analysis import import_directory, get_bit_error_rate
import matplotlib.pyplot as plt

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

    fig, ax = plt.subplots()
    ax.plot(delay_list, bit_error_rate_list, marker="o")
