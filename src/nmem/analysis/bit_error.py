import numpy as np


def calculate_bit_error_rate(data_dict: dict) -> np.ndarray:
    num_meas = data_dict.get("num_meas")[0][0]
    w1r0 = data_dict.get("write_1_read_0")[0].flatten() / num_meas
    w0r1 = data_dict.get("write_0_read_1")[0].flatten() / num_meas
    ber = (w1r0 + w0r1) / 2
    return ber

def calculate_ber_errorbar(ber, N=None):
    """
    Calculate the standard error for bit error rate (BER).
    If N is not provided, use len(ber) if ber is an array, else 1.
    """
    if N is None:
        N = len(ber) if hasattr(ber, "__len__") and not isinstance(ber, float) else 1
    return np.sqrt(ber * (1 - ber) / N)


def get_total_switches_norm(data_dict: dict) -> np.ndarray:
    num_meas = data_dict.get("num_meas")[0][0]
    w0r1 = data_dict.get("write_0_read_1").flatten()
    w1r0 = num_meas - data_dict.get("write_1_read_0").flatten()
    total_switches_norm = (w0r1 + w1r0) / (num_meas * 2)
    return total_switches_norm

def get_bit_error_rate(data_dict: dict) -> np.ndarray:
    return data_dict.get("bit_error_rate").flatten()

def get_bit_error_rate_args(bit_error_rate: np.ndarray) -> list:
    nominal_args = np.argwhere(bit_error_rate < 0.45)
    inverting_args = np.argwhere(bit_error_rate > 0.55)

    if len(inverting_args) > 0:
        inverting_arg1 = inverting_args[0][0]
        inverting_arg2 = inverting_args[-1][0]
    else:
        inverting_arg1 = np.nan
        inverting_arg2 = np.nan

    if len(nominal_args) > 0:
        nominal_arg1 = nominal_args[0][0]
        nominal_arg2 = nominal_args[-1][0]
    else:
        nominal_arg1 = np.nan
        nominal_arg2 = np.nan

    return nominal_arg1, nominal_arg2, inverting_arg1, inverting_arg2


def get_operating_points(data_dict: dict) -> np.ndarray:
    berargs = get_bit_error_rate_args(get_bit_error_rate(data_dict))
    nominal_operating_point = np.mean([berargs[0], berargs[1]])
    inverting_operating_point = np.mean([berargs[2], berargs[3]])
    return nominal_operating_point, inverting_operating_point
