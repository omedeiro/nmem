from typing import Tuple

import numpy as np


def htron_critical_current(
    heater_current: float,
    slope: float,
    intercept: float,
) -> float:
    """Calculate the critical current of the device.

    Parameters
    ----------
    heater_current : float
        The current through the heater I_H.
    slope : float
        The slope of linear enable response I_C(I_H) .
    intercept : float
        The y-intercept of the linear enable response in microamps.

    Returns
    -------
    critical_current : float
        The critical current of the device in microamps.
    """
    channel_current = heater_current * slope + intercept
    return channel_current


def htron_heater_current(
    channel_current: float,
    slope: float,
    intercept: float,
) -> float:
    heater_current = (channel_current - intercept) / slope
    return heater_current


def calculate_critical_current(
    heater_current: float,
    cell_dict: dict,
) -> float:
    """Calculate the critical current of the device.

    Parameters
    ----------
    heater_current : float
        The current through the heater I_H in microamps.
    cell_dict : dict
        A dictionary containing the following
        - slope : float
            The slope of linear enable response I_C(I_H) .
            - intercept : float
                The y-intercept of the linear enable response in microamps.
            - max_critical_current : float
                The maximum critical current of the device in microamps.

    Returns
    -------
    critical_current : float
        The critical current of the device in microamps."""
    slope = cell_dict["slope"]
    intercept = cell_dict["intercept"]
    critical_current = htron_critical_current(heater_current, slope, intercept)
    if critical_current > (cell_dict.get("max_critical_current", np.inf) * 1e6):
        return cell_dict["max_critical_current"] * 1e6
    else:
        return critical_current


def calculate_heater_current(
    channel_current: float,
    cell_dict: dict,
) -> float:
    slope = cell_dict["slope"]
    intercept = cell_dict["intercept"]
    return htron_heater_current(channel_current, slope, intercept)


def calculate_heater_power(heater_current: float, heater_resistance: float) -> float:
    return heater_current**2 * heater_resistance


def calculate_right_branch_inductance(
    alpha: float, left_branch_inductance: float
) -> float:
    return left_branch_inductance * alpha / (1 - alpha)


def calculate_left_branch_inductance(
    alpha: float, right_branch_inductance: float
) -> float:
    return right_branch_inductance * (1 - alpha) / alpha


def calculate_alpha(
    left_branch_inductance: float, right_branch_inductance: float
) -> float:
    """Calculate the ratio of the right branch inductance to the total inductance. LR/(LL+LR)

    Parameters
    ----------
    left_branch_inductance : float
        The inductance of the left branch.
    right_branch_inductance : float
        The inductance of the right branch.

    Returns
    -------
    alpha : float
        The ratio of the right branch inductance to the total inductance. LR/(LL+LR)
    """

    return right_branch_inductance / (left_branch_inductance + right_branch_inductance)


def calculate_right_branch_current(
    alpha: float,
    channel_current: float,
    persistent_current: float,
) -> float:
    """State 1 is defined as a positive persistent current, clockwise
    State 0 is defined as a negative persistent current, counter-clockwise
    Right branch sums current during read.
    """
    return channel_current * (1 - alpha) + persistent_current


def calculate_left_branch_current(
    alpha: float, channel_current: float, persistent_current: float
) -> float:
    """State 1 is defined as a positive persistent current, clockwise
    State 0 is defined as a negative persistent current, counter-clockwise
    Left branch negates current during read.
    """
    return channel_current * alpha - persistent_current


def calculate_channel_current_zero(
    left_critical_current: np.ndarray,
    right_critical_current: np.ndarray,
    persistent_current: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the channel current limits for the left and right branches when in state 0.

    Parameters
    ----------
    left_critical_current : float
        The critical current of the left branch.
    right_critical_current : float
        The critical current of the right branch.
    persistent_current : float
        The persistent current in the loop.
    alpha : float
        The ratio of the right branch inductance to the total inductance. LR/(LL+LR)

    Returns
    -------
    left_limit : float
        The current limit of the left branch.
    right_limit : float
        The current limit of the right branch.
    """
    left_limit = (left_critical_current - persistent_current) / alpha
    right_limit = (right_critical_current + persistent_current) / (1 - alpha)

    return left_limit, right_limit


def calculate_channel_current_one(
    left_critical_current: np.ndarray,
    right_critical_current: np.ndarray,
    persistent_current: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the channel current limits for the left and right branches when in state 1.

    Parameters
    ----------
    left_critical_current : float
        The critical current of the left branch.
    right_critical_current : float
        The critical current of the right branch.
    persistent_current : float
        The persistent current in the loop.
    alpha : float
        The ratio of the right branch inductance to the total inductance. LR/(LL+LR)

    Returns
    -------
    left_limit : float
        The current limit of the left branch.
    right_limit : float
        The current limit of the right branch.
    """
    left_limit = (left_critical_current + persistent_current) / alpha
    right_limit = (right_critical_current - persistent_current) / (1 - alpha)

    return left_limit, right_limit


def calculate_zero_state_current(
    channel_critical_currents: np.ndarray,
    width_ratio: float,
    persistent_currents: np.ndarray,
    alpha: float,
    iretrap_enable: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the current required to switch the device when in state 0

    Parameters
    ----------
    left_critical_current : np.ndarray
        The critical current of the left branch.
    right_critical_current : np.ndarray
        The critical current of the right branch.
    persistent_current : np.ndarray
        The persistent current in the loop.
    iretrap : float
        The retrapping current of the device.

    Returns
    -------
    zero_state_current : np.ndarray
        The current that causes the channel to switch in state 0.
    """

    right_critical_currents = channel_critical_currents / (
        1 + iretrap_enable / width_ratio
    )
    left_critical_currents = right_critical_currents / width_ratio
    left_retrapping_currents = left_critical_currents * iretrap_enable
    right_retrapping_currents = right_critical_currents * iretrap_enable
    current_to_switch_left = (
        left_critical_currents - persistent_currents
    ) / alpha + OFFSET_C
    current_to_switch_right = (
        left_critical_currents + right_retrapping_currents + OFFSET_B
    )

    fa = right_critical_currents + left_retrapping_currents
    fb = left_critical_currents + right_retrapping_currents
    fc = (left_critical_currents - persistent_currents) / alpha
    fd = (right_critical_currents - persistent_currents) / (1 - alpha)
    upper_limit = np.minimum(fa, fd)
    lower_limit = np.minimum(fb, fc)
    zero_state_current = np.where(
        persistent_currents > 0,
        lower_limit,
        upper_limit,
    )
    zero_state_current_index = np.where(
        current_to_switch_left > current_to_switch_right, 0, 1
    )

    # State currents are positive
    zero_state_current = np.where(zero_state_current < 0, 0, zero_state_current)

    return zero_state_current, zero_state_current_index


def calculate_one_state_current(
    channel_critical_currents: np.ndarray,
    width_ratio: float,
    persistent_currents: np.ndarray,
    alpha: float,
    iretrap_enable: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the current required to switch the device to state 1.

    Parameters
    ----------
    left_critical_current : np.ndarray
        The critical current of the left branch.
    right_critical_current : np.ndarray
        The critical current of the right branch.
    persistent_current : np.ndarray
        The persistent current in the loop.
    iretrap : float
        The retrapping current of the device.

    Returns
    -------
    one_state_current : np.ndarray
        The current that causes the channel to switch in state 1."""

    right_critical_currents = channel_critical_currents / (
        1 + iretrap_enable / width_ratio
    )
    left_critical_currents = right_critical_currents / width_ratio
    right_retrapping_currents = right_critical_currents * iretrap_enable
    left_retrapping_currents = left_critical_currents * iretrap_enable

    fa = right_critical_currents + left_retrapping_currents
    fb = left_critical_currents + right_retrapping_currents
    fc = (left_critical_currents - persistent_currents) / alpha
    fd = (right_critical_currents - persistent_currents) / (1 - alpha)
    upper_limit = np.minimum(fa, fd)
    lower_limit = np.maximum(fb, fc)

    print(f"persistent_currents: {persistent_currents}")
    one_state_currents = np.where(
        persistent_currents > 0,
        upper_limit,
        lower_limit,
    )

    one_state_currents_index = np.where(lower_limit < upper_limit, 0, 1)

    # State currents are positive
    one_state_currents = np.where(one_state_currents < 0, 0, one_state_currents)

    return one_state_currents, one_state_currents_index


def calculate_persistent_current(
    data_dict: dict,
) -> Tuple[np.ndarray, dict]:
    left_critical_currents_mesh = data_dict["left_critical_currents_mesh"]
    right_critical_currents_mesh = data_dict["right_critical_currents_mesh"]
    write_currents_mesh = data_dict["write_currents_mesh"]
    alpha = data_dict["alpha"]
    iretrap_enable = data_dict["iretrap_enable"]
    # Assuming no persistent current in the loop
    persistent_current = np.zeros_like(left_critical_currents_mesh)

    # Current is inductively split between the left and right branches
    left_branch_current = calculate_left_branch_current(
        alpha, write_currents_mesh, persistent_current
    )

    # CONDITION A - WRITE STATE
    # -----------
    # If the left branch current is greater than the
    # left critical current, the branch switches.
    # Where the left branch switched the persistent current is set to the write current.
    # Therefore, the right branch must carry the full write current and not switch
    condition_a = (left_branch_current > left_critical_currents_mesh) & (
        write_currents_mesh < right_critical_currents_mesh
    )
    persistent_current = np.where(condition_a, write_currents_mesh, persistent_current)

    # CONDITION B - INVERTING STATE
    # -----------
    # If the left branch switches and the redirected write current is enough
    # to switch the right branch, then the new left branch current is the
    # write current minus the right retrapping current.
    condition_b = (left_branch_current > left_critical_currents_mesh) & (
        write_currents_mesh > right_critical_currents_mesh
    )
    new_left_branch_current = write_currents_mesh - (
        right_critical_currents_mesh * iretrap_enable
    )

    # CONDITION C - WRITE INVERTING STATE
    # -----------
    # If CONDITION B is true and the new left branch current is less than the left critical current, the state is inverted
    condition_c = condition_b & (new_left_branch_current < left_critical_currents_mesh)
    persistent_current = np.where(
        condition_c,
        new_left_branch_current,
        persistent_current,
    )

    # CONDITION D
    # -----------
    # If CONDITION B is true and the new left branch current is greater than the left critical current, there will be an output voltage.
    # Ip is assumed to be the retrapping current
    condition_d = condition_b & (new_left_branch_current > left_critical_currents_mesh)

    persistent_current = np.where(
        condition_d,
        left_critical_currents_mesh,
        persistent_current,
    )

    # Limit the persistent current by the maximum critical current of the left branch
    # with the enable current active. This is the maximum persistent current.
    # persistent_current = np.where(
    #     persistent_current > max_left_critical_current,
    #     0,
    #     persistent_current,
    # )

    # Limit persistent current to positive values.
    # persistent_current = np.where(persistent_current < 0, 0, persistent_current)

    # Regions where the critical current is negative are invalid.
    # Set the persistent current to zero in these regions.
    # persistent_current = np.where(
    #     left_critical_currents_mesh < 0, 0, persistent_current
    # )
    regions = {
        "write_state": condition_a,
        "inverting": condition_c,
        "voltage": condition_d,
    }

    return persistent_current, regions


def calculate_read_currents(data_dict: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the read currents and margins of the device.

    Parameters
    ----------
    data_dict : dict
        A dictionary containing the following
        - left_critical_currents_mesh : np.ndarray
            The critical current of the left branch.
        - right_critical_currents_mesh : np.ndarray
            The critical current of the right branch.
        - persistent_currents : np.ndarray
            The persistent current in the loop.
        - alpha : float
            The ratio of the right branch inductance to the total inductance. LR/(LL+LR)
        - iretrap : float
            The retrapping current of the device.

    Returns
    -------
    read_currents : np.ndarray
        The read currents of the device.
    read_margins : np.ndarray
        The read margins of the device.
    """
    channel_critical_currents_mesh = data_dict["channel_critical_currents_mesh"]
    width_ratio = data_dict["width_ratio"]
    persistent_currents = data_dict["persistent_currents"]
    alpha = data_dict["alpha"]
    iretrap_enable = data_dict["iretrap_enable"]

    state_currents = calculate_state_currents(
        channel_critical_currents_mesh,
        persistent_currents,
        alpha,
        width_ratio,
        iretrap_enable,
    )

    zero_state_currents = state_currents[0]
    one_state_currents = state_currents[1]
    zero_state_currents_inv = state_currents[2]
    one_state_currents_inv = state_currents[3]
    fa = state_currents[4]
    fb = state_currents[5]
    fc = state_currents[6]

    current_dict = {
        "zero_state_currents": zero_state_currents,
        "one_state_currents": one_state_currents,
        "zero_state_currents_inv": zero_state_currents_inv,
        "one_state_currents_inv": one_state_currents_inv,
        "fa": fa,
        "fb": fb,
        "fc": fc,
    }
    return current_dict


def calculate_ideal_read_current(
    zero_state_current: float, one_state_current: float
) -> float:
    return (zero_state_current + one_state_current) / 2


def calculate_ideal_read_margin(
    zero_state_current: float, one_state_current: float
) -> float:
    return np.abs(zero_state_current - one_state_current) / 2


def calculate_ideal_read_margin_signed(
    zero_state_current: float, one_state_current: float
) -> float:
    return (zero_state_current - one_state_current) / 2


def calculate_left_upper_bound(
    persistent_current: float, read_current: float, alpha: float
) -> float:
    return persistent_current + read_current * alpha


def calculate_left_lower_bound(
    persistent_current: float, read_current: float, alpha: float
) -> float:
    return np.max([-persistent_current + read_current * alpha, 0])


def calculate_right_upper_bound(
    persistent_current: float, read_current: float, alpha: float
) -> float:
    return read_current


def calculate_right_lower_bound(
    persistent_current: float, read_current: float, alpha: float
) -> float:
    return np.max([persistent_current + read_current * (1 - alpha), 0])


OFFSET_A = -297.0
OFFSET_B = -240.0
OFFSET_C = 20.0


def calculate_state_currents(
    channel_critical_current: float,
    persistent_current: float,
    alpha: float,
    width_ratio: float,
    iretrap_enable: float,
) -> list:

    right_critical_current = channel_critical_current / (
        1 + (iretrap_enable / width_ratio)
    )
    left_critical_current = right_critical_current / width_ratio
    right_retrapping_current = right_critical_current * iretrap_enable
    left_retrapping_current = left_critical_current * iretrap_enable

    fa = right_critical_current + left_retrapping_current + OFFSET_A
    fb = left_critical_current + right_retrapping_current + OFFSET_B
    fc = (left_critical_current - persistent_current) / alpha + OFFSET_C

    # minichl = fc
    # minichr = fb
    # maxichr = fa

    zero_state_current = np.minimum(fa, 950)
    one_state_current = np.maximum(fb, fc)
    one_state_current_inv = np.minimum(np.maximum(fb, fc), zero_state_current)

    zero_state_current_inv = np.minimum(fb, fc)

    state_currents = [
        zero_state_current,
        one_state_current,
        zero_state_current_inv,
        one_state_current_inv,
        fa,
        fb,
        fc,
    ]
    return state_currents
