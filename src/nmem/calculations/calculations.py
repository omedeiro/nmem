import numpy as np


def htron_critical_current(
    slope: float, intercept: float, heater_current: float
) -> float:
    channel_current = heater_current * slope + intercept
    return channel_current


def calculate_right_branch_inductance(alpha: float, ll: float) -> float:
    return alpha * ll / (1 - alpha)


def calculate_left_branch_inductance(alpha, lr):
    return (1 - alpha) * lr / alpha


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


def calculate_0_state_currents(
    left_critical_current: np.ndarray,
    right_critical_current: np.ndarray,
    persistent_current: np.ndarray,
    alpha: float,
):
    """Calculate the current required to switch the device to state 0.

    Parameters
    ----------
    left_critical_current : np.ndarray
        The critical current of the left branch.
    right_critical_current : np.ndarray
        The critical current of the right branch.

    Returns
    -------
    The bias current limits assuming the device is in state 0."""

    # The 0 state is defined as a negative persistent current
    # persistent_current = np.where(
    #     persistent_current > 0, -persistent_current, persistent_current
    # )

    bias_current_to_switch_left = (
        np.abs(left_critical_current - persistent_current) / alpha
    )
    bias_current_to_switch_right = (right_critical_current + persistent_current) / (
        1 - alpha
    )

    return bias_current_to_switch_left, bias_current_to_switch_right


def calculate_1_state_currents(
    left_critical_current: np.ndarray,
    right_critical_current: np.ndarray,
    persistent_current: np.ndarray,
    alpha: float,
):
    """Calculate the current required to switch the device to state 1.

    Parameters
    ----------
    left_critical_current : float
        The critical current of the left branch.
    right_critical_current : float
        The critical current of the right branch.
    persistent_current : float
        The persistent current in the loop.
    iretrap : float
        The retrapping current of the device.
    alpha : float
        The ratio of the right branch inductance to the total inductance. LR/(LL+LR)

    Returns
    -------
    The bias current limits assuming the device is in state 1."""

    # The 1 state is defined as a positive persistent current
    # persistent_current = np.where(
    #     persistent_current < 0, -persistent_current, persistent_current
    # )

    bias_current_to_switch_left = (left_critical_current + persistent_current) / alpha
    bias_current_to_switch_right = np.abs(
        right_critical_current - persistent_current
    ) / (1 - alpha)

    return bias_current_to_switch_left, bias_current_to_switch_right


def calculate_alpha(ll, lr):
    """ll < lr"""
    return lr / (ll + lr)


def calculate_persistent_current(
    data_dict: dict,
) -> np.ndarray:

    left_critical_currents_mesh = data_dict["left_critical_currents_mesh"]
    right_critical_currents_mesh = data_dict["right_critical_currents_mesh"]
    write_currents_mesh = data_dict["write_currents_mesh"]
    alpha = data_dict["alpha"]
    max_left_critical_current = data_dict["max_left_critical_current"]
    iretrap = data_dict["iretrap"]

    # Assuming no persistent current in the loop
    persistent_current = np.zeros_like(left_critical_currents_mesh)

    # Current is inductively split between the left and right branches
    left_branch_current = calculate_left_branch_current(
        alpha, write_currents_mesh, persistent_current
    )
    right_branch_current = calculate_right_branch_current(
        alpha, write_currents_mesh, persistent_current
    )

    # CONDITION A
    # -----------
    # If the left branch current is greater than the
    # left critical current, the branch switches.
    # Where the left branch switched the persistent current is set to the write current.
    # Therefore, the right branch must carry the full write current and not switch
    condition_a = (left_critical_currents_mesh < left_branch_current) & (
        write_currents_mesh < right_critical_currents_mesh
    )
    persistent_current = np.where(condition_a, write_currents_mesh, persistent_current)

    # CONDITION B
    # -----------
    # If the left branch switches and the redirected write current is enough
    # to switch the right branch, then both branches are switched.
    #
    condition_b = (left_critical_currents_mesh < left_branch_current) & (
        write_currents_mesh > right_critical_currents_mesh
    )
    persistent_current = np.where(
        condition_b,
        left_critical_currents_mesh * iretrap,
        persistent_current,
    )

    # Limit the persistent current by the maximum critical current of the left branch
    # with no enable current. This is the maximum persistent current.
    left_persistent_switch = np.abs(persistent_current) > max_left_critical_current
    persistent_current = np.where(
        left_persistent_switch,
        max_left_critical_current,
        persistent_current,
    )

    regions = {
        # "right_switch": right_switch,
        # "right_retrap": right_retrap_left_current,
        "left_persistent_switch": left_persistent_switch,
    }

    return persistent_current, regions


def calculate_read_currents(data_dict: dict):
    left_critical_currents_mesh = data_dict["left_critical_currents_mesh"]
    right_critical_currents_mesh = data_dict["right_critical_currents_mesh"]
    write_currents = data_dict["write_currents_mesh"]
    alpha = data_dict["alpha"]
    max_left_critical_current = data_dict["max_left_critical_current"]
    max_right_critical_current = data_dict["max_right_critical_current"]
    persistent_currents = data_dict["persistent_currents"]


    zero_switching_current_left, zero_switching_current_right = calculate_0_state_currents(
        left_critical_currents_mesh,
        write_currents_mesh,
        persistent_currents,
        alpha,
    )
    one_switching_current_left, one_switching_current_right = calculate_1_state_currents(
        left_critical_currents_mesh,
        write_currents_mesh,
        persistent_currents,
        alpha,
    )

    read_currents = (zero_switching_current_left + one_switching_current_left) / 2


    return read_currents


def calculate_critical_current_bounds(persistent_current, read_current, alpha):
    return read_current * alpha * np.ones((2, 1)) + [
        -persistent_current,
        persistent_current,
    ]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    left_critical_current = 22
    right_critical_current = 87
    persistent_current = 60
    alpha = 0.63
    iretrap = 0.9
    left_critical_currents_mesh, write_currents_mesh = np.meshgrid(
        np.linspace(100, 0, 100), np.linspace(50, 250, 100)
    )
    SCALE = 2
    data_dict = {
        "left_critical_currents_mesh": left_critical_currents_mesh * SCALE,
        "right_critical_currents_mesh": left_critical_currents_mesh * 3 * SCALE,
        "write_currents_mesh": write_currents_mesh,
        "alpha": alpha,
        "max_left_critical_current": 100 * SCALE,
        "max_right_critical_current": 300 * SCALE,
        "iretrap": iretrap,
    }

    persistent_current, regions = calculate_persistent_current(data_dict)
    data_dict["persistent_currents"] = persistent_current
    read_currents = calculate_read_currents(data_dict)

    plt.pcolormesh(left_critical_currents_mesh, write_currents_mesh, read_currents)
    plt.gca().invert_xaxis()
    plt.colorbar()
