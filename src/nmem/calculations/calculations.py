import numpy as np


def htron_critical_current(
    slope: float, intercept: float, heater_current: float
) -> float:
    channel_current = heater_current * slope + intercept
    return channel_current


def calculate_right_branch_inductance(
    alpha: float, left_branch_inductance: float
) -> float:
    return left_branch_inductance * alpha / (1 - alpha)


def calculate_left_branch_inductance(
    alpha: float, right_branch_inductance: float
) -> float:
    return right_branch_inductance * (1 - alpha) / alpha


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
):
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
):
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
    left_critical_currents: np.ndarray,
    right_critical_currents: np.ndarray,
    persistent_currents: np.ndarray,
    alpha: float,
    iretrap: float,
):
    """Calculate the current required to switch the device to state 0.

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

    zero_state_current = np.maximum(
        (left_critical_currents - persistent_currents) / alpha,
        right_critical_currents + left_critical_currents * iretrap,
    )

    return zero_state_current


def calculate_one_state_current(
    left_critical_currents: np.ndarray,
    right_critical_currents: np.ndarray,
    persistent_currents: np.ndarray,
    alpha: float,
    iretrap: float,
):
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

    one_state_current = np.maximum(
        (right_critical_currents - persistent_currents) / (1 - alpha),
        left_critical_currents + right_critical_currents * iretrap,
    )
    return one_state_current


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

    # Limit persistent current to positive values.
    persistent_current = np.where(persistent_current < 0, 0, persistent_current)

    regions = {
        # "right_switch": right_switch,
        # "right_retrap": right_retrap_left_current,
        "left_persistent_switch": left_persistent_switch,
    }

    return persistent_current, regions


def calculate_read_currents(data_dict: dict):
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
    left_critical_currents_mesh = data_dict["left_critical_currents_mesh"]
    right_critical_currents_mesh = data_dict["right_critical_currents_mesh"]
    persistent_currents = data_dict["persistent_currents"]
    alpha = data_dict["alpha"]
    iretrap = data_dict["iretrap"]

    zero_state_current = calculate_zero_state_current(
        left_critical_currents_mesh,
        right_critical_currents_mesh,
        persistent_currents,
        alpha,
        iretrap,
    )
    one_state_current = calculate_one_state_current(
        left_critical_currents_mesh,
        right_critical_currents_mesh,
        persistent_currents,
        alpha,
        iretrap,
    )

    read_currents = (zero_state_current + one_state_current) / 2
    read_margins = np.abs(zero_state_current - one_state_current) / 2
    return read_currents, read_margins


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    left_critical_current = 22
    right_critical_current = 87
    persistent_current = 60
    alpha = 0.63
    iretrap = 0.1
    left_critical_currents_mesh, write_currents_mesh = np.meshgrid(
        np.linspace(0, 100, 100), np.linspace(50, 250, 100)
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

    READ_SET = 350
    persistent_currents, regions = calculate_persistent_current(data_dict)
    data_dict["persistent_currents"] = persistent_currents
    read_currents, read_margin = calculate_read_currents(data_dict)
    read_currents = np.where(read_currents < write_currents_mesh, 0, read_currents)
    read_currents = np.where(persistent_currents == 0, 0, read_currents)

    # read_currents = np.where(READ_SET > read_currents+read_margin, 0, read_currents)
    # read_currents = np.where(READ_SET < read_currents-read_margin, 0, read_currents)

    plt.pcolormesh(left_critical_currents_mesh, write_currents_mesh, read_currents)
    plt.gca().invert_xaxis()
    plt.colorbar()
