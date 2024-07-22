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


def calculate_left_branch_limits(alpha, write_current, persistent_current):
    return [
        calculate_left_branch_current(alpha, write_current, persistent_current),
        calculate_left_branch_current(alpha, write_current, -persistent_current),
    ]


def calculate_right_branch_limits(alpha, write_current, persistent_current):
    return [
        calculate_right_branch_current(alpha, write_current, -persistent_current),
        calculate_right_branch_current(alpha, write_current, persistent_current),
    ]


def calculate_read_limits_left(alpha, left_critical_currents, persistent_current):
    return [
        (left_critical_currents - persistent_current) / alpha,
        (left_critical_currents + persistent_current) / alpha,
    ]


def calculate_read_limits_right(alpha, right_critical_current, persistent_current):
    return [
        (right_critical_current - persistent_current) / (1 - alpha),
        (right_critical_current + persistent_current) / (1 - alpha),
    ]


def calculate_0_current(
    left_critical_current: np.ndarray,
    right_critical_current: np.ndarray,
    alpha: float,
    persistent_current: np.ndarray,
    iretrap: float,
) -> np.ndarray:
    """Calculate the current required to switch the device to state 0.

    Parameters
    ----------
    left_critical_current : float
        The critical current of the left branch.
    right_critical_current : float
        The critical current of the right branch."""

    read_current_left = (left_critical_current - persistent_current) / alpha
    read_current_right = right_critical_current + left_critical_current * iretrap

    return np.maximum(read_current_left, read_current_right)


def calculate_1_current(
    left_critical_current: np.ndarray,
    right_critical_current: np.ndarray,
    alpha: float,
    persistent_current: np.ndarray,
    iretrap: float,
) -> np.ndarray:
    """Calculate the current required to switch the device to state 1.

    Parameters
    ----------
    left_critical_current : float
        The critical current of the left branch.
    right_critical_current : float
        The critical current of the right branch."""

    read_current_left = left_critical_current + right_critical_current * iretrap
    read_current_right = (right_critical_current - persistent_current) / (1 - alpha)

    return np.maximum(read_current_left, read_current_right)


def calculate_alpha(ll, lr):
    """ll < lr"""
    return lr / (ll + lr)


def calculate_persistent_current(
    left_critical_currents: np.ndarray,
    write_currents: np.ndarray,
    alpha: float,
    max_left_critical_current: float,
    max_right_critical_current: float,
    iretrap: float,
    width_left: float,
    width_right: float,
) -> np.ndarray:
    """Calculate the persistent current in the loop for a given set of parameters.

    Parameters
    ----------
    left_critical_currents : np.ndarray
        The critical current of the left branch.
    write_currents : np.ndarray
        The write current of the device.
    alpha : float
        The ratio of the left branch inductance to the total inductance.
    max_left_critical_current : float
        The maximum critical current of the left branch. At minimum temperature
    max_right_critical_current : float
        The maximum critical current of the right branch. At minimum temperature
    iretrap : float
        The retrapping current of the device.
    width_left : float
        The width of the left branch.
    width_right : float
        The width of the right branch.

    Returns
    -------
    np.ndarray
    The persistent current in the loop.
    """
    # The right critical current is the left critical current scaled
    # by the ratio of the switching currents.
    ic_ratio = max_left_critical_current / max_right_critical_current
    right_critical_currents = left_critical_currents / ic_ratio

    # Assuming no persistent current in the loop
    persistent_current = np.zeros_like(left_critical_currents)

    # Current is inductively split between the left and right branches
    left_branch_current = write_currents * (1 - alpha)
    right_branch_current = write_currents * alpha

    # If the left branch current is greater than the
    # left critical current, the branch switches.
    left_switch = left_branch_current > left_critical_currents

    # Where the left branch switched the persistent current is set to the write current.
    persistent_current = np.where(left_switch, write_currents, persistent_current)

    # Therefore, the right branch must carry the full write current
    right_branch_current = np.where(left_switch, write_currents, right_branch_current)

    # If the right branch also switches
    right_switch = right_branch_current > right_critical_currents

    # LEFT_SQUARES = width_left
    # RIGHT_SQUARES = width_right / width_left
    # SHEET_RESISTANCE = 1

    # left_hotspot_resistance = SHEET_RESISTANCE * LEFT_SQUARES
    # right_hotspot_resistance = SHEET_RESISTANCE * RIGHT_SQUARES

    # hotspot_resistance_ratio = left_hotspot_resistance / (
    #     left_hotspot_resistance + right_hotspot_resistance
    # )

    # # The current is then resistively split between the left and right branches
    # left_branch_current = np.where(
    #     right_switch, write_currents * hotspot_resistance_ratio, left_branch_current
    # )
    # right_branch_current = np.where(
    #     right_switch,
    #     write_currents * (1 - hotspot_resistance_ratio),
    #     right_branch_current,
    # )

    # If the resistive right branch current is less than the right retrapping
    right_retrap = right_branch_current < right_critical_currents * iretrap

    persistent_current = np.where(
        right_switch,
        left_critical_currents - right_critical_currents * iretrap,
        persistent_current,
    )
    persistent_current = np.abs(persistent_current)

    left_persistent_switch = persistent_current > max_left_critical_current
    persistent_current = np.where(
        left_persistent_switch, max_left_critical_current, persistent_current
    )

    regions = {
        "left_switch": left_switch,
        "right_switch": right_switch,
        "right_retrap": right_retrap,
        "left_persistent_switch": left_persistent_switch,
    }

    # persistent_current = np.where(persistent_current>ichl*1e6, ichl*1e6, persistent_current)
    return persistent_current, regions


def calculate_read_currents(
    left_critical_currents: np.ndarray,
    write_currents: np.ndarray,
    persistent_currents: np.ndarray,
    alpha: float,
    max_left_critical_current: float,
    max_right_critical_current: float,
    iretrap: float,
):
    ic_ratio = max_left_critical_current / max_right_critical_current

    [xx, yy] = np.meshgrid(left_critical_currents, write_currents)
    right_critical_currents = left_critical_currents / ic_ratio
    # read_currents = np.zeros_like(xx)

    zero_switching_current = calculate_0_current(
        xx,
        yy,
        alpha,
        persistent_currents,
        iretrap,
    )
    one_switching_current = calculate_1_current(
        xx,
        yy,
        alpha,
        persistent_currents,
        iretrap,
    )
    print(f"zero_switching_current: {zero_switching_current.shape}")
    print(f"one_switching_current: {one_switching_current.shape}")
    read_currents = (zero_switching_current + one_switching_current) / 2
    print(f"read_currents: {read_currents.shape}")
    # Negative read currents are not possible
    mask_negative = read_currents < 0
    # read_currents[mask_negative] = 0

    # # # Read current NA when persistent current is zero
    mask_zero_persistent = persistent_currents == 0
    # read_currents[mask_zero_persistent] = 0

    # # # Read current cannot be less than the write current
    mask_less_than_write = np.abs(read_currents) < write_currents
    # read_currents[mask_less_than_write] = 0

    # # Read current cannot be greater than the right critical current
    mask_greater_than_right = read_currents > right_critical_currents
    # read_currents[mask_greater_than_right] = 0

    mask_list = [mask_negative, mask_zero_persistent, mask_less_than_write]
    # mask_list = []
    return read_currents, mask_list


def calculate_critical_current_bounds(persistent_current, read_current, alpha):
    return read_current * alpha * np.ones((2, 1)) + [
        -persistent_current,
        persistent_current,
    ]
