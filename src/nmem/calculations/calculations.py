
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


def caluclate_branch_critical_currents(
    critical_current: float, width_left: float, width_right: float
) -> np.ndarray:
    ratio = width_left / (width_left + width_right)
    return critical_current * np.array([ratio, 1 - ratio])


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
    ichl: float,
    ichr: float,
    alpha: float,
    persistent_current: float,
    iretrap: float = 0.5,
) -> float:
    return np.max([(ichl - persistent_current) / alpha, ichl * iretrap + ichr])


def calculate_1_current(
    ichl: float,
    ichr: float,
    alpha: float,
    persistent_current: float,
    iretrap: float = 0.5,
) -> float:
    return np.max([(ichr - persistent_current) / (1 - alpha), ichr * iretrap + ichl])


def calculate_alpha(ll, lr):
    """ll < lr"""
    return lr / (ll + lr)


def calculate_persistent_current(
    left_critical_currents: np.ndarray,
    write_currents: np.ndarray,
    alpha: float,
    ichl: float,
    ichr: float,
    iretrap: float,
):
    # The right critical current is the left critical current scaled
    # by the ratio of the switching currents.
    right_critical_current = left_critical_currents * ichr / ichl

    # Assuming no persistent current in the loop
    persistent_current = np.zeros_like(left_critical_currents)
    left_branch_current = write_currents * alpha
    right_branch_current = write_currents * (1 - alpha)

    left_switch = left_branch_current > left_critical_currents
    # Assuming that all of the excess current is persistent. None shunted to ground.
    persistent_current = np.where(left_switch, write_currents, persistent_current)

    # If the left branch switches the right branch must carry the full write current
    right_branch_current = np.where(left_switch, write_currents, right_branch_current)

    # If the right branch also switches
    right_switch = right_branch_current > right_critical_current
    persistent_current = np.where(
        right_switch, left_critical_currents * iretrap, persistent_current
    )

    return persistent_current


def calculate_read_currents(
    left_critical_currents: np.ndarray,
    write_currents: np.ndarray,
    persistent_currents: np.ndarray,
    alpha: float,
    ichr: float,
    ichl: float,
):
    [xx, yy] = np.meshgrid(left_critical_currents, write_currents)
    right_critical_currents = left_critical_currents * ichr / ichl
    read_currents = np.zeros_like(xx)
    for i in range(len(write_currents)):
        for j in range(len(left_critical_currents)):
            zero_switching_current = calculate_0_current(
                left_critical_currents[j],
                right_critical_currents[j],
                alpha,
                persistent_currents[j, i],
            )
            one_switching_current = calculate_1_current(
                left_critical_currents[j],
                right_critical_currents[j],
                alpha,
                persistent_currents[j, i],
            )

            read_currents[j, i] = one_switching_current

    # Negative read currents are not possible
    mask_negative = read_currents < 0
    read_currents[mask_negative] = 0

    # # Read current NA when persistent current is zero
    mask_zero_persistent = persistent_currents == 0
    read_currents[mask_zero_persistent] = 0

    # # Read current cannot be less than the write current
    mask_less_than_write = np.abs(read_currents) < write_currents
    # read_currents[mask_less_than_write] = 0

    # # Read current cannot be greater than the right critical current
    # mask_greater_than_right = read_currents > right_critical_currents
    # read_currents[mask_greater_than_right] = 0

    # mask_list = [mask_negative, mask_zero_persistent, mask_less_than_write]
    mask_list = []
    return read_currents, mask_list


def calculate_critical_current_bounds(persistent_current, read_current, alpha):
    return read_current * alpha * np.ones((2, 1)) + [
        -persistent_current,
        persistent_current,
    ]
