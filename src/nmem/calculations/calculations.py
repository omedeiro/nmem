import numpy as np


def htron_critical_current(slope, intercept, heater_current):
    return heater_current * slope + intercept

def calculate_right_branch_inductance(alpha, ll):
    return alpha * ll / (1 - alpha)


def calculate_left_branch_inductance(alpha, lr):
    return (1 - alpha) * lr / alpha


def calculate_right_branch_current(alpha, channel_current, persistent_current):
    """State 1 is defined as a positive persistent current, clockwise
    State 0 is defined as a negative persistent current, counter-clockwise
    Right branch sums current during read.
    """
    return channel_current * (1 - alpha) + persistent_current


def calculate_left_branch_current(alpha, channel_current, persistent_current):
    """State 1 is defined as a positive persistent current, clockwise
    State 0 is defined as a negative persistent current, counter-clockwise
    Left branch negates current during read.
    """
    return channel_current * alpha - persistent_current


def caluclate_branch_critical_currents(critical_current, width_left, width_right):
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


def calculate_0_current(ichl, ichr, alpha, persistent_current, iretrap=0.5):
    return np.max([(ichl - persistent_current) / alpha, ichl * iretrap + ichr])


def calculate_1_current(ichl, ichr, alpha, persistent_current, iretrap=0.5):
    return np.max([(ichr - persistent_current) / (1 - alpha), ichr * iretrap + ichl])


def calculate_alpha(ll, lr):
    """ll < lr"""
    return lr / (ll + lr)


def calculate_persistent_current(left_critical_current, write_current, alpha, ichl, ichr):

    # Assuming no persistent current in the loop
    left_branch_current = write_current * alpha
    right_branch_current = write_current * (1 - alpha)

    # Assuming that all of the excess current is persistent. None shunted to ground.
    persistent_current = left_branch_current - left_critical_current

    # Exclude negative persistent currents.
    # Write current cannot be less than the left critical current, no hotspot
    mask_negative = persistent_current < 0
    persistent_current = np.where(mask_negative, 0, persistent_current)

    # The right critical current is the left critical current scaled
    # by the ratio of the switching currents.
    right_critical_current = left_critical_current * ichr / ichl

    # Exclude persistent current values that are greater than the right critical current
    mask_right_switch = right_branch_current > right_critical_current
    inverted_persistent_current = right_branch_current - right_critical_current
    persistent_current = np.where(
        mask_right_switch,
        inverted_persistent_current,
        persistent_current,
    )

    # Exclude persistent current values that are greater than the critical current
    mask_persistent_switch = np.abs(persistent_current) > np.abs(left_critical_current)
    persistent_current[mask_persistent_switch] = 0

    mask_list = [mask_negative, mask_right_switch, mask_persistent_switch]
    return persistent_current, mask_list


def calculate_read_currents(
    left_critical_currents, write_currents, persistent_currents, alpha, ichr, ichl
):
    [xx, yy] = np.meshgrid(left_critical_currents, write_currents)
    right_critical_currents = xx * ichr / ichl
    read_currents = np.zeros_like(xx)
    for i in range(len(write_currents)):
        for j in range(len(left_critical_currents)):

            left_lower, left_upper = calculate_read_limits_left(
                alpha, xx[i, j], persistent_currents[i, j]
            )

            right_lower, right_upper = calculate_read_limits_right(
                alpha, right_critical_currents[i, j], persistent_currents[i, j]
            )
            # right_lower, right_upper = calculate_right_branch_limits(
            #     ALPHA, yy[i, j], persistent_currents[i, j]
            # )

            # read_lower = (
            #     np.max([left_lower, right_lower]) + persistent_currents[i, j]
            # ) / ALPHA
            # read_upper = (
            #     np.min([left_upper, right_upper]) + persistent_currents[i, j]
            # ) / ALPHA

            read_currents[i, j] = right_critical_currents[i, j] / alpha

    # Negative read currents are not possible
    mask_negative = read_currents < 0
    read_currents[mask_negative] = 0

    # # Read current NA when persistent current is zero
    mask_zero_persistent = persistent_currents == 0
    read_currents[mask_zero_persistent] = 0

    # # Read current cannot be less than the write current
    mask_less_than_write = np.abs(read_currents) < write_currents
    read_currents[mask_less_than_write] = 0

    # # Read current cannot be greater than the right critical current
    # mask_greater_than_right = read_currents > right_critical_currents
    # read_currents[mask_greater_than_right] = 0

    # mask_list = [mask_negative, mask_zero_persistent, mask_less_than_write]
    mask_list = []
    return read_currents, mask_list
