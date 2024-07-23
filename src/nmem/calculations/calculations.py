import numpy as np
import matplotlib.ticker as ticker


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


def calculate_0_state_currents(
    left_critical_current: np.ndarray,
    right_critical_current: np.ndarray,
    persistent_current: np.ndarray,
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

    Returns
    -------
    The bias current limits assuming the device is in state 0."""

    # The 0 state is defined as a negative persistent current
    if persistent_current > 0:
        persistent_current = -persistent_current

    bias_current_left = (left_critical_current - persistent_current) / alpha
    bias_current_right = (right_critical_current + persistent_current) / (1 - alpha)

    return bias_current_left, bias_current_right


def calculate_1_state_currents(
    left_critical_current: np.ndarray,
    right_critical_current: np.ndarray,
    persistent_current: np.ndarray,
    iretrap: float,
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
    if persistent_current < 0:
        persistent_current = -persistent_current

    bias_current_left = (left_critical_current + persistent_current) / alpha
    bias_current_right = (right_critical_current - persistent_current) / (1 - alpha)

    return bias_current_left, bias_current_right


def calculate_alpha(ll, lr):
    """ll < lr"""
    return lr / (ll + lr)


def calculate_persistent_current(
    left_critical_currents: np.ndarray,
    write_currents: np.ndarray,
    alpha: float,
    iretrap: float,
    max_left_critical_current: float,
    max_right_critical_current: float,
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
        The ratio of the right branch inductance to the total inductance. LR/(LL+LR)
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
    left_branch_current = write_currents * (1 - alpha)  # LL / (LL+LR)
    right_branch_current = write_currents * alpha  # LR / (LL+LR)

    # If the left branch current is greater than the
    # left critical current, the branch switches.
    left_switch = left_branch_current > left_critical_currents

    # Where the left branch switched the persistent current is set to the write current.
    persistent_current = np.where(left_switch, write_currents, persistent_current)

    # Therefore, the right branch must carry the full write current.
    right_branch_current = np.where(left_switch, write_currents, right_branch_current)

    # If the right branch also switches...
    right_switch = (right_branch_current > right_critical_currents) * left_switch

    # The right branch will direct back to the left branch
    right_retrap_left_current = write_currents - right_critical_currents * iretrap
    persistent_current = np.where(
        right_switch, -right_retrap_left_current, persistent_current
    )

    # If the redirected current is enough the switch the left branch,
    # both branches are switched during the write and the persistent current
    # is set to zero
    right_retrap_left_switch = right_retrap_left_current > left_critical_currents
    persistent_current = np.where(
        right_switch * right_retrap_left_switch,
        0,
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
        "left_switch": left_switch,
        "right_switch": right_switch,
        "right_retrap": right_retrap_left_current,
        "left_persistent_switch": left_persistent_switch,
    }

    return persistent_current, regions


def calculate_read_currents(
    left_critical_currents: np.ndarray,
    write_currents: np.ndarray,
    persistent_currents: np.ndarray,
    alpha: float,
    iretrap: float,
    max_left_critical_current: float,
    max_right_critical_current: float,
):
    ic_ratio = max_left_critical_current / max_right_critical_current

    [xx, yy] = np.meshgrid(left_critical_currents, write_currents)
    right_critical_currents = left_critical_currents / ic_ratio
    # read_currents = np.zeros_like(xx)

    zero_switching_current = calculate_0_current(
        xx,
        yy,
        persistent_currents,
        alpha,
        iretrap,
    )
    one_switching_current = calculate_1_current(
        xx,
        yy,
        persistent_currents,
        alpha,
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
    read_currents = np.where(mask_zero_persistent, 0, read_currents)

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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    left_critical_current = 22
    right_critical_current = 87
    persistent_current = 60
    zero_current_left, zero_current_right = calculate_0_state_currents(
        np.array([left_critical_current]),
        np.array([right_critical_current]),
        np.array([persistent_current]),
        0.63,
        0.9,
    )
    one_current_left, one_current_right = calculate_1_state_currents(
        np.array([left_critical_current]),
        np.array([right_critical_current]),
        np.array([persistent_current]),
        0.63,
        0.9,
    )
    print(
        f"Zero current left: {zero_current_left}, Zero current right: {zero_current_right}"
    )
    print(
        f"One current left: {one_current_left}, One current right: {one_current_right}"
    )

    READ_BIAS = 10.0
    fig, ax = plt.subplots()
    plt.bar(
        [-2, -1],
        [-persistent_current+READ_BIAS*(0.63), persistent_current+READ_BIAS*(0.63)],
        color=["b", "r"],
        width=1,
        bottom=left_critical_current,
    )
    plt.bar(
        [1, 2],
        [persistent_current+READ_BIAS*(1-0.63), READ_BIAS],
        color=["b", "r"],
        width=1,
        bottom=right_critical_current,
    )
    # plt.hlines(READ_BIAS, -4, 4, color="k")
    # plt.hlines(-READ_BIAS, -4, 4, color="k")
    # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%+g"))

    plt.yticks()
    plt.xticks([-1.5, 1.5], ["0", "1"])
    plt.xlabel("State")
    plt.grid(axis="y")

    plt.hlines(left_critical_current, -2.75, -0.25, color="k")
    plt.hlines(right_critical_current, 0.25, 2.75, color="k")
    ax.set_ylim(bottom=0)
    # plt.text(0, 50, "not switching", ha="center", va="center", color="black")
    # plt.text(0, -50, "switching", ha="center", va="center", color="black")
    # plt.hlines(right_critical_current, -1, 6, color="r")
