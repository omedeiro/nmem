def calculate_branch_currents(
    state_currents: list, persistent_current: float, alpha: float, retrap: float
):
    zero_current = state_currents[0]
    one_current = state_currents[1]

    list1 = [
        zero_current * alpha + persistent_current,
        one_current * (1 - alpha) + persistent_current,
    ]
    list2 = [
        (zero_current - list1[1]) / retrap,
        (one_current - list1[0]) / retrap,
    ]

    return list1, list2


if __name__ == "__main__":
    state_currents = [620, 640]
    persistent_current = 1
    alpha = 0.32
    retrap = 0.64

    print(calculate_branch_currents(state_currents, persistent_current, alpha, retrap))
