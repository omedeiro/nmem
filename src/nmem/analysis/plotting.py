




# def plot_optimal_enable_currents(ax: Axes, data_dict: dict) -> Axes:
#     cell = get_current_cell(data_dict)
#     enable_read_current = get_optimal_enable_read_current(cell)
#     enable_write_current = get_optimal_enable_write_current(cell)
#     ax.vlines(
#         [enable_write_current],
#         *ax.get_ylim(),
#         linestyle="--",
#         color="grey",
#         label="_Enable Write Current",
#     )
#     ax.vlines(
#         [enable_read_current],
#         *ax.get_ylim(),
#         linestyle="--",
#         color="r",
#         label="_Enable Read Current",
#     )
#     return ax







