import os
from tdgl import SolverOptions, solve

def run_simulation(device, current, stime, path, prev_sol=None):
    os.makedirs(path, exist_ok=True)
    output_file = os.path.join(path, f"current_{int(current)}uA.h5")
    options = SolverOptions(
        skip_time=500,
        solve_time=stime,
        output_file=output_file,
        field_units="mT",
        current_units="uA",
        save_every=50,
    )
    return solve(device, options, terminal_currents={"source": current, "drain": -current}, applied_vector_potential=0, seed_solution=prev_sol)

def get_current_through_path(solution, path, dataset="supercurrent", with_units=True):
    return solution.current_through_path([tuple(coord) for coord in path], with_units=with_units, dataset=dataset)

def make_animation_from_solution(solution, output_path, tag, quantities=("order_parameter", "phase"), fps=20):
    from tdgl.visualization.animate import create_animation
    from IPython.display import HTML, display
    import h5py
    with h5py.File(solution.path, "r") as h5file:
        anim = create_animation(h5file, quantities=quantities, fps=fps)
        html = anim.to_html5_video()
    display(HTML(html))
    with open(os.path.join(output_path, f"{tag}.html"), "w") as f:
        f.write(html)
