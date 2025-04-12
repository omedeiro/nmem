import argparse
import os
from datetime import datetime
from nmem.simulation.pytdgl.devices.memory_cell import make_device
from nmem.simulation.pytdgl.sim.util import run_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--current", type=float, default=2000, help="Source current in uA")
parser.add_argument("--time", type=float, default=100, help="Solve time in ps")
parser.add_argument("--tag", type=str, default="", help="Tag for output folder")
args = parser.parse_args()

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
tagged_folder = f"{args.tag}_{timestamp}" if args.tag else timestamp
output_path = os.path.join("output", tagged_folder)
os.makedirs(output_path, exist_ok=True)

device = make_device()
device.make_mesh(max_edge_length=0.062)  # Example value: 10 × XI
solution = run_simulation(device, current=args.current, stime=args.time, path=output_path)

print("Simulation complete. Saved to:", solution.path)
