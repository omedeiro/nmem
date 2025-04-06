import subprocess
import os
import ltspice
import numpy as np
import pandas as pd

ltspice_path = r"C:\Users\omedeiro\AppData\Local\Programs\ADI\LTspice\LTspice.exe"  
template_netlist = "nmem_cell_read_v3.cir" 
wave_dir = r"C:\Users\omedeiro\nmem\src\nmem\simulation\spice_circuits\data\write_amp_sweep"
output_data = []

# Loop through generated files
for fname in sorted(os.listdir(wave_dir)):
    if not fname.endswith(".txt"):
        continue

    # Full paths
    full_pwl_path = os.path.join(wave_dir, fname)
    amp_uA = int(fname.split("_")[-1].replace("u.txt", ""))

    # Generate a temporary netlist for this run
    modified_netlist = os.path.join(wave_dir, f"temp_{amp_uA}u.cir")
    with open(template_netlist, "r") as f:
        print(f"full_pwl_path: {full_pwl_path}")
        net = f.read().replace("CHANPWL", f'"{full_pwl_path}"')
    with open(modified_netlist, "w") as f:
        f.write(net)

    # Run LTspice
    subprocess.run([ltspice_path, "-b", modified_netlist], check=True)

    # Load .raw file
    raw_path = modified_netlist.replace(".cir", ".raw")
    l = ltspice.Ltspice(raw_path)
    l.parse()

    # Example: extract max of voltage at node 'out'
    time = l.get_time()
    ir = l.get_data("Ix(HR:drain)")
    persistent_current = ir[-1]*1e6


    output_data.append({"Amplitude (uA)": amp_uA, "Persistent Current (uA)": persistent_current})

# Save to CSV and display
df = pd.DataFrame(output_data)
df.to_csv(r"C:\Users\omedeiro\nmem\src\nmem\simulation\spice_circuits\data\write_sweep_results.csv", index=False)
df
