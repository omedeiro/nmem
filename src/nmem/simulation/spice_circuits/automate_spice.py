import subprocess
import os
import ltspice
import numpy as np
import pandas as pd

ltspice_path = r"C:\Users\omedeiro\AppData\Local\Programs\ADI\LTspice\LTspice.exe"  # Update for your system
template_netlist = "/home/omedeiro/hTron-behavioral-model/nmem_cell_read_v3.asc"  # Your LTspice circuit template
wave_dir = "data/write_amp_sweep"
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
        net = f.read().replace("chan_pwl", f'"{full_pwl_path}"')
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
    vout = l.get_data('V(out)')
    vpeak = np.max(vout)

    output_data.append({
        "Amplitude (uA)": amp_uA,
        "Peak V(out) (V)": vpeak
    })

# Save to CSV and display
df = pd.DataFrame(output_data)
df.to_csv("data/write_sweep_results.csv", index=False)
df
