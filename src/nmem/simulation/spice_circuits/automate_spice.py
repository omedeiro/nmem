import logging
import os
import subprocess

import ltspice
import numpy as np
import pandas as pd

ltspice_path = r"C:\Users\omedeiro\AppData\Local\Programs\ADI\LTspice\LTspice.exe"
template_netlist = "nmem_cell_read_v3.cir"
wave_dir = (
    r"C:\Users\omedeiro\nmem\src\nmem\simulation\spice_circuits\data\write_amp_sweep"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def generate_netlist(template_path, pwl_path, output_path):
    """Generate a temporary netlist with the given PWL file."""
    with open(template_path, "r") as f:
        netlist = f.read().replace("CHANPWL", f'"{pwl_path}"')
    with open(output_path, "w") as f:
        f.write(netlist)
    logging.info(f"Generated netlist: {output_path}")


def run_ltspice(ltspice_path, netlist_path):
    """Run LTspice simulation."""
    subprocess.run([ltspice_path, "-b", netlist_path], check=True)
    logging.info(f"LTspice simulation completed for: {netlist_path}")


def parse_raw_file(raw_path):
    """Parse the .raw file and extract persistent current."""
    l = ltspice.Ltspice(raw_path)
    l.parse()
    time = l.get_time()
    ir = l.get_data("Ix(HR:drain)")
    mask = (time >= 3.9e-6) & (time <= 4.1e-6)
    # mask = (time >= 0.5e-6) & (time <= 0.6e-6)

    persistent_current = np.mean(ir[mask]) * 1e6  # µA
    logging.info(f"Extracted persistent current: {persistent_current:.2f} µA")
    return persistent_current


def save_results_to_csv(data, output_csv):
    """Save the results to a CSV file."""
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    logging.info(f"Results saved to {output_csv}")


def main():
    output_data = []
    for fname in sorted(os.listdir(wave_dir)):
        if not fname.endswith(".txt"):
            continue

        full_pwl_path = os.path.join(wave_dir, fname)
        amp_uA = int(fname.split("_")[-1].replace("u.txt", ""))
        modified_netlist = os.path.join(wave_dir, f"temp_{amp_uA:+05g}u.cir")

        generate_netlist(template_netlist, full_pwl_path, modified_netlist)
        run_ltspice(ltspice_path, modified_netlist)

        raw_path = modified_netlist.replace(".cir", ".raw")
        persistent_current = parse_raw_file(raw_path)

        output_data.append(
            {"Amplitude (uA)": amp_uA, "Persistent Current (uA)": persistent_current}
        )

    save_results_to_csv(
        output_data,
        r"C:\Users\omedeiro\nmem\src\nmem\simulation\spice_circuits\data\write_sweep_results.csv",
    )


if __name__ == "__main__":
    main()
