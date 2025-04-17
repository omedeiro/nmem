import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
# from nmem.analysis.analysis import set_plot_style

# set_plot_style()  # Optional, comment out if unavailable

# Load log
log_path = Path("New schedule.log")
with open(log_path, 'r', encoding='utf-8') as f:
    log_text = f.read()
    lines = log_text.splitlines()

# Patterns
device_header_re = re.compile(r'\d+\s+(WSP_[A-Za-z0-9_]+)')
z_try_re = re.compile(r'z try:\s+(\d+)')
xy_try_re = re.compile(r'xy try:\s+(\d+)')
z_height_re = re.compile(r'z:\s+([\d.]+)\s+\[mm\]')
rotation_re = re.compile(r'rotation:\s+([\d.]+)\s+\[mrad\]')
car_block_pattern = re.compile(r"^\s*\d+\s+(?P<car_file>\w+\.car).*?(?=^\s*\d+\s+\w+\.car|\Z)", re.DOTALL | re.MULTILINE)
wafer_rotation_pattern = re.compile(r"wafer rotation\s+(-?[\d.]+)\s+\[mrad\]")
shift_pattern = re.compile(r"shift\s+\(\s*(-?[\d.]+),\s*(-?[\d.]+)\s*\)\s+\[mm\]")

# Extract z height, retry count, rotation
data = []
rotations = []
current_device = None
current_entry = {}

for line in lines:
    line = line.strip()
    if match := device_header_re.match(line):
        if current_entry:
            data.append(current_entry)
        current_device = match.group(1)
        current_entry = {"device": current_device, "xy_try": None, "z_try": None, "z_height_mm": None}
    elif match := xy_try_re.search(line):
        current_entry["xy_try"] = int(match.group(1))
    elif match := z_try_re.search(line):
        current_entry["z_try"] = int(match.group(1))
    elif match := z_height_re.search(line):
        current_entry["z_height_mm"] = float(match.group(1))
    elif match := rotation_re.search(line):
        if current_device:
            rotations.append({"device": current_device, "rotation_mrad": float(match.group(1))})

if current_entry:
    data.append(current_entry)

df_z = pd.DataFrame(data).dropna(subset=["z_height_mm"])
df_rot = pd.DataFrame(rotations)
df_rot_valid = df_rot[df_rot["rotation_mrad"] >= 1.0]

# Inter-pass alignment deltas
inter_pass_deltas = []

for match in car_block_pattern.finditer(log_text):
    car_file = match.group("car_file")
    block_text = match.group(0)

    # Match alignment search pass and mark
    search_matches = re.findall(
        r"alignment search (\d)-([A-Z])[\s\S]*?searched position:\s+([\d.]+)\s+([\d.]+)", block_text
    )
    mark_dict = {}
    for pass_id, mark_id, x_str, y_str in search_matches:
        key = mark_id.upper()
        mark_dict.setdefault(key, {})[pass_id] = (float(x_str), float(y_str))

    for mark_id, passes in mark_dict.items():
        if "1" in passes and "2" in passes:
            (x1, y1), (x2, y2) = passes["1"], passes["2"]
            dx_nm = (x2 - x1) * 1e6
            dy_nm = (y2 - y1) * 1e6
            inter_pass_deltas.append({
                "car_file": car_file,
                "mark_id": mark_id,
                "dx_nm": round(dx_nm, 2),
                "dy_nm": round(dy_nm, 2)
            })

# Convert to DataFrame
delta_table = pd.DataFrame(inter_pass_deltas)
dx_nm = delta_table["dx_nm"].to_numpy()
dy_nm = delta_table["dy_nm"].to_numpy()

# --- Stats ---
z_mean, z_std = df_z["z_height_mm"].mean(), df_z["z_height_mm"].std()
r_mean, r_std = df_rot_valid["rotation_mrad"].mean(), df_rot_valid["rotation_mrad"].std()

# --- Plotting ---
fig, axs = plt.subplots(1, 3, figsize=(10, 3.5))

# Z height
axs[0].hist(df_z["z_height_mm"], bins=20, edgecolor='black', color="#1f77b4")
axs[0].set_xlabel("Z Height [mm]")
axs[0].set_ylabel("Count")
axs[0].text(0.97, 0.97, f"$\\mu$ = {z_mean:.4f} mm\n$\\sigma$ = {z_std:.4f} mm",
            transform=axs[0].transAxes, fontsize=10, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9))
axs[0].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# Rotation
axs[1].hist(df_rot_valid["rotation_mrad"], bins=20, edgecolor='black', color="#1f77b4")
axs[1].set_xlabel("Rotation [mrad]")
axs[1].set_ylabel("Count")
axs[1].text(0.97, 0.97, f"$\\mu$ = {r_mean:.2f} mrad\n$\\sigma$ = {r_std:.2f} mrad",
            transform=axs[1].transAxes, fontsize=10, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9))
axs[1].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# Alignment offsets
ax = axs[2]
sns.kdeplot(
    x=dx_nm, y=dy_nm, fill=True, cmap="crest",
    bw_adjust=0.7, levels=10, thresh=0.05, ax=ax
)
ax.scatter(
    dx_nm, dy_nm,
    color="#333333", s=15, marker='o',
    label="Alignment Marks", alpha=0.8
)
ax.axhline(0, color='black', linestyle='--', linewidth=1)
ax.axvline(0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel("ΔX [nm]")
ax.set_ylabel("ΔY [nm]")
ax.axis('equal')
# ax.set_title("Inter-Pass Alignment Offset")
ax.legend()


plt.tight_layout()
plt.savefig("alignment_analysis.pdf", dpi=300)
plt.show()

# Display final table
print(delta_table)

fig, ax = plt.subplots(figsize=(8, 6))
plt.hist(
    dx_nm, bins=20, edgecolor='black', color="#1f77b4", alpha=0.7,
    label="ΔX [nm]"
)
plt.hist(
    dy_nm, bins=20, edgecolor='black', color="#ff7f0e", alpha=0.7,
    label="ΔY [nm]"
)
plt.xlabel("Alignment Offset [nm]")
plt.ylabel("Count")
plt.title("Histogram of Alignment Offsets")
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.savefig("alignment_offsets_histogram.pdf", dpi=300)
plt.show()