
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

# === Global Configuration ===
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

# === Constants ===
RADIUS = 50  # mm
GRID_RES = 400j
NUM_BOUNDARY_POINTS = 100
CSV_BEFORE = "MapResult_623.csv"
CSV_AFTER = "MapResult_624.csv"
ANNOTATE_POINTS = False  # Toggle text annotations on/off

# === Utility Functions ===

def load_and_clean_thickness(path):
    df = pd.read_csv(path)
    df['d(nm)'] = df['d(nm)'].astype(str).str.extract(r'([-+]?\d*\.\d+|\d+)').astype(float)
    grouped = df.groupby(['Y', 'X'])['d(nm)'].mean().reset_index()
    map_df = grouped.pivot(index='Y', columns='X', values='d(nm)')
    return map_df.drop(index='Y', errors='ignore').drop(columns='X', errors='ignore')

def generate_boundary_points(radius, n_points, fill_value):
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    bx = radius * np.cos(angles)
    by = radius * np.sin(angles)
    bz = np.full_like(bx, fill_value)
    return bx, by, bz

def interpolate_map(data_map, radius, grid_x, grid_y, boundary_pts):
    x_vals = data_map.columns.astype(float)
    y_vals = data_map.index.astype(float)
    xy = np.array([(x, y) for y in y_vals for x in x_vals])
    z = data_map.values.flatten()
    mask = ~np.isnan(z)
    xy, z = xy[mask], z[mask]

    bx, by, bz = boundary_pts
    aug_xy = np.column_stack([np.concatenate([xy[:, 0], bx]),
                              np.concatenate([xy[:, 1], by])])
    aug_z = np.concatenate([z, bz])

    grid_z = griddata(aug_xy, aug_z, (grid_x, grid_y), method='cubic')
    distance = np.sqrt(grid_x**2 + grid_y**2)
    grid_z[distance > radius] = np.nan
    return grid_z, xy, z

def plot_wafer_maps(maps, titles, cmaps, grid_x, grid_y, radius):
    fig, axes = plt.subplots(1, 3, figsize=(7, 3.5), dpi=300)  # 7.2" ≈ 2-column width
    for ax, title, (grid_z, pts, vals), cmap in zip(axes, titles, maps, cmaps):
        circle = plt.Circle((0, 0), radius, color='k', lw=0.5, fill=False)
        contour = ax.contourf(grid_x, grid_y, grid_z, levels=30, cmap=cmap)
        # ax.scatter(pts[:, 0], pts[:, 1], c='k', s=8, zorder=10)
        if ANNOTATE_POINTS:
            for (x, y), v in zip(pts, vals):
                ax.text(x, y, f"{v:.1f}", ha='center', va='center', fontsize=5, color='white', zorder=11)
        ax.add_artist(circle)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        cbar = fig.colorbar(contour, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Thickness (nm)", fontsize=9)
    plt.tight_layout()
    return fig

# === Load and Interpolate Data ===
before_map = load_and_clean_thickness(CSV_BEFORE)
after_map = load_and_clean_thickness(CSV_AFTER)
delta_map = after_map - before_map

grid_x, grid_y = np.mgrid[-RADIUS:RADIUS:GRID_RES, -RADIUS:RADIUS:GRID_RES]

before_boundary = generate_boundary_points(RADIUS, NUM_BOUNDARY_POINTS, np.nanmean(before_map.values))
after_boundary  = generate_boundary_points(RADIUS, NUM_BOUNDARY_POINTS, np.nanmean(after_map.values))
delta_boundary  = generate_boundary_points(RADIUS, NUM_BOUNDARY_POINTS, np.nanmean(delta_map.values))

before_interp = interpolate_map(before_map, RADIUS, grid_x, grid_y, before_boundary)
after_interp  = interpolate_map(after_map, RADIUS, grid_x, grid_y, after_boundary)
delta_interp  = interpolate_map(delta_map, RADIUS, grid_x, grid_y, delta_boundary)

# === Plot and Save ===
fig = plot_wafer_maps(
    [before_interp, after_interp, delta_interp],
    ["Before Deposition", "After Deposition", "Deposited Thickness"],
    [plt.cm.Blues, plt.cm.Greens, plt.cm.inferno],
    grid_x, grid_y, RADIUS
)

fig.savefig("wafer_maps_before_after_delta.pdf", bbox_inches="tight", dpi=300)
