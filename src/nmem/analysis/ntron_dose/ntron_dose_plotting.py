import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from nmem.analysis.analysis import set_plot_style
from PIL import Image

set_plot_style()  # Optional, comment out if unavailable

def center_crop_zoom(img, zoom_factor=2):
    """Rotate by 90 degrees, then crop the center and zoom in by the given factor."""
    rotated = img.rotate(-90)  # counterclockwise
    w, h = rotated.size
    crop_w, crop_h = int(w / zoom_factor), int(h / zoom_factor)

    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    right = left + crop_w
    bottom = top + crop_h

    cropped = rotated.crop((left, top, right, bottom))
    zoomed = cropped.resize((w, h), Image.LANCZOS)
    return zoomed

# --- User input ---
image_folder = "ntron_dose_images"
output_file = "dose_grid.pdf"
doses = np.linspace(250, 440, 20)  # Doses in µC/cm²
ncols = 5
nrows = (len(doses) + ncols - 1) // ncols

# Load, rotate, crop, and zoom images
images = []
for i in range(20):
    img_path = os.path.join(image_folder, f"C6_{i:02d}.tif")
    img = Image.open(img_path)
    zoomed_img = center_crop_zoom(img, zoom_factor=8)
    images.append(zoomed_img)

# Plot
fig = plt.figure(figsize=(7, 4), facecolor='white')
gs = gridspec.GridSpec(nrows, ncols, wspace=0.15, hspace=0.25)

for i, (dose, img) in enumerate(zip(doses, images)):
    ax = fig.add_subplot(gs[i])
    ax.imshow(img)
    ax.set_title(f"{dose:.0f} µC/cm$^2$", fontsize=10, pad=2)
    ax.axis("off")

fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.2, wspace=0.2)

plt.savefig(output_file, bbox_inches="tight", transparent=False)

plt.show()
plt.close()
