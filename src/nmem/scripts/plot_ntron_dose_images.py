import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from nmem.analysis.styles import apply_global_style, get_consistent_figure_size
from nmem.analysis.utils import center_crop_zoom

# Apply global plot styling
apply_global_style()


def main(
    image_folder="../data/ntron_dose_test_images",
    doses=None,
    ncols=5,
    zoom_factor=8,
    save_dir=None,
):
    if doses is None:
        doses = np.linspace(250, 440, 20)  # Doses in µC/cm²
    nrows = (len(doses) + ncols - 1) // ncols
    # Load, rotate, crop, and zoom images
    images = []
    for i in range(len(doses)):
        img_path = os.path.join(image_folder, f"C6_{i:02d}.tif")
        img = Image.open(img_path)
        zoomed_img = center_crop_zoom(img, zoom_factor=zoom_factor)
        images.append(zoomed_img)
    # Plot
    figsize = get_consistent_figure_size("grid")
    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = gridspec.GridSpec(nrows, ncols, wspace=0.15, hspace=0.25)
    for i, (dose, img) in enumerate(zip(doses, images)):
        ax = fig.add_subplot(gs[i])
        ax.imshow(img)
        ax.set_title(f"{dose:.0f} µC/cm$^2$", fontsize=10, pad=2)
        ax.axis("off")
    fig.subplots_adjust(
        left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.2, wspace=0.2
    )
    if save_dir:
        fig.savefig(
            os.path.join(save_dir, "ntron_dose_images.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
