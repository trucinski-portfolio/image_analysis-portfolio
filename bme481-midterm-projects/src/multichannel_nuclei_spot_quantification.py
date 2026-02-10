import numpy as np
from pathlib import Path
from skimage import io, measure
from skimage.filters import threshold_otsu, gaussian
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

# Paths

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

IMAGE_PATH = DATA_DIR / "nuclei.png"

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Missing required image: {IMAGE_PATH}\n"
        "Place the file in the repo's data/ folder."
    )

# Parameters

SPOT_SENSITIVITY_FACTOR = 3.35   # increases detection sensitivity for puncta
SPOT_MIN_DISTANCE = 3
DISTANCE_SMOOTH_SIGMA = 1.25

# Load image

image = io.imread(IMAGE_PATH)

blue = image[:, :, 2]   # DAPI / nuclei
green = image[:, :, 1]  # punctate signal

# Nuclei segmentation (Blue)

thr_blue = threshold_otsu(blue)
binary_nuclei = blue > thr_blue
labeled_nuclei, num_nuclei = measure.label(binary_nuclei, return_num=True)

# Spot detection (Green)

thr_green = threshold_otsu(green)
adjusted_thr = thr_green * SPOT_SENSITIVITY_FACTOR
binary_spots = green > adjusted_thr

distance = ndi.distance_transform_edt(binary_spots)
distance_smoothed = gaussian(distance, sigma=DISTANCE_SMOOTH_SIGMA)

local_max = peak_local_max(
    distance_smoothed,
    min_distance=SPOT_MIN_DISTANCE,
    labels=binary_spots
)

markers = np.zeros_like(green, dtype=int)
markers[tuple(local_max.T)] = np.arange(1, local_max.shape[0] + 1)

separated_spots = watershed(-distance_smoothed, markers, mask=binary_spots)

# Count spots robustly (exclude background label 0)
spot_labels = np.unique(separated_spots)
spot_labels = spot_labels[spot_labels != 0]
num_spots = len(spot_labels)

spots_per_nucleus = num_spots / num_nuclei if num_nuclei > 0 else 0

# Centroids for plotting
spot_props = measure.regionprops(separated_spots)

# Save derived output (labels)

plt.figure(figsize=(8, 8))
plt.imshow(separated_spots, cmap="nipy_spectral")
plt.title("Watershed-Separated Spots (Derived Output)")
plt.axis("off")
plt.savefig(OUT_DIR / "green_channel_spot_labels.png", dpi=200, bbox_inches="tight")
plt.show()

# Save RAW-inclusive figure (2x2)

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Blue channel
axes[0, 0].imshow(blue, cmap="gray")
axes[0, 0].set_title("Blue Channel (DAPI / Nuclei)")
axes[0, 0].axis("off")

# Nuclei labels
axes[0, 1].imshow(labeled_nuclei, cmap="nipy_spectral")
axes[0, 1].set_title(f"Nuclei Segmentation (n={num_nuclei})")
axes[0, 1].axis("off")

# Green channel + spot centroids
axes[1, 0].imshow(green, cmap="gray")
axes[1, 0].set_title(f"Green Channel Spots (n={num_spots})")
for prop in spot_props:
    y, x = prop.centroid
    axes[1, 0].plot(x, y, "rx", markersize=3.5)
axes[1, 0].axis("off")

# Color image + spot centroids
axes[1, 1].imshow(image)
axes[1, 1].set_title(f"Composite + Spots per Nucleus = {spots_per_nucleus:.2f}")
for prop in spot_props:
    y, x = prop.centroid
    axes[1, 1].plot(x, y, "rx", markersize=3.5)
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig(OUT_DIR / "nuclei_spot_quantification_raw.png", dpi=200, bbox_inches="tight")
plt.show()

# Save summary text

summary = (
    f"Nuclei count: {num_nuclei}\n"
    f"Spot count: {num_spots}\n"
    f"Spots per nucleus: {spots_per_nucleus:.2f}\n"
    f"Spot sensitivity factor (Otsu * factor): {SPOT_SENSITIVITY_FACTOR}\n"
    f"Spot min_distance: {SPOT_MIN_DISTANCE}\n"
    f"Distance smoothing sigma: {DISTANCE_SMOOTH_SIGMA}\n"
)
(OUT_DIR / "spot_quantification_summary.txt").write_text(summary)

print(summary)