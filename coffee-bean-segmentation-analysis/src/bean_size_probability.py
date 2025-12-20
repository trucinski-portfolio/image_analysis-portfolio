import numpy as np
from pathlib import Path
from skimage import io, measure
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, gaussian
from skimage.segmentation import watershed
import matplotlib.pyplot as plt

# Path initialization

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_PATH = BASE_DIR / "data" / "CoffeeBeans.tif"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load + binarize (Otsu)

image = io.imread(IMAGE_PATH, as_gray=True)
thresh = threshold_otsu(image)
binary_image = image <= thresh  # beans are darker -> foreground

# Segmentation (distance transform + watershed)

distance = ndi.distance_transform_edt(binary_image)
smoothed_distance = gaussian(distance, sigma=2)

local_maxima = smoothed_distance == ndi.maximum_filter(smoothed_distance, size=20)
markers, _ = ndi.label(local_maxima)

labeled_image = watershed(-smoothed_distance, markers, mask=binary_image)
labeled_image = measure.label(labeled_image, background=0)

# Quantification

MIN_AREA = 150  # ignore tiny regions/noise
bean_properties = measure.regionprops(labeled_image)

bean_sizes = np.array([prop.area for prop in bean_properties if prop.area > MIN_AREA])
num_beans = bean_sizes.size

if num_beans == 0:
    raise RuntimeError("No beans detected above MIN_AREA threshold. Try lowering MIN_AREA or adjusting markers.")

mean_size = float(np.mean(bean_sizes))
num_small_beans = int(np.sum(bean_sizes < (mean_size / 2)))
prob_less_than_half_mean = num_small_beans / num_beans

# Console output
print(f"Total number of coffee beans (area > {MIN_AREA}): {num_beans}")
print(f"Average size (mean area): {mean_size:.2f} pixels")
print(f"P(bean area < mean/2): {prob_less_than_half_mean:.3f} ({num_small_beans}/{num_beans})")

# Save a short text summary 

summary_path = OUTPUT_DIR / "bean_size_summary.txt"
summary_text = (
    f"Coffee Bean Size Summary\n"
    f"------------------------\n"
    f"Min area threshold: {MIN_AREA}\n"
    f"Count (beans): {num_beans}\n"
    f"Mean area (pixels): {mean_size:.2f}\n"
    f"Count < mean/2: {num_small_beans}\n"
    f"P(area < mean/2): {prob_less_than_half_mean:.3f}\n"
)
summary_path.write_text(summary_text)

# Histogram (density)

plt.figure(figsize=(10, 6))
plt.hist(bean_sizes, bins=30, edgecolor="black", density=True)
plt.title("Coffee Bean Size Distribution (Area)")
plt.xlabel("Bean Area (pixels)")
plt.ylabel("Probability Density")

hist_path = OUTPUT_DIR / "bean_size_density.png"
plt.savefig(hist_path, dpi=200, bbox_inches="tight")
plt.show()