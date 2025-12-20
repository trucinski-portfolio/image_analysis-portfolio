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

# Load image and binarize

image = io.imread(IMAGE_PATH, as_gray=True)
thresh = threshold_otsu(image)

# Beans are darker than background -> invert threshold
binary_image = image <= thresh

# Distance transform + smoothing

distance = ndi.distance_transform_edt(binary_image)
smoothed_distance = gaussian(distance, sigma=2)

# Marker detection

local_maxima = smoothed_distance == ndi.maximum_filter(smoothed_distance, size=20)
markers, _ = ndi.label(local_maxima)


# Watershed segmentation

labeled_image = watershed(-smoothed_distance, markers, mask=binary_image)

# Relabel for clean, consecutive labels
labeled_image = measure.label(labeled_image, background=0)


# Visualization

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image, cmap="gray")

bean_properties = measure.regionprops(labeled_image)

for prop in bean_properties:
    y, x = prop.centroid
    if prop.area > 150:  # minimum size threshold to ignore noise
        ax.text(
            x,
            y,
            str(prop.label),
            color="red",
            fontsize=10,
            ha="center",
            va="center"
        )

ax.set_title("Segmented and Labeled Coffee Beans (Watershed)")
ax.axis("off")

# Save output

output_path = OUTPUT_DIR / "watershed_labeled_beans.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.show()