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

# Beans are darker than background -> invert threshold
binary_image = image <= thresh

# Segmentation (distance transform + watershed)

distance = ndi.distance_transform_edt(binary_image)
smoothed_distance = gaussian(distance, sigma=2)

local_maxima = smoothed_distance == ndi.maximum_filter(smoothed_distance, size=20)
markers, _ = ndi.label(local_maxima)

labeled_image = watershed(-smoothed_distance, markers, mask=binary_image)
labeled_image = measure.label(labeled_image, background=0)

# Quantification (areas + count)

bean_properties = measure.regionprops(labeled_image)
bean_sizes = [prop.area for prop in bean_properties]
num_beans = len(bean_sizes)

print(f"Total number of coffee beans: {num_beans}")

# Figure 1: labeled beans overlay - isolated in watershed_segmentation.py

#fig, ax = plt.subplots(figsize=(10, 10))
#ax.imshow(image, cmap="gray")

#for prop in bean_properties:
#    y, x = prop.centroid
#    if prop.area > 150:  # minimum size threshold to ignore noise
#        ax.text(x, y, str(prop.label), color="red", fontsize=10, ha="center", va="center")

#ax.set_title("Segmented and Labeled Coffee Beans")
#ax.axis("off")

#overlay_path = OUTPUT_DIR / "beans_labeled_overlay.png"
#plt.savefig(overlay_path, dpi=200, bbox_inches="tight")
#plt.show()

# Figure 2: size distribution histogram

plt.figure(figsize=(10, 6))
plt.hist(bean_sizes, bins=20, edgecolor="black")
plt.title("Size Distribution of Coffee Beans")
plt.xlabel("Bean Size (Area, pixels)")
plt.ylabel("Frequency")

hist_path = OUTPUT_DIR / "bean_size_histogram.png"
plt.savefig(hist_path, dpi=200, bbox_inches="tight")
plt.show()