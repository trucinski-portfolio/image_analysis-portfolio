from pathlib import Path
from skimage import io
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

# Resolve paths relative to this script
BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_PATH = BASE_DIR / "data" / "CoffeeBeans.tif"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load image as grayscale
image = io.imread(IMAGE_PATH, as_gray=True)

# Otsu thresholding
thresh = threshold_otsu(image)

# Beans appear darker than background, so invert threshold
binary_image = image <= thresh

# Plot results
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(image, cmap="gray")
ax[0].set_title("Original Grayscale Image")
ax[0].axis("off")

ax[1].imshow(binary_image, cmap="gray")
ax[1].set_title("Binary Image (Otsu Threshold)")
ax[1].axis("off")

# Save output
output_path = OUTPUT_DIR / "otsu_binary.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.show()