from pathlib import Path
import matplotlib.pyplot as plt
from skimage import io, measure
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_laplace
from scipy import ndimage as ndi

# Paths

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

IMAGE_PATH = DATA_DIR / "B6_DAPI_1.tif"

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Missing required image: {IMAGE_PATH}\n"
        "Place the file in the repo's data/ folder (not included publicly)."
    )

# Parameters

SIGMA = 7
MIN_AREA = 70  # pixels; tune to remove tiny artifacts

# Load + segment

img = io.imread(IMAGE_PATH)
log_img = gaussian_laplace(img.astype(float), sigma=SIGMA)

thr = threshold_otsu(log_img)
binary = log_img <= thr

labeled, num_all = ndi.label(binary)
props = measure.regionprops(labeled)

# Filter by size
valid = [p for p in props if p.area >= MIN_AREA]
num_valid = len(valid)

print(f"Total components: {num_all}")
print(f"Valid nuclei (area >= {MIN_AREA}): {num_valid}")

# Plot (2x2) and save image as .png file 

fig, ax = plt.subplots(2, 2, figsize=(12, 12))

ax[0, 0].imshow(img, cmap="gray")
ax[0, 0].set_title("Original (DAPI)")
ax[0, 0].axis("off")

ax[0, 1].imshow(binary, cmap="gray")
ax[0, 1].set_title("Binary (LoG + Otsu)")
ax[0, 1].axis("off")

ax[1, 0].imshow(labeled, cmap="nipy_spectral")
ax[1, 0].set_title("Connected Components")
ax[1, 0].axis("off")

ax[1, 1].imshow(labeled, cmap="nipy_spectral")
ax[1, 1].set_title(f"Filtered + Annotated (>= {MIN_AREA}px)")
ax[1, 1].axis("off")

# Annotate valid nuclei with consecutive labels
for idx, p in enumerate(valid, start=1):
    y, x = p.centroid
    ax[1, 1].text(x, y, str(idx), color="white", fontsize=6, ha="center", va="center")

fig.suptitle(f"Total: {num_all} | Valid: {num_valid}", fontsize=12)
out_path = OUT_DIR / "dapi_cc_filtered_labeled.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.show()