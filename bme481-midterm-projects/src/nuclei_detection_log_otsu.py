from pathlib import Path
import matplotlib.pyplot as plt
from skimage import io
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

SIGMA = 7  # controls blob scale; tune based on nuclei size

# Load + Laplacian of Gaussian

img = io.imread(IMAGE_PATH)

# LoG response (blobs appear as negative extrema depending on contrast)
log_img = gaussian_laplace(img.astype(float), sigma=SIGMA)

# Otsu threshold on LoG response
thr = threshold_otsu(log_img)
binary = log_img <= thr  # invert if nuclei appear dark in LoG response

# Label connected components
labeled, num = ndi.label(binary)

print(f"Detected connected components (pre-filter): {num}")

# Save figure

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(img, cmap="gray")
ax[0].set_title("Original (DAPI)")
ax[0].axis("off")

ax[1].imshow(binary, cmap="gray")
ax[1].set_title(f"LoG + Otsu (components: {num})")
ax[1].axis("off")

out_path = OUT_DIR / "dapi_log_otsu_binary.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.show()