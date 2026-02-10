import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import io
from sklearn.decomposition import NMF

# Paths

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

IMAGE_PATH = DATA_DIR / "HE_Brain.bmp"
MASK_PATH = DATA_DIR / "HE_Brain_mask.bmp"

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Missing: {IMAGE_PATH}")
if not MASK_PATH.exists():
    raise FileNotFoundError(f"Missing: {MASK_PATH}")

# Parameters

N_COMPONENTS = 2
N_RUNS = 3
MAX_ITER = 1000

# Load image + mask

image = io.imread(IMAGE_PATH)
mask = io.imread(MASK_PATH)

# Ensure mask is boolean (common: 0/255 uint8)
mask_bool = mask > 0

V = image.reshape(-1, 3).astype(float)
V = V / (V.max() + 1e-12)

# Initialize W using mask prior
# Component 0: "mask-positive" regions
# Component 1: "mask-negative" regions

W_init = np.zeros((V.shape[0], N_COMPONENTS), dtype=float)
flat_mask = mask_bool.ravel()

W_init[flat_mask, 0] = 1.0
W_init[~flat_mask, 1] = 1.0

# Normalize init rows to avoid weird scaling
row_sums = W_init.sum(axis=1, keepdims=True) + 1e-12
W_init = W_init / row_sums

# Run NMF multiple times with custom init

for run in range(N_RUNS):
    H_init = np.random.RandomState(run).rand(N_COMPONENTS, V.shape[1])

    model = NMF(
        n_components=N_COMPONENTS,
        init="custom",
        random_state=run,
        max_iter=MAX_ITER
    )
    W = model.fit_transform(V, W=W_init, H=H_init)

    fig, axes = plt.subplots(1, N_COMPONENTS, figsize=(12, 6))
    for c in range(N_COMPONENTS):
        basis = W[:, c].reshape(image.shape[0], image.shape[1])
        axes[c].imshow(basis, cmap="gray")
        axes[c].set_title(f"Mask-Prior NMF Run {run+1} • Component {c+1}")
        axes[c].axis("off")

    plt.tight_layout()
    plt.savefig(OUT_DIR / f"nmf_mask_prior_run_{run+1}.png", dpi=200, bbox_inches="tight")
    plt.show()