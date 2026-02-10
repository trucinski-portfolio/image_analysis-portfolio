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
if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Missing: {IMAGE_PATH}")

# Parameters

DOWNSAMPLE = 2          # Factor of 2 = every other pixel
N_COMPONENTS = 2        # H&E ~ 2 dominant stains (roughly)
N_RUNS = 3
MAX_ITER = 1000

# Load + downsample

image = io.imread(IMAGE_PATH)
image_small = image[::DOWNSAMPLE, ::DOWNSAMPLE, :]

# Flatten to V (pixels x channels)
V = image_small.reshape(-1, 3).astype(float)

# Optional normalization (helps NMF behave)
V = V / (V.max() + 1e-12)

# Run NMF multiple times

Ws, Hs = [], []
for seed in range(N_RUNS):
    model = NMF(
        n_components=N_COMPONENTS,
        init="random",
        random_state=seed,
        max_iter=MAX_ITER
    )
    W = model.fit_transform(V)
    H = model.components_
    Ws.append(W)
    Hs.append(H)

# Visualize components for each run

fig, axes = plt.subplots(N_RUNS, N_COMPONENTS, figsize=(10, 4 * N_RUNS))
for r in range(N_RUNS):
    for c in range(N_COMPONENTS):
        basis = Ws[r][:, c].reshape(image_small.shape[0], image_small.shape[1])
        axes[r, c].imshow(basis, cmap="gray")
        axes[r, c].set_title(f"NMF Run {r+1} • Component {c+1}")
        axes[r, c].axis("off")

plt.tight_layout()
plt.savefig(OUT_DIR / "nmf_random_init_components.png", dpi=200, bbox_inches="tight")
plt.show()

# Robust run-to-run comparison
# (accounts for component permutation)

def normalize_cols(W):
    norms = np.linalg.norm(W, axis=0) + 1e-12
    return W / norms

def best_perm_diff(Wa, Wb):
    """
    Compare two W matrices while allowing component swap.
    For 2 components, just check identity vs swap and take min.
    """
    WaN = normalize_cols(Wa)
    WbN = normalize_cols(Wb)

    # identity pairing
    d_id = np.mean(np.abs(WaN - WbN))

    # swapped pairing
    Wb_swap = WbN[:, ::-1]
    d_swap = np.mean(np.abs(WaN - Wb_swap))

    return min(d_id, d_swap)

for i in range(N_RUNS):
    for j in range(i + 1, N_RUNS):
        diff = best_perm_diff(Ws[i], Ws[j])
        print(f"Run {i+1} vs Run {j+1} diff (perm-invariant): {diff:.6f}")

print(f"W shape: {Ws[0].shape}, H shape: {Hs[0].shape}")