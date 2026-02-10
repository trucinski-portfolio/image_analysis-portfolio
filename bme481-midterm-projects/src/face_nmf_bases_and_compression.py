import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.decomposition import NMF

# Paths

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "pain_crops"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Missing directory: {DATA_DIR}\n"
        "Place the pain_crops folder under data/."
    )

# Parameters

DOWNSAMPLE = 2
N_COMPONENTS = 50
RANDOM_STATE = 0
MAX_ITER = 1000
TARGET_FILENAME = "f1a2.jpg"  # image to reconstruct

# Load images

image_files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(".jpg")])
if len(image_files) == 0:
    raise RuntimeError(f"No .jpg files found in {DATA_DIR}")

faces = []
shapes = []

for fname in image_files:
    img = io.imread(DATA_DIR / fname, as_gray=True).astype(float)

    # Normalize to [0,1] if needed
    if img.max() > 1.0:
        img = img / 255.0

    img_ds = img[::DOWNSAMPLE, ::DOWNSAMPLE]
    faces.append(img_ds)
    shapes.append(img_ds.shape)

# Enforce consistent shape 
h, w = shapes[0]
if not all(s == (h, w) for s in shapes):
    raise ValueError("Not all images have the same shape after downsampling. "
                     "Resize/crop them to a consistent size before NMF.")

faces_matrix = np.stack([f.reshape(-1) for f in faces], axis=0)  # (n_images, n_pixels)

# Fit NMF

nmf = NMF(
    n_components=N_COMPONENTS,
    init="random",
    random_state=RANDOM_STATE,
    max_iter=MAX_ITER
)
W = nmf.fit_transform(faces_matrix)      # (n_images, n_components)
H = nmf.components_                      # (n_components, n_pixels)

recon = W @ H
recon_error = np.linalg.norm(faces_matrix - recon)
print(f"Reconstruction Error (Frobenius norm): {recon_error:.4f}")

# Visualize basis images (H rows)

def plot_basis_grid(H, h, w, start, count, out_name):
    n = int(np.ceil(np.sqrt(count)))
    fig, axes = plt.subplots(n, n, figsize=(12, 12))
    axes = np.array(axes).ravel()

    for idx in range(n * n):
        ax = axes[idx]
        comp = start + idx
        if comp < start + count and comp < H.shape[0]:
            basis_img = H[comp].reshape(h, w)
            ax.imshow(basis_img, cmap="gray")
            ax.set_title(f"Basis {comp+1}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUT_DIR / out_name, dpi=200, bbox_inches="tight")
    plt.show()

plot_basis_grid(H, h, w, start=0,  count=25, out_name="nmf_bases_1_25.png")
plot_basis_grid(H, h, w, start=25, count=25, out_name="nmf_bases_26_50.png")

# Reconstruct a specific image

target_path = DATA_DIR / TARGET_FILENAME
if not target_path.exists():
    raise FileNotFoundError(f"Target file not found: {target_path}")

target = io.imread(target_path, as_gray=True).astype(float)
if target.max() > 1.0:
    target = target / 255.0
target_ds = target[::DOWNSAMPLE, ::DOWNSAMPLE]

if target_ds.shape != (h, w):
    raise ValueError(f"Target image shape {target_ds.shape} != expected {(h,w)}")

target_vec = target_ds.reshape(1, -1)
target_w = nmf.transform(target_vec)
target_recon = (target_w @ H).reshape(h, w)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(target_ds, cmap="gray")
ax[0].set_title(f"Original: {TARGET_FILENAME}")
ax[0].axis("off")

ax[1].imshow(target_recon, cmap="gray")
ax[1].set_title("Reconstruction (NMF)")
ax[1].axis("off")

plt.tight_layout()
plt.savefig(OUT_DIR / "nmf_reconstruction_compare.png", dpi=200, bbox_inches="tight")
plt.show()

# Compression analysis (honest)

n_images = faces_matrix.shape[0]
n_pixels = faces_matrix.shape[1]

total_original_elements = n_images * n_pixels
total_nmf_elements = (n_images * N_COMPONENTS) + (N_COMPONENTS * n_pixels)
compression_ratio_elements = total_nmf_elements / total_original_elements

# Approx bytes (assume original stored as uint8, NMF stored as float32)
original_bytes = total_original_elements * 1
nmf_bytes = total_nmf_elements * 4
compression_ratio_bytes = nmf_bytes / original_bytes

print("\n--- Compression ---")
print(f"Original: {total_original_elements} elements (~{original_bytes/1e6:.2f} MB if uint8)")
print(f"NMF: {total_nmf_elements} elements (~{nmf_bytes/1e6:.2f} MB if float32)")
print(f"Compression ratio (elements): {compression_ratio_elements:.4f}")
print(f"Compression ratio (bytes, uint8 vs float32): {compression_ratio_bytes:.4f}")
print(f"Data saved (elements): {(1 - compression_ratio_elements) * 100:.2f}%")