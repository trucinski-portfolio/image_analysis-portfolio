from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paths

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

IMAGE_PATH = DATA_DIR / "hallway.webp"

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Missing required image: {IMAGE_PATH}\n"
        "Place the file in the repo's data/ folder."
    )

# Robust line intersection (no slope/intercept)

def intersect_lines_p(l1, l2, w, h, min_angle_deg=5):
    """
    l1, l2: lines from HoughLinesP as (x1,y1,x2,y2)
    Returns (x,y) intersection if valid and within bounds; else None.
    """
    x1, y1, x2, y2 = map(float, l1)
    x3, y3, x4, y4 = map(float, l2)

    # Direction vectors
    v1 = np.array([x2 - x1, y2 - y1])
    v2 = np.array([x4 - x3, y4 - y3])

    # Filter near-parallel lines by angle
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    cosang = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    angle = np.degrees(np.arccos(abs(cosang)))
    if angle < min_angle_deg:
        return None

    # Solve intersection using line-line intersection formula
    # Based on determinant approach
    def det(a, b, c, d):
        return a * d - b * c

    denom = det(x1 - x2, y1 - y2, x3 - x4, y3 - y4)
    if abs(denom) < 1e-9:
        return None

    px = det(det(x1, y1, x2, y2), x1 - x2, det(x3, y3, x4, y4), x3 - x4) / denom
    py = det(det(x1, y1, x2, y2), y1 - y2, det(x3, y3, x4, y4), y3 - y4) / denom

    if 0 <= px <= w and 0 <= py <= h:
        return int(round(px)), int(round(py))
    return None

# Load + preprocess

image = cv2.imread(str(IMAGE_PATH))
if image is None:
    raise RuntimeError(f"cv2.imread failed for: {IMAGE_PATH}")

h, w = image.shape[:2]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Hough transform 
lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=100,
    minLineLength=120,
    maxLineGap=10
)

if lines is None or len(lines) < 2:
    raise RuntimeError("Not enough lines detected to estimate a vanishing point.")

line_image = image.copy()

# Draw detected lines
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Compute intersections robustly

intersections = []
lines_flat = [l[0] for l in lines]

for i in range(len(lines_flat)):
    for j in range(i + 1, len(lines_flat)):
        pt = intersect_lines_p(lines_flat[i], lines_flat[j], w=w, h=h, min_angle_deg=7)
        if pt is not None:
            intersections.append(pt)

if len(intersections) == 0:
    raise RuntimeError("No valid intersections found. Try lowering min_angle_deg or adjusting Hough params.")

# Robust center estimate (median beats mean)
xs = np.array([p[0] for p in intersections])
ys = np.array([p[1] for p in intersections])

center = (int(np.median(xs)), int(np.median(ys)))
print(f"Estimated vanishing point (median): {center} from {len(intersections)} intersections")

# Draw vanishing point
cv2.circle(line_image, center, 20, (255, 0, 0), -1)   # blue filled circle
cv2.circle(line_image, center, 30, (0, 0, 255), 3)    # red outline

# -----------------------------
# Plot results
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 8))

axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Hallway Image")
axes[0].axis("off")

axes[1].imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
axes[1].set_title("Hough Lines + Vanishing Point")
axes[1].axis("off")

# show intersection cloud as a scatter 
blank = np.zeros((h, w, 3), dtype=np.uint8)
for (x, y) in intersections[:: max(1, len(intersections)//2000)]: 
    blank[y, x] = (255, 255, 255)
blank[center[1], center[0]] = (0, 0, 255)

axes[2].imshow(blank)
axes[2].set_title("Intersection Cloud (downsampled)")
axes[2].axis("off")

plt.tight_layout()
plt.show()

# Save output (includes raw image)
out_path = OUT_DIR / "hallway_vanishing_point.png"
cv2.imwrite(str(out_path), line_image)