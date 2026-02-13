# Image Analysis Portfolio

A collection of image analysis projects demonstrating classical and modern techniques for biomedical and natural image processing. Built with Python using scikit-image, OpenCV, scikit-learn, and scipy.

## Projects

### BME481 Biological Image Analysis

Coursework from BME481 (Biological Image Analysis) covering nuclei detection, histological stain decomposition, spot quantification, facial dictionary learning, and vanishing point estimation.

**Nuclei Detection & Quantification**

- **Nuclei Detection (LoG + Otsu)** — Detects nuclei in DAPI-stained fluorescence microscopy images using Laplacian of Gaussian filtering and Otsu thresholding, followed by connected-component labeling and size-based filtering.
- **Multichannel Spot Quantification** — Segments nuclei and punctate signal spots (FISH-like assay) across separate color channels, then quantifies spots per nucleus using watershed segmentation and peak detection.

**Histology Stain Decomposition (NMF)**

- **Random Initialization** — Decomposes H&E-stained brain histology into hematoxylin and eosin components using Non-Negative Matrix Factorization, with stability analysis across multiple random seeds.
- **Mask-Prior Initialization** — Uses an anatomical tissue mask to guide NMF initialization, constraining component decomposition to biologically meaningful regions.

**Facial Expression Dictionary Learning**

- **NMF Basis Extraction & Compression** — Learns 50 basis images from a facial pain expression dataset using NMF, then evaluates image reconstruction quality and compression ratios.

**Geometric Analysis**

- **Vanishing Point Detection** — Estimates the vanishing point in perspective scenes using Canny edge detection, probabilistic Hough line transform, and robust median-based intersection estimation.

### Coffee Bean Segmentation & Size Analysis

A classical image processing pipeline for segmenting, counting, and statistically analyzing coffee beans in microscopy-style images.

- **Otsu Thresholding** — Converts grayscale bean images to binary masks using automatic Otsu thresholding.
- **Watershed Segmentation** — Separates touching beans using distance transforms, Gaussian smoothing, local maxima detection, and watershed segmentation with area-based filtering.
- **Size Distribution Analysis** — Extracts region properties to generate bean area histograms and probability density plots, including statistical measures (mean area, probability of undersized beans).

## Tech Stack

| Category | Libraries |
|---|---|
| Image Processing | scikit-image, OpenCV, scipy.ndimage |
| Machine Learning | scikit-learn (NMF) |
| Scientific Computing | NumPy, SciPy |
| Visualization | Matplotlib, Pillow |
| Image I/O | tifffile, imageio |

## Getting Started

Each project has its own virtual environment and dependencies. To run a project:

```bash
cd <project-directory>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/<script_name>.py
```

Output images and summaries are saved to the `outputs/` directory within each project.

## Repository Structure

```
.
├── bme481-midterm-projects/
│   ├── data/              # Microscopy, histology, and facial expression images
│   ├── outputs/           # Generated visualizations and quantitative summaries
│   ├── src/               # Analysis scripts (7 modules)
│   └── requirements.txt
├── coffee-bean-segmentation-analysis/
│   ├── src/               # Segmentation and analysis scripts (4 modules)
│   ├── outputs/           # Segmentation results and statistical plots
│   └── requirements.txt
└── README.md
```
