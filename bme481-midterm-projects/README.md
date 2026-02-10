BME481 – Biological Image Analysis

This repository contains a curated set of biological and biomedical image analysis pipelines developed for BME 481 (Biological Image Analysis). The project emphasizes quantitative image processing, segmentation, feature extraction, and matrix factorization methods applied to microscopy, histology, and structured natural images.

The work focuses on interpretable, reproducible pipelines commonly used in biomedical research and diagnostics, including nuclei detection, spot quantification, tissue stain decomposition, and low-rank image representations.

```text
BME481-BIOLOGICAL-IMAGE-ANALYSIS/
├── data/                 # Input images (microscopy, histology, test images)
├── outputs/              # Generated figures, labeled images, and summaries
├── src/                  # Analysis scripts (modular, reproducible)
├── requirements.txt      # Python environment specification
├── README.md
└── .gitignore
```

Key Analysis Modules

1. Nuclei Detection via LoG + Otsu Thresholding

Script: nuclei_detection_log_otsu.py
	•	Laplacian of Gaussian (LoG) filtering for blob enhancement
	•	Otsu thresholding for adaptive binarization
	•	Connected-component labeling and size filtering
	•	Used for DAPI-stained nuclei detection

Outputs:
	•	Binary nuclei masks
	•	Labeled nuclei overlays
	•	Size-filtered detection results

---

2. Multi-Channel Nuclei & Spot Quantification

Script: multichannel_nuclei_spot_quantification.py
	•	Blue channel: nuclei segmentation (DAPI)
	•	Green channel: punctate signal detection
	•	Distance transform + watershed to separate merged spots
	•	Quantifies spots per nucleus, a common metric in cell signaling and FISH-like assays

Outputs:
	•	Watershed-separated spot labels
	•	Quantification summary (spots_per_nucleus)
	•	Visual overlays for validation

---

3. Watershed-Based Object Separation

Script: nuclei_labeling_size_filter.py
	•	Distance transform smoothing
	•	Marker-based watershed segmentation
	•	Size-based object filtering
	•	Used to resolve touching or clustered biological objects

---

4. Histology Stain Decomposition via NMF

Scripts:
	•	histology_nmf_random_init.py
	•	histology_nmf_mask_prior.py

Applies Non-Negative Matrix Factorization (NMF) to H&E-stained tissue images:
	•	Separates stain components (e.g., hematoxylin vs eosin)
	•	Demonstrates stability across random initializations
	•	Compares unconstrained vs mask-informed priors

Key Insight:
NMF provides a parts-based, interpretable decomposition of histological images, useful for stain normalization and tissue quantification.

---

5. Face Image Dictionary Learning & Compression

Script: face_nmf_bases_and_compression.py
	•	Learns NMF bases from a face image library
	•	Visualizes basis images as localized facial components
	•	Reconstructs held-out images using learned bases
	•	Quantifies compression ratio and reconstruction error

Demonstrates:
	•	Dictionary learning
	•	Low-rank approximation
	•	Trade-offs between compression and fidelity

---

6. Vanishing Point Detection (Hough Transform) // NOT PUSHED DUE TO IMAGE COPYRIGHT //

Script: vanishing_point_hough.py
	•	Edge detection + probabilistic Hough transform
	•	Line intersection analysis
	•	Estimates vanishing point in structured scenes

Included as a geometric image analysis example relevant to spatial inference.

---

Reproducibility

All analyses were developed and tested using a clean Python virtual environment.

To reproduce results:

```text
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

Outputs

The outputs/ directory contains:
	•	Labeled segmentation images
	•	Watershed results
	•	NMF basis visualizations
	•	Reconstruction comparisons
	•	Quantitative summaries (text)

These outputs serve as visual and numerical validation of each pipeline.

---

Scientific Focus

This project emphasizes:
	•	Interpretable image analysis methods
	•	Quantitative validation over heuristic visualization
	•	Relevance to microscopy, histology, and biomedical diagnostics
	•	Reproducible research practices

---

Author

Thomas Rucinski
M.S. Biomedical Engineering (Bioinformatics & Machine Learning focus)