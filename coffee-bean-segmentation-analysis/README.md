# Coffee Bean Segmentation & Size Analysis (Image Processing)

This project segments and labels coffee beans in a microscopy-style .tif image using classical image processing:
- Otsu thresholding
- Distance transform + watershed segmentation
- Connected-component labeling and region properties (area)
- Size distribution analysis and probability calculations

## Methods (High level)
1. **Binarization (Otsu)**: convert grayscale to binary and invert so beans are foreground.
2. **Segmentation**: distance transform → Gaussian smoothing → local maxima markers → watershed.
3. **Quantification**: compute region areas, count objects, plot histogram.
4. **Inference**: compute mean area and probability that a bean is smaller than half the mean.

## Results
Example outputs are saved in `outputs/`:
- Binary image
- Segmented + labeled beans
- Area histogram
- Bean count + summary statistics

## How to run
> Note: the source image (`CoffeeBeans.tif`) is not included in this repository.

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python src/otsu_thresholding.py
python src/watershed_segmentation.py
python src/bean_size_distribution.py
python src/bean_size_probability.py