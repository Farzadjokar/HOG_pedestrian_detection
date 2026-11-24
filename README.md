# HOG Pedestrian Detector

A classic computer-vision pedestrian detector built from scratch: **Histogram of Oriented Gradients (HOG)** features + a **linear SVM**, trained on the **PA-100k** dataset, with a simple **Tkinter** desktop app. Open an image and the app draws a bounding box around every pedestrian it finds, no settings required.

<p align="center">
  <img src="assets/demo.png" alt="Detection demo: the app drawing green bounding boxes around an adult and a child crossing a street" width="420">
</p>

*Demo: the app detects an adult and a child at a crosswalk. The number on each box is the SVM confidence score.*

## Features

- **HOG + LinearSVC** classifier (64x128 window, 3780-dimensional descriptor)
- **Trained on PA-100k** pedestrian images
- **Automatic negatives:** background images are downloaded from the INRIA person dataset on Hugging Face (`marcelarosalesj/inria-person`)
- **Image pyramid** detection, so pedestrians of any size are found (small images are enlarged, large pedestrians are found on shrunk levels)
- **Tkinter app** that loads the model automatically and detects as soon as you open an image
- Detection runs in a background thread, so the window never freezes

## How it works

**Training (`train_classifier.py`)**

1. Reads PA-100k pedestrian crops (`train.csv`, `val.csv`, `test.csv`) as positives, resized to 64x128 and horizontally flipped for augmentation.
2. Downloads person-free images from the INRIA dataset on Hugging Face and samples random windows as negatives (split by image into train / val / test to avoid leakage).
3. Reduces false positives with:
   - grayscale HOG, so colour cannot separate the two datasets
   - blur, JPEG and contrast augmentation applied to both classes (PA-100k is low-resolution, INRIA negatives are sharp)
   - partial-body and wrong-scale windows cut from PA-100k as extra negatives
   - several rounds of **hard-negative mining**
4. Trains a `LinearSVC`, prints validation and test reports, and saves the model.

**Detection (`pedestrian_app.py`)**

1. Converts the image to grayscale and builds an image pyramid.
2. Slides the 64x128 HOG window over every level using OpenCV's `HOGDescriptor` with the trained SVM weights.
3. Maps hits back to the original image and groups overlapping hits. A pedestrian needs several overlapping hits, which removes most isolated false positives.
4. Draws a green box and the SVM score for each pedestrian.

## Project structure

```
.
├── train_classifier.py        # training script
├── pedestrian_app.py          # Tkinter detection app
├── pedestrian_hog_svm.npz     # trained weights (created by training)
├── pedestrian_hog_svm.joblib  # full scikit-learn model (created by training)
├── requirements.txt
├── assets/
│   └── demo.png
├── data/                      # PA-100k images (not in the repo)
├── train.csv / val.csv / test.csv / annotation.mat   # PA-100k (not in the repo)
└── inria_negatives/           # created automatically on first training run
```

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

Tkinter ships with Python on Windows and macOS. On Debian/Ubuntu: `sudo apt install python3-tk`.

## Dataset setup

Download **PA-100k** yourself (it is not included here) and place the files in the project folder:

```
train.csv
val.csv
test.csv
annotation.mat
data/
    000001.jpg
    000002.jpg
    ...
```

The training script finds `train.csv` in the current folder (or in a `PA-100k/` folder next to the script), or you can pass `--root /path/to/PA-100k`.

## Train

```bash
python train_classifier.py
```

This downloads the negative images on the first run (cached in `inria_negatives/`), trains the classifier, and writes `pedestrian_hog_svm.npz` and `pedestrian_hog_svm.joblib` to the current folder.

Useful options:

| Option | Default | Description |
| --- | --- | --- |
| `--root` | auto | PA-100k folder |
| `--neg_dir` | none | use your own folder of person-free images instead of Hugging Face |
| `--max_train` | 20000 | max positive training images (`0` = all) |
| `--neg_per_image` | 40 | random windows sampled per negative image |
| `--partial_ratio` | 0.5 | partial / misaligned negatives, as a fraction of positives |
| `--mine_rounds` | 3 | hard-negative mining rounds |
| `--C` | 0.01 | SVM regularisation |
| `--no_flip` | off | disable horizontal-flip augmentation |

## Run the app

```bash
python pedestrian_app.py
```

Click **Open Image**. The model is loaded automatically from the script's folder and detection starts immediately. **Save Result** stores the image with the boxes drawn on it.

### Tuning

Constants at the top of `pedestrian_app.py`:

| Constant | Effect |
| --- | --- |
| `SCORE_THR` | higher = fewer false positives, lower = fewer missed people |
| `MIN_NEIGHBORS` | overlapping hits required to accept a pedestrian |
| `PYRAMID_STEP` | smaller = more pyramid levels (slower, finer scale coverage) |
| `DETECT_MAX_SIDE` | larger = finds smaller people in big images (slower) |

## Results

Add your own numbers here after training (the script prints a confusion matrix and precision / recall for validation and test).

| Split | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Validation | | | |
| Test | | | |

## Limitations

- HOG + SVM is a 2005-era method. It works best on upright, reasonably sharp, fully visible pedestrians and struggles with heavy blur, crowds, occlusion, unusual poses and top-down camera angles.
- PA-100k contains only cropped pedestrians, so the quality of the negative images strongly affects the number of false positives.
- Reported window-level accuracy is not the same as detection performance on full images. Judge the detector on real photos too.
- For production use, modern CNN detectors (YOLO, SSD, Faster R-CNN) are far more accurate.

## References

- N. Dalal and B. Triggs, *Histograms of Oriented Gradients for Human Detection*, CVPR 2005.
- X. Liu et al., *HydraPlus-Net: Attentive Deep Features for Pedestrian Analysis*, ICCV 2017 (PA-100k).
- INRIA Person Dataset, via Hugging Face: `marcelarosalesj/inria-person`.

Please check the licences and terms of use of PA-100k and the INRIA dataset before redistributing any data or trained models.

## License

Add your licence here (for example MIT).