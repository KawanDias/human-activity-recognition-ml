# Human Activity Recognition using Machine Learning

**A machine learning project for human movement classification using triaxial accelerometer data, signal processing, and Random Forest.**

Developed as an academic project at **IFPR — Instituto Federal do Paraná**, this project explores how accelerometer signals can be processed, transformed into meaningful features, and classified to recognize different human activities.

---

## Overview

Human Activity Recognition (HAR) is the process of identifying physical activities from sensor data. This project implements a classification pipeline that analyzes accelerometer signals and distinguishes between three activity classes:

- **Rest**
- **Put Away Object**
- **Turn Page**

The approach combines time-series preprocessing, feature engineering, frequency-domain analysis, dimensionality reduction, and supervised machine learning.

## Objectives

- Process and standardize triaxial accelerometer signals.
- Extract meaningful features from time and frequency domains.
- Explore feature distributions using dimensionality reduction.
- Train and evaluate a Random Forest classifier.
- Investigate activity classification using a continuous data stream.

## Methodology

The project follows a signal-processing and machine-learning pipeline:

1. **Data preprocessing:** Accelerometer signals are resampled and interpolated to standardize the time series.
2. **Window segmentation:** Signals are divided into one-second windows.
3. **Feature extraction:** A total of 44 features are extracted from the time and frequency domains.
4. **Frequency analysis:** The Fast Fourier Transform (FFT) is used to analyze the frequency characteristics of the signals.
5. **Dimensionality reduction:** UMAP is used to explore the distribution of the extracted features.
6. **Model training:** A Random Forest classifier learns to distinguish the activity classes.
7. **Evaluation:** The model is evaluated using a stratified holdout test set.
8. **Continuous inference:** An experiment investigates classification using a continuous stream of accelerometer data.

## Dataset

The experimental dataset contains **300 samples**, equally distributed across the three activity classes.

| Property | Value |
|---|---:|
| Total samples | 300 |
| Activity classes | 3 |
| Samples per class | 100 |
| Window duration | 1 second |
| Extracted features | 44 |
| Training set | 70% |
| Validation set | 15% |
| Test set | 15% |

A stratified split is used to preserve the class distribution across the training, validation, and test sets.

## Results

The Random Forest classifier achieved **97.78% accuracy** on the holdout test set.

| Evaluation metric | Result |
|---|---:|
| Test samples | 45 |
| Correct predictions | 44 |
| Incorrect predictions | 1 |
| Test accuracy | 97.78% |

These results indicate that the extracted features were effective at distinguishing the three activity classes within the experimental dataset.

The project also explores inference on a continuous data stream as an initial investigation into real-time activity recognition.

> **Note:** The reported accuracy applies to the experimental test set. Further evaluation with larger datasets, different users, and varying sensor conditions would be necessary to assess generalization to real-world scenarios.

## Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical computing |
| SciPy | Signal processing |
| FFT | Frequency-domain analysis |
| UMAP | Dimensionality reduction and feature visualization |
| Random Forest | Supervised classification |

## Key Concepts

- Human Activity Recognition (HAR)
- Triaxial accelerometer data
- Time-series preprocessing
- Signal resampling and interpolation
- Sliding-window analysis
- Feature engineering
- Time- and frequency-domain analysis
- Fast Fourier Transform (FFT)
- Dimensionality reduction
- Supervised machine learning
- Model evaluation
- Continuous-stream inference

## Project Structure

The repository structure will depend on the implementation files included in the project. A possible organization is:

```text
human-activity-recognition-ml/
├── data/              # Experimental dataset
├── notebooks/         # Data analysis and experiments
├── src/               # Processing and classification code
├── results/           # Evaluation metrics and visualizations
├── requirements.txt   # Python dependencies
└── README.md
```

*This is a suggested structure; the directories should be adjusted to match the actual repository.*

## Getting Started

### Prerequisites

- Python 3.x
- pip

### Installation

Clone the repository:

```bash
git clone https://github.com/KawanDias/human-activity-recognition-ml.git
cd human-activity-recognition-ml
```

Create and activate a virtual environment:

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

> The installation and execution instructions should be updated to match the dependency list and entry-point scripts available in the repository.

## Limitations and Future Work

Potential directions for further development include:

- Expanding the dataset with additional activity classes.
- Collecting more samples under different conditions.
- Comparing Random Forest with alternative classification algorithms.
- Evaluating generalization across different users and sensor placements.
- Investigating robustness under sensor noise and variations in movement.
- Further developing continuous-stream inference for real-time applications.

## Academic Context

This project was developed as part of an **Artificial Intelligence course** at **IFPR — Instituto Federal do Paraná**, in October 2026.

**Author:** Kawan Dias

---

*Exploring the intersection of signal processing and machine learning to recognize human movement patterns.*
