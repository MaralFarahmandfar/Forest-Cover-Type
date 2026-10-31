# Forest Cover Type Classification with Deep Neural Networks

A hands-on project where I apply the techniques from the first two courses of Andrew Ng's Deep Learning Specialization (regularization, optimization, batch normalization, hyperparameter tuning) to a real, large, imbalanced multi-class problem, using TensorFlow/Keras.

> **Status:** work in progress. Data exploration and baseline models are done; neural network experiments are next.

## Problem

Predict the forest cover type (7 classes) of a 30m x 30m patch of land from cartographic variables such as elevation, slope, distance to water/roads, and soil type.

- **Dataset:** [UCI Covertype](https://archive.ics.uci.edu/dataset/31/covertype) (581,012 samples, 54 features)
- **Features:** 10 numeric, 4 binary wilderness-area columns, 40 binary soil-type columns
- **Target:** `Cover_Type`, classes 1 to 7

## Goals

1. Build a fair comparison between classic baselines and a deep neural network.
2. Measure the effect of each training technique with a clean ablation study, instead of reporting only one final model.
3. Handle class imbalance correctly (stratified split, macro F1, per-class recall).
4. Serve the final model through a small API.

## Findings so far

### Data exploration (`01_data_exploration`)

- The data is clean: no missing values, all columns are integers.
- The classes are **strongly imbalanced**. Class 2 is 48.8% and class 1 is 36.5% of the data, while class 4 is only 0.47%. A model that always predicts class 2 already gets about 49% accuracy, so accuracy alone is not a good metric here.
- The numeric features have very different scales (for example `Elevation` is in the thousands while `Slope` is below 70), so the 10 numeric features are standardized. The binary columns are left unchanged.

![Class distribution](results/class_distribution.png)

### Baselines (`02_baselines`)

Data split: 80% train, 10% dev, 10% test, stratified by class, with a fixed seed. The scaler is fit on the training set only. All numbers below are on the **dev set**; the test set is not used until the final evaluation.

| Model | Accuracy | Macro F1 |
|---|---|---|
| Majority class | 0.488 | 0.094 |
| Logistic Regression | 0.725 | 0.526 |
| Random Forest | 0.954 | 0.923 |

- The majority-class model gets 48.8% accuracy without learning anything, but its macro F1 is only 0.09. This is why macro F1 and per-class recall are the main metrics.
- Logistic Regression (a linear model, like a network without hidden layers) fails on the small classes. For example, 96% of class 5 is predicted as class 2, and 49% of class 6 is predicted as class 3.
- Random Forest is a strong reference for this tabular problem. Its weakest classes are 5 (recall 0.77) and 4 (recall 0.83).
- The neural network should clearly beat Logistic Regression. Getting close to the Random Forest macro F1 is the target, and I will report honestly if it does not.

Confusion matrices (rows are true classes, normalized so the diagonal is the recall of each class):

| Logistic Regression | Random Forest |
|---|---|
| ![Logistic Regression confusion matrix](results/cm_logreg.png) | ![Random Forest confusion matrix](results/cm_rf.png) |

## Roadmap

- [x] `01_data_exploration`: data quality, class distribution, feature scales
- [x] `02_baselines`: label shift (1-7 to 0-6), stratified train/dev/test split, normalization, majority-class baseline, Logistic Regression, Random Forest
- [ ] `03_nn_experiments`: ablation study with Keras
  - [ ] Network depth and He initialization
  - [ ] Regularization: none vs. L2 vs. Dropout
  - [ ] Optimizers: SGD vs. Momentum vs. Adam, and mini-batch sizes
  - [ ] Learning rate decay
  - [ ] Batch Normalization
  - [ ] Random search over hyperparameters
- [ ] Final evaluation on the test set: macro F1, per-class recall, confusion matrix, error analysis
- [ ] FastAPI endpoint for the final model

## Project structure

```
.
├── data/
│   ├── covtype.data.gz    # raw dataset (not tracked by git)
│   └── processed/         # arrays created by notebook 02 (not tracked by git)
├── notebooks/             # numbered notebooks, one task each
├── src/
│   ├── data.py            # data loading and column names
│   └── evaluate.py        # metrics and confusion matrix plots
├── results/               # figures and result tables
├── models/                # saved scaler and models (not tracked by git)
└── requirements.txt       # main dependencies (pinned versions)
```

## Setup

The project uses **Python 3.11** (needed for the TensorFlow version used here).

```bash
# 1. Create and activate a virtual environment
# Windows:
py -3.11 -m venv venv
venv\Scripts\activate
# Linux / macOS:
python3.11 -m venv venv
source venv/bin/activate

# 2. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Data

The dataset is not included in the repository.

1. Download the Covertype dataset from the [UCI page](https://archive.ics.uci.edu/dataset/31/covertype).
2. Extract the zip and put the file `covtype.data.gz` inside the `data/` folder. Do not unzip the `.gz` file; pandas reads it directly.

The raw file has no header, so the column names are defined in `src/data.py` following the UCI documentation.

## How to run

Run the notebooks in order. Notebook `02_baselines` creates the processed data in `data/processed/` and the fitted scaler in `models/`, which the next notebooks load. Create the `data/processed/` and `models/` folders first if they do not exist.

## Reproducibility

A fixed random seed (`SEED = 42`) is used for all splits and training runs. Package versions are pinned in `requirements.txt`.

## Dataset citation

Blackard, J. (1998). *Covertype*. UCI Machine Learning Repository. https://doi.org/10.24432/C50K5N