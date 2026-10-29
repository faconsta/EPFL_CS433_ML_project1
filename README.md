# EPFL Machine Learning (CS-433) — Project 1

## Team Members


* **Author 1**: Alan Liesenborghs (`alan.liesenborghs@epfl.ch`)

* **Author 2**: Noé Staeheli (`noe.staeheli@epfl.ch`)

* **Author 3**: Félix Constantin (`felix.constantin@epfl.ch`)

## Project Overview

This project tackles binary classification on the Behavioral Risk Factor Surveillance System (BRFSS) survey dataset, aiming to predict whether an individual has been diagnosed with Myocardial Infarction or Coronary Heart Disease (`MICHD`).

### Key Challenges & Strategy

* **Severe Class Imbalance**: Approximately 91.2% negative instances (`-1`) vs. 8.8% positive instances (`+1`). Accuracy is uninformative in this context; evaluation and optimization rely strictly on the $F_1$**-score**.

* **Pure NumPy Implementation**: All data cleaning, preprocessing, feature transformations, optimization methods, and threshold searches are built from scratch using standard Python and NumPy without external machine learning libraries.

* **Threshold Calibration**: Since raw linear scores cluster heavily in the negative domain, we decouple continuous model scoring from the decision boundary via grid-search optimization of the decision threshold $t^*$.

## Repository Structure

```
.
├── implementations.py       # The 6 mandatory ML optimization methods
├── preprocessing.py         # Data cleaning, NaN imputation, feature standardization, and threshold search
├── helpers.py               # Data loading and CSV submission export utilities
├── run.ipynb                # End-to-end execution notebook reproducing the final AIcrowd submission
├── report.pdf               # 2-page project report
├── .gitignore               # Excludes large CSV datasets and Python cache files
└── README.md                # Project overview and reproduction guidelines

```

## Dataset Setup

Due to GitHub's file size limit (100 MB), the raw datasets are not tracked in this repository.

1. Download the BRFSS CSV datasets (`x_train.csv`, `y_train.csv`, and `x_test.csv`).

2. Place all three `.csv` files directly in the root folder of this repository (alongside `run.ipynb` and `implementations.py`).

## Requirements

The project runs using standard Python 3 (3.8+) and NumPy:

```
pip install numpy jupyterlab

```

No external machine learning packages (such as `scikit-learn` or `pytorch`) are utilized.

## How to Reproduce the Submission

The submission is generated deterministically through `run.ipynb` (or an equivalent `run.py` script):

1. Launch your Jupyter environment:

   ```
   jupyter notebook run.ipynb
   
   ```

2. Set `sub_sample=False` in the data loading step to run on the full training set (328,135 samples).

3. Execute all cells (`Kernel -> Restart & Run All`).

4. The pipeline will:

   * Load the full training and test datasets.

   * Filter features with $>80\%$ missing values and impute remaining `NaN`s using training column means/medians.

   * Standardize continuous variables ($\mu = 0$, $\sigma = 1$) and prepend the bias intercept.

   * Train the regularized linear model (`ridge_regression` from `implementations.py`).

   * Calibrate the decision threshold ($t^* \approx -0.55$) to maximize the $F_1$-score.

   * Generate `submission_ridge.csv` ready for AIcrowd evaluation.

## Baseline Results Summary

| Model | Decision Threshold $t^*$ | Precision | Recall | Validation $F_1$ | 
| ----- | ----- | ----- | ----- | ----- | 
| Default Ridge ($t=0.0$) | $0.00$ | $0.5714$ | $0.0071$ | $0.0141$ | 
| **Optimized Ridge (**$\lambda = 10^{-4}$**)** | $-0.55$ | $0.3709$ | $0.5579$ | $0.4456$ | 
