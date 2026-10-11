import numpy as np


### Data preprocessing and post-processing utilities ###

def preprocess_data(x_train, x_test, nan_threshold=0.8):
  """Cleans and standardizes features using strictly training statistics."""
  # 1. Drop columns with too many NaNs
  nan_ratio = np.isnan(x_train).mean(axis=0)
  valid_cols = nan_ratio < nan_threshold
  x_train_filt = x_train[:, valid_cols]
  x_test_filt = x_test[:, valid_cols]

  # 2. Impute NaNs with column means of training set
  train_means = np.nanmean(x_train_filt, axis=0)
  train_means = np.nan_to_num(train_means, nan=0.0)

  x_train_imp = np.where(np.isnan(x_train_filt), train_means, x_train_filt)
  x_test_imp = np.where(np.isnan(x_test_filt), train_means, x_test_filt)

  # 3. Drop constant columns
  stds = np.std(x_train_imp, axis=0)
  non_constant = stds > 1e-8
  x_train_clean = x_train_imp[:, non_constant]
  x_test_clean = x_test_imp[:, non_constant]

  # 4. Standardize to mean 0, variance 1
  means = np.mean(x_train_clean, axis=0)
  stds_clean = np.std(x_train_clean, axis=0)
  safe_stds = np.where(stds_clean == 0, 1.0, stds_clean)

  x_train_std = (x_train_clean - means) / safe_stds
  x_test_std = (x_test_clean - means) / safe_stds

  return x_train_std, x_test_std


def add_bias_term(x):
  """Adds an intercept column of ones."""
  return np.hstack((np.ones((x.shape[0], 1)), x))


def find_best_threshold(y_true, scores):
  """Scans thresholds to maximize the F1-score on {-1, 1} targets."""
  thresholds = np.linspace(-0.90, -0.40, 100)
  best_threshold = -0.55
  best_f1 = 0.0

  for t in thresholds:
    y_pred = np.where(scores >= t, 1, -1)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))

    denom = 2 * tp + fp + fn
    f1 = (2 * tp) / denom if denom > 0 else 0.0

    if f1 > best_f1:
      best_f1 = f1
      best_threshold = t

  return best_threshold, best_f1