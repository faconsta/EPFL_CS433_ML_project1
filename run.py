import numpy as np
from helpers import create_csv_submission, load_csv_data
from implementations import ridge_regression
from preprocessing import preprocess_data, add_bias_term, find_best_threshold



### Load the dataset ###

x_train, x_test, y_train, train_ids, test_ids = load_csv_data(
    data_path=".", sub_sample=False
)

print(f"Train samples: {x_train.shape[0]}, Features: {x_train.shape[1]}")
print(f"Test samples:  {x_test.shape[0]}")



### Proprocess and add intercept ###

# 1. Clean, impute, standardize
x_train_std, x_test_std = preprocess_data(x_train, x_test, nan_threshold=0.8)

# 2. Add the intercept (column of 1s)
tx_train = add_bias_term(x_train_std)
tx_test = add_bias_term(x_test_std)

print(f"Features after preprocessing (with bias): {tx_train.shape[1]}")



### train ridge regression and optimize threshold ###

# 1. Train Ridge Regression
lambda_ = 1e-4
w_ridge, loss = ridge_regression(y_train, tx_train, lambda_)
print(f"Ridge training complete! Training MSE loss: {loss:.4f}")

# 2. Get continuous training scores
train_scores = tx_train @ w_ridge

# 3. Find optimal decision cutoff
best_thresh, best_f1 = find_best_threshold(y_train, train_scores)
print(f"Best decision threshold: {best_thresh:.4f}")
print(f"Best training F1-score:   {best_f1:.4f}")



### Generate AIcrowd submission file ###

# 1. Compute scores on the test set
test_scores = tx_test @ w_ridge

# 2. Classify using the optimal threshold
y_test_pred = np.where(test_scores >= best_thresh, 1, -1)

# 3. Check predicted positive ratio (should be roughly ~8% to 15%)
pos_rate = (np.sum(y_test_pred == 1) / len(y_test_pred)) * 100
print(f"Percentage of positive predictions: {pos_rate:.2f}%")

# 4. Save CSV file for AIcrowd
create_csv_submission(test_ids, y_test_pred, "submission_ridge.csv")
print("Submission saved successfully as 'submission_ridge.csv'!")