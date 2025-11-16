# Machine_Learning_Cancer_Detection
Bachelor thesis 2025/2026 - Data Engineering

Cancer Detection program which allows users to find the most optimal algorithm for thew cancer detection

Comments will be made within the code. Only more detailed exlanations will be held within the theoretical paper: [Bachelor thesis - Cancer Detection ML scientific tool](https://pgedupl-my.sharepoint.com/:w:/g/personal/s193242_student_pg_edu_pl/EaFnt6AP4u1JsutwymB5L6UBMfDO7X3jm2la7NecYFdIhg?e=pPljy9)

## How to Run the Program

To run the program, navigate to the `src` directory and use the `main.py` script with the required arguments. Below is an example showcasing the full functionality of the program. Paste the example into your terminal:

### Example: Full Pipeline with RFECV, kbest, corr, Logistic Regression, Threshold Tuning, and XAI

```bash
python .\main.py `
  --data "data_sources\liquid_biopsy_data.csv" `
  --use_validation "separate" `
  --preprocess "['corr','kbest', 'rfecv']" `
  --preprocess_params "{'corr_threshold': 0.9, 'prefilter_k': 1000}" `
  --model SVM `
  --model_params "{'kernel': 'linear', 'C': 1.0, 'use_calibrated': True, 'cv_calibration': 5}" `
  --postprocess `
  --postprocess_params "{'tuning_method': 'f1'}" `
  --eval "['accuracy','F1','recall','precision','AUC ROC']" `
  --xai `
  --xai_sample 215
```

### Explanation of Arguments:
- `--data`: Path to the dataset (e.g., `data_sources\liquid_biopsy_data.csv`). The dataset must include split columns (`isTraining`, `isValidation`, `isTest`) and a target column (`cancer`).
- `--use_validation`: Validation strategy:
  - `separate`: Keeps train/validation/test separate (default). Required for `--postprocess`.
  - `merge_train_test`: Merges 80% of validation into train and 20% into test.
- `--preprocess`: List of preprocessing steps as a Python list. Available methods:
  - `'corr'`: Correlation removal (removes highly correlated features).
  - `'kbest'`: SelectKBest prefiltering using ANOVA F-value.
  - `'rfecv'`: Recursive Feature Elimination with Cross-Validation.
- `--preprocess_params`: Parameters for preprocessing as a Python dictionary:
  - `corr_threshold` (float, default=0.90): Pearson correlation threshold.
  - `prefilter_k` (int, default=1500): Number of features for SelectKBest.
- `--model`: Machine learning model to use. Options:
  - `LogisticRegression` (LR)
  - `DecisionTree` (DT)
  - `SVM` (Support Vector Machine)
  - `DNN` (Deep Neural Network)
- `--model_params`: Model-specific hyperparameters as a Python dictionary. Examples:
  - Logistic Regression: `{'max_iter': 1000, 'C': 1.0, 'solver': 'lbfgs'}`
  - Decision Tree: `{'max_depth': 6, 'min_samples_leaf': 5}`
  - SVM: `{'kernel': 'linear', 'C': 1.0}`
  - DNN: `{'hidden_layers': [128, 64], 'activation': 'relu', 'epochs': 50}`
- `--postprocess`: Enables threshold tuning (requires `--use_validation separate`).
- `--postprocess_params`: Parameters for threshold tuning as a Python dictionary:
  - `tuning_method` (default=`recall`): Options include `recall`, `f1`, `youden`, etc.
  - `show_tuning_plots` (default=`True`): Whether to display optimization plots.
- `--eval`: List of evaluation metrics as a Python list. Available metrics:
  - `accuracy`, `precision`, `recall`, `F1`, `AUC ROC`, `AUC PR`, `Confusion Matrix`.
- `--xai`: Enables Explainable AI (XAI) analysis using SHAP.
- `--xai_sample`: Index of the sample in a test set for detailed XAI analysis. Generates a SHAP waterfall plot for the specified sample.

This example demonstrates the full pipeline, including feature selection, model training, threshold tuning, evaluation, and explainability analysis.

