import argparse
def parse_args(argv=None):
    description = """
    Bachelor thesis - Cancer Detection ML scientific tool 2025
=========================================================
A comprehensive machine learning pipeline for cancer detection using gene expression data.
Supports multiple models, feature selection methods, threshold optimization, and explainable AI.
"""
 
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=True
    )
    
    data_group = p.add_argument_group('Data & Validation Strategy')
    data_group.add_argument("--data",required=True,
        help="Path to CSV file with gene expression data. Must contain split columns "
             "(isTraining, isValidation, isTest) and target column (cancer)."
    )
    data_group.add_argument("--use_validation",default="separate",choices=["separate", "merge_train_test"],
        help="Validation set handling strategy:\n"
             "  separate: Keep train/validation/test separate (default)\n"
             "              Required for --postprocess (threshold tuning)\n"
             "  merge_train_test: Merge 80%% of validation into train, 20%% into test\n"
             "                      Maximizes training data, disables threshold tuning"
    )
    preprocess_group = p.add_argument_group('Preprocessing & Feature Selection')
    preprocess_group.add_argument("--preprocess",default="[]",
        help="List of preprocessing steps as Python list. Available methods:\n"
             "   'corr' - Correlation removal (removes highly correlated features)\n"
             "   'kbest' - SelectKBest prefiltering using ANOVA F-value\n"
             "   'rfecv' - Recursive Feature Elimination with Cross-Validation\n"
             "Example: \"['corr','kbest']\" or \"['rfecv','corr','kbest']\""
    )
    preprocess_group.add_argument("--preprocess_params",default="{}",
        help="Preprocessing parameters as Python dict:\n"
             "   corr_threshold (float, default=0.90): Pearson correlation threshold\n"
             "   prefilter_k (int, default=1500): Number of features for SelectKBest\n"
             "Example: \"{'corr_threshold': 0.85, 'prefilter_k': 1000}\""
    )
    model_group = p.add_argument_group('Model Configuration')
    model_group.add_argument("--model", required=True, choices=["DecisionTree", "DT", "LogisticRegression", "LR", "SVM", "SVC", "DNN"],
        help="Machine learning model:\n"
             "   DT (Decision Tree)\n"
             "   LR (Logistic Regression)\n"
             "   SVM (Support Vector Machine)\n"
             "   DNN (Deep Neural Network)"
    )
    model_group.add_argument("--model_params", default="{}",
        help="Model-specific hyperparameters as Python dict:\n\n"
             "Decision Tree (DT):\n"
             "   max_depth (int): Maximum tree depth (None for unlimited)\n"
             "   min_samples_leaf (int, default=1): Minimum samples in leaf node\n"
             "   criterion ('gini'|'entropy', default='gini'): Split quality measure\n"
             "  Example: \"{'max_depth': 6, 'min_samples_leaf': 5}\"\n\n"
             "Logistic Regression (LR):\n"
             "   max_iter (int, default=100): Maximum optimization iterations\n"
             "   C (float, default=1.0): Inverse regularization strength (smaller = stronger)\n"
             "   solver ('lbfgs'|'saga'|'liblinear', default='lbfgs'): Optimization algorithm\n"
             "   penalty ('l1'|'l2'|'none', default='l2'): Regularization type\n"
             "  Example: \"{'max_iter': 1000, 'C': 0.5, 'solver': 'lbfgs'}\"\n\n"
             "SVM:\n"
             "   kernel ('linear'|'rbf'|'poly', default='linear'): Kernel function\n"
             "   C (float, default=1.0): Regularization parameter\n"
             "   gamma ('scale'|'auto'|float, default='scale'): Kernel coefficient (rbf/poly)\n"
             "   use_calibrated (bool, default=True): Use CalibratedClassifierCV (linear only)\n"
             "  Example: \"{'kernel': 'linear', 'C': 1.0, 'use_calibrated': True}\"\n\n"
             "Deep Neural Network (DNN):\n"
             "   hidden_layers (list[int]): Units in hidden layers (e.g., [128, 64])\n"
             "   activation (str, default='relu'): Activation function for hidden layers\n"
             "   dropout_rate (float, default=0.2): Dropout rate\n"
             "   learning_rate (float, default=0.001): Learning rate for Adam optimizer\n"
             "   epochs (int, default=50): Number of training epochs\n"
             "   batch_size (int, default=32): Batch size\n"
             "  Example: \"{'hidden_layers': [128, 64], 'activation': 'relu', 'epochs': 100}\""
    )
    postprocess_group = p.add_argument_group('Postprocessing (Threshold Tuning)')
    postprocess_group.add_argument("--postprocess",action="store_true",
        help="Enable threshold tuning (requires --use_validation separate).\n"
             "Optimizes classification threshold on validation set for better performance."
    )
    postprocess_group.add_argument("--postprocess_params",default="{}",
        help="Threshold tuning parameters as Python dict:\n"
             "   tuning_method ('recall'|'f1'|'f2'|'youden'|'cost', default='recall'):\n"
             "      - recall: Maximize sensitivity (default, medical priority)\n"
             "      - f1: Maximize F1-score (balanced precision-recall)\n"
             "      - f2: Maximize F2-score (2x recall weight vs precision)\n"
             "      - youden: Maximize Youden's J statistic (TPR - FPR)\n"
             "      - cost: Minimize cost-sensitive loss function\n"
             "   cost_fn (float, default=10.0): False Negative cost (for method='cost')\n"
             "   cost_fp (float, default=1.0): False Positive cost (for method='cost')\n"
             "   show_tuning_plots (bool, default=True): Display optimization plots\n"
             "Example: \"{'tuning_method': 'f1', 'show_tuning_plots': False}\""
    )
    eval_group = p.add_argument_group('Evaluation & Explainability')
    eval_group.add_argument("--eval",default="['AUC ROC','accuracy','Confusion matrix']",
        help="Evaluation metrics as Python list. Available metrics:\n"
             "   'accuracy': Overall classification accuracy\n"
             "   'precision': Positive Predictive Value (PPV)\n"
             "   'recall': Sensitivity / True Positive Rate (TPR)\n"
             "   'F1': Harmonic mean of precision and recall\n"
             "   'AUC ROC': Area Under Receiver Operating Characteristic curve\n"
             "   'AUC PR': Area Under Precision-Recall curve\n"
             "   'Confusion Matrix': 2x2 confusion matrix with TN/FP/FN/TP\n"
             "Example: \"['accuracy','F1','AUC ROC','Confusion Matrix']\""
    )
    eval_group.add_argument("--xai", action="store_true",
        help="Enable Explainable AI (XAI) analysis using SHAP.\n"
             "Provides feature importance rankings and model interpretation.\n"
             "   Decision Tree: SHAP TreeExplainer with waterfall/beeswarm plots\n"
             "   Logistic Regression: Coefficient-based analysis\n"
             "   SVM: SHAP LinearExplainer or KernelExplainer\n"
             "   DNN: SHAP DeepExplainer with beeswarm plots"
    )
    eval_group.add_argument("--xai_sample", default=None,
        help="Optional sample index for XAI analysis. If provided, generates a SHAP waterfall plot "
             "explaining the contribution of features for this sample to the model's predictions."
    )
    return p.parse_args(argv)





