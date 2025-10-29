import json
import sys
from pathlib import Path
import ast
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, roc_curve, auc, 
    precision_recall_curve, average_precision_score,
    recall_score, precision_score, f1_score
)
import matplotlib.pyplot as plt
from preprocessing import correlation_removal, feature_selection
from postprocessing import threshold_tuning
from xai import run_xai
from cli import parse_args
from models import train_model
import os
#Must be set BEFORE importing tensorflow:
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  #Suppress TF C++ logs: 0=ALL,1=INFO,2=WARNING,3=ERROR

import warnings
import logging

#Filter unwanted warnings
warnings.filterwarnings("ignore", message=r"The structure of `inputs`.*", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module=r"shap")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"shap")

#Global logging configuration (set once at entry point)
logging.basicConfig(level=logging.WARNING)  #Show WARNING+ by default
log = logging.getLogger("xai")              #Used in xai.py

#Now import TF
import tensorflow as tf
tf.get_logger().setLevel("ERROR") #Python TF logger to ERROR level


RANDOM_STATE = 42


def validate_split_columns(df):
    print("[VALIDATION] Checking data split columns...")
    
    #Check if required columns exist
    required_columns = ['isTraining', 'isValidation', 'isTest']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"CSV missing required split columns: {missing}. "
            f"Required columns are: {required_columns}"
        )
    #Check that values are only 0 or 1
    for col in required_columns:
        unique_vals = sorted(df[col].unique())
        if not set(unique_vals).issubset({0, 1}):
            raise ValueError(
                f"Column '{col}' must contain only 0 or 1. "
                f"Found values: {unique_vals}"
            )
    #Check that each row belongs to exactly one split
    split_sum = df['isTraining'] + df['isValidation'] + df['isTest']
    
    #Find rows where sum != 1 (invalid rows)
    invalid_rows = split_sum != 1
    if invalid_rows.any():
        n_invalid = invalid_rows.sum()        
        error_msg = (
            f"Found {n_invalid} rows with invalid split assignment\n"
        )
        raise ValueError(error_msg)
    
    #Check that each split has at least some data
    train_count = (df['isTraining'] == 1).sum()
    val_count = (df['isValidation'] == 1).sum()
    test_count = (df['isTest'] == 1).sum()
    total = len(df)
    
    if train_count == 0:
        raise ValueError("Training set is empty")
    if test_count == 0:
        raise ValueError("Test set is empty")
    
    #Validation can be empty-warning only
    if val_count == 0:
        print("[VALIDATION][WARN] Validation set is empty (no rows with isValidation=1)")
        print("[VALIDATION][WARN] This is acceptable if using --use_validation merge_train_test")
    
    #Statistics
    print(f"[VALIDATION] Data distribution:")
    print(f"- Training:{train_count:5d} rows ({100*train_count/total:5.1f}%)")
    print(f"- Validation:{val_count:5d} rows ({100*val_count/total:5.1f}%)")
    print(f"- Test:{test_count:5d} rows ({100*test_count/total:5.1f}%)")
    print(f"- Total:{total:5d} rows")


#Flexible data splitting with validation handling options
def prepare_data_split(X, y, df, use_validation="separate"):

    # First, extract base splits
    X_train_base = X[df["isTraining"] == 1]
    y_train_base = y[df["isTraining"] == 1]
    X_val_base = X[df["isValidation"] == 1]
    y_val_base = y[df["isValidation"] == 1]
    X_test_base = X[df["isTest"] == 1]
    y_test_base = y[df["isTest"] == 1]
    
    #Option A: separate train/validation/test
    if use_validation == "separate":
        return X_train_base, X_val_base, X_test_base, y_train_base, y_val_base, y_test_base
    
    #Option B: Split validation 80/20 into train/test
    elif use_validation == "merge_train_test":
        if X_val_base.shape[0] > 0:
            X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(
                X_val_base, y_val_base,
                test_size=0.2,
                stratify=y_val_base,
                random_state=RANDOM_STATE
            )
            X_train = np.vstack([X_train_base, X_val_train])
            y_train = np.concatenate([y_train_base, y_val_train])
            X_test = np.vstack([X_test_base, X_val_test])
            y_test = np.concatenate([y_test_base, y_val_test])
            X_val = np.array([]).reshape(0, X.shape[1])  
            y_val = np.array([])
            print(f"[DATA] Merged validation: +{X_val_train.shape[0]} to train, +{X_val_test.shape[0]} to test")
        else:
            X_train, X_val, X_test = X_train_base, X_val_base, X_test_base
            y_train, y_val, y_test = y_train_base, y_val_base, y_test_base
        
        return X_train, X_val, X_test, y_train, y_val, y_test


def pipeline(
                dane: str,
                use_validation: str,
                preprocesing: list,
                model_name: str,
                model_params: dict,
                preprocess_params: dict,
                postprocess_params: dict,
                postprocess: bool,
                EVAL: list,
                XAI: bool,
                ):

#region Data Prep
    #Flag innit
    feature_selection_flag = False
    correlation_removal_flag = False

    #Identify model kind (string)
    model_name_norm = model_name.strip().lower()
    if model_name_norm in ("logisticregression", "lr"):
        model_kind = "Logistic Regression"
        step = 50  #For FS: larger step for regression makes sense
    elif model_name_norm in ("decisiontree", "dt"):
        model_kind = "Decision Tree"
        step = 30  #For FS: smaller step for trees makes sense
    elif model_name_norm in ("svm", "svc"):
        model_kind = "SVM"
        step = 30
    elif model_name_norm in ("dnn", "deepneuralnetwork"):
        model_kind = "DNN"
        step = 50
    else:
        raise ValueError("Unknown model. Use: DecisionTree/DT or LogisticRegression/LR or SVM/SVC or DNN")


    #Identify preprocessing steps (list of strings)
    if preprocesing:
        fs_method = []
        steps = [str(s).strip().lower() for s in (preprocesing or [])]

        if(any(s in ("recursive feature elimination with cross-validation","recursive feature elimination" , "rfecv", "rfe") for s in steps)):
            fs_method.append("rfecv")
            
        if(any(s in ("kbest", "selectkbest", "select_k_best", "k best") for s in steps)):
            fs_method.append("kbest")

        if (any(s in ("correlation removal", "correlation_removal", "corr", "corr remv", "cr") for s in steps)):
            fs_method.append("corr")
            correlation_removal_flag = True

        feature_selection_flag = True

    #Postprocessing (threshold tuning)
    postprocess_flag = False
    if(postprocess):
        postprocess_flag = True

    #Validate postprocess and merge_train_test combination
    if postprocess_flag and use_validation == "merge_train_test":
        raise ValueError(
            "Cannot use --postprocess (threshold tuning) with --use_validation merge_train_test. "
            "Threshold tuning requires a validation set. Please use --use_validation separate "
            "or remove the --postprocess flag."
        )
    
    #XAI idenitification
    xai_flag = False
    if XAI:
        xai_flag = True

    #Load data
    path = Path(dane)

    #File Validations
    if not path.exists():
        raise FileNotFoundError(f"Cannot find file: {path}\nPlease provide the correct path.")
    if(not str(path).endswith(".csv")):
        raise ValueError("Only .csv files are supported. Please provide a valid CSV file.")
    if path.stat().st_size == 0:
        raise ValueError("The provided CSV file is empty. Please provide a non-empty, valid CSV file.")
    
    df = pd.read_csv(path, low_memory=False)


    if str(path)==r"data_sources\liquid_biopsy_data.csv":
        print("[DATA] Using liquid_biopsy_data.csv dataset")
        X_df = df.iloc[:,1:-16]

    else:
        print("[DATA] Using your dataset, assuming last column is target")
        X_df = df.iloc[:,:-1]

    #Missing values check - close the pipeline if found
    if X_df.isnull().values.any():
        print("[DATA][ERROR] Missing values detected. Please provide data without any missing values.")
        sys.exit(1)

    #Validate split columns (isTraining, isValidation, isTest)
    validate_split_columns(df)
    
    #target: cancer
    y_df = df.cancer

    #Gene names for future referances
    feature_names = X_df.columns.to_list() 

    X = X_df.to_numpy()
    y = y_df.to_numpy()
    
    #Call prepare_data_split to get train/val/test splits
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_split(
        X, y, df, use_validation=use_validation
    )

    #Log the split sizes
    print(f"[DATA] Data split - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"[DATA] Data split - X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"[DATA] Data split - X_test: {X_test.shape}, y_test: {y_test.shape}")

    #Feature selection
    fs_mask = None
    if feature_selection_flag:
        corr_threshold = float(preprocess_params.get("corr_threshold", 0.90))  #default threshold
        prefilter_k = preprocess_params.get("prefilter_k", 1500)  #default k for SelectKBest prefiltering

        X_train, X_test, fs_mask = feature_selection(
            steps=step,
            X_train=X_train, y_train=y_train, X_test=X_test,
            model_name=model_kind,fs_methods=fs_method, prefilter_k=prefilter_k, corr_threshold=corr_threshold)
        feature_names = [name for name, keep in zip(feature_names, fs_mask) if keep]

        if fs_mask is not None and X_val.shape[0] > 0:
            X_val = X_val[:, fs_mask]
        

    #MinMaxScaler - Applied AFTER feature engineering for correct scaling statistics
    if model_kind in ("Logistic Regression", "SVM", "DNN"):
        print(f"[SCALING] Applying MinMaxScaler to {X_train.shape[1]} features (models:{model_kind})")
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)  #Learn min/max from train, then scale
        
        if X_val.shape[0] > 0:
            X_val = scaler.transform(X_val)  #Apply train statistics (no data leakage)
            print(f"[SCALING] Applied to validation set: {X_val.shape}")
        X_test = scaler.transform(X_test)  #Apply train statistics (no data leakage)
        print(f"[SCALING] Feature ranges after scaling: [{X_train.min():.3f}, {X_train.max():.3f}]")
    else:
        print(f"[SCALING] Skipped for {model_kind} (tree-based models don't require scaling)")


#endregion

#region Model selection + training
    
    model, model_kind = train_model(
        model_kind=model_kind,
        model_params=model_params,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_val=X_val,
        y_val=y_val
    )

#endregion

#region XAI
    XAI_method = None
    XAI_top_features = None

    if xai_flag:
        XAI_method, XAI_top_features = run_xai(model_kind, model, feature_names, X_train, X_test, X_val)
#endregion

#region Post-processing = Threshold tuning
    tuning_info = None
    if postprocess_flag:
        #Extract postprocessing parameters from postprocess_params
        tuning_method = postprocess_params.get("tuning_method", "recall")  #default: recall (medical priority)
        cost_fn = postprocess_params.get("cost_fn", 10.0)  #False Negative cost
        cost_fp = postprocess_params.get("cost_fp", 1.0)   #False Positive cost
        show_plots = postprocess_params.get("show_tuning_plots", True)
        
        y_pred, tuning_info = threshold_tuning(
            model=model,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            method=tuning_method,
            cost_fn=cost_fn,
            cost_fp=cost_fp,
            thresholds=500,
            show_plots=show_plots
        )
    else:
        y_pred = model.predict(X_test)

#endregion

#region Ewaluacja

    results = {}

    #flags (aliases)
    eval_lower = [m.lower() for m in (EVAL or [])]
    want_acc = any(m in ["accuracy", "ac"] for m in eval_lower)
    want_prec = any(m in ["precision", "prec", "p"] for m in eval_lower)
    want_rec = any(m in ["recall", "sensitivity", "r"] for m in eval_lower)
    want_f1 = any(m in ["f1", "f1-score", "f1score"] for m in eval_lower)
    want_cm = any(m in ["confusion matrix", "confusion_matrix", "cm"] for m in eval_lower)
    want_roc = any(m in ["auc roc", "auc_roc", "roc auc", "roc_auc"] for m in eval_lower)
    want_pr  = any(m in ["auc pr", "auc_pr", "pr auc", "pr_auc", "average precision", "ap"] for m in eval_lower)

    #point metrics (TEST)
    if want_acc:
        results["accuracy"] = accuracy_score(y_test, y_pred)
    if want_prec:
        results["precision"] = precision_score(y_test, y_pred, zero_division=0)
    if want_rec:
        results["recall"] = recall_score(y_test, y_pred, zero_division=0)
    if want_f1:
        results["f1"] = f1_score(y_test, y_pred, zero_division=0)

    #confusion matrix (TEST)
    if want_cm:
        cm = confusion_matrix(y_test, y_pred)
        TN, FP, FN, TP = cm.ravel()
        print(f"[EVAL] Confusion Matrix: TN={TN}, FP={FP}, FN={FN}, TP={TP}")
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
        ax.set_title("Confusion Matrix — Test Set", fontsize=14)
        plt.tight_layout()
        plt.show()
        results["Confusion matrix"] = cm.tolist()

    #ROC + PR on a single figure (1x2)
    if (want_roc or want_pr) and hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]

        #Prepare figure
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        #ROC
        if want_roc:
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            ax[0].plot(fpr, tpr, color='#2E86AB', linewidth=2, label=f"AUC = {roc_auc:.4f}")
            ax[0].plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5)
            ax[0].set_xlabel("False Positive Rate", fontsize=11)
            ax[0].set_ylabel("True Positive Rate", fontsize=11)
            ax[0].set_title("ROC Curve - Test Set", fontsize=14)
            ax[0].legend(loc="lower right", fontsize=10)
            ax[0].grid(True, ls=':', alpha=0.6)
            results["AUC ROC"] = float(roc_auc)
        else:
            ax[0].axis("off")  #Keep layout clean if only PR requested

        #PR
        if want_pr:
            precision, recall, _ = precision_recall_curve(y_test, y_score)
            pr_auc = auc(recall, precision)
            ap = average_precision_score(y_test, y_score)
            ax[1].plot(recall, precision, color='#2E86AB', linewidth=2, label=f"AUC = {pr_auc:.4f} (AP={ap:.4f})")
            ax[1].set_xlabel("Recall", fontsize=11)
            ax[1].set_ylabel("Precision", fontsize=11)
            ax[1].set_title("Precision-Recall Curve — Test Set", fontsize=14)
            ax[1].legend(loc="lower left", fontsize=10)
            ax[1].grid(True, ls=':', alpha=0.6)
            results["AUC PR"] = float(ap)  #Store AP as in previous style
        plt.tight_layout()
        plt.show()


    #Add threshold tuning info to results if performed
    if tuning_info is not None and tuning_info['tuning_performed']:
        results["threshold_tuning"] = {
            "best_threshold": float(tuning_info['best_threshold']),
            "best_score": float(tuning_info['best_score']),
            "tuning_performed": tuning_info['tuning_performed']
        }
        
    return {
        "model": model.__class__.__name__,
        "metrics_requested": EVAL,
        "metrics": results,
        "xai_method": XAI_method,
        "XAI_top_features": XAI_top_features,
    }
#endregion


def main(argv=None):
    args = parse_args(argv)
    try:
        preprocess_params = ast.literal_eval(args.preprocess_params)
        if not isinstance(preprocess_params, dict):
            raise ValueError("--preprocess_params must be a Python dict")
        model_params = ast.literal_eval(args.model_params)
        if not isinstance(model_params, dict):
            raise ValueError("--model_params must be a Python dict")
        postprocess_params = ast.literal_eval(args.postprocess_params)
        if not isinstance(postprocess_params, dict):
            raise ValueError("--postprocess_params must be a Python dict")
        preprocess_list = ast.literal_eval(args.preprocess)
        if not isinstance(preprocess_list, list):
            raise ValueError("--preprocess must be a Python list (np. ['feature selection','corr','selectKBest'])")
        eval_list = ast.literal_eval(args.eval)
        if not isinstance(eval_list, list):
            raise ValueError("--eval must be a Python list (np. ['AUC ROC','accuracy'])")
            
    except Exception as e:
        print(f"Parameter parsing error: {e}", file=sys.stderr)
        sys.exit(2)

    out = pipeline(
        dane=args.data,
        use_validation=args.use_validation,
        preprocesing=preprocess_list,
        model_name=args.model,
        model_params=model_params,
        preprocess_params=preprocess_params,
        postprocess_params=postprocess_params,
        postprocess=args.postprocess,
        EVAL=eval_list,
        XAI=args.xai)

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
