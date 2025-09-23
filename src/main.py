import argparse
import json
import sys
from pathlib import Path
import ast
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge, Lasso 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    accuracy_score, roc_auc_score, roc_curve, auc, 
    precision_recall_curve, average_precision_score, fbeta_score,
    make_scorer, recall_score, precision_score, f1_score, balanced_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.base import clone
from preprocessing import correlation_removal, feature_selection
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV


RANDOM_STATE = 42  # dla powtarzalności wyników

#for threshold tuning
def plot_threshold_curve(tuned_model):
    thresholds = tuned_model.cv_results_["thresholds"]
    scores = tuned_model.cv_results_["scores"]

    plt.figure(figsize=(7,5))
    plt.plot(thresholds, scores, label="Score vs threshold")
    plt.plot(
        tuned_model.best_threshold_,
        tuned_model.best_score_,
        "o",
        markersize=10,
        color="tab:orange",
        label=f"Best thr={tuned_model.best_threshold_:.3f}"
    )
    plt.xlabel("Threshold")
    plt.ylabel("Score (CV)")
    plt.title("Threshold tuning curve")
    plt.legend()
    plt.grid(ls=":")
    plt.tight_layout()
    plt.show()



def pipeline(
    dane: str,
    preprocesing: list,
    model_name: str,
    model_params: dict,
    postprocess: bool,
    EVAL: list,
    XAI: bool,
):

#region Data Prep

    # Flag innit
    feature_selection_flag = False
    correlation_removal_flag = False

    # 1) Identify model kind (string)
    model_name_norm = model_name.strip().lower()
    if model_name_norm in ("logisticregression", "lr"):
        model_kind = "Logistic Regression"
        step = 50  # dla FS = dla regresji ma sens większy krok
    elif model_name_norm in ("decisiontree", "dt"):
        model_kind = "Decision Tree"
        step = 30  # dla FS = drzew ma sens mały krok
    elif model_name_norm in ("svm", "svc"):
        model_kind = "SVM"
        step = 30
    else:
        raise ValueError("Nieznany model. Użyj: DecisionTree/DT lub LogisticRegression/LR lub SVM/SVC")


    # 2) Identify preprocessing steps (list of strings)
    steps = [str(s).strip().lower() for s in (preprocesing or [])]
    if(any(s in ("feature selection", "feature_selection", "fs" , "f s") for s in steps)):
        feature_selection_flag = True
    if (any(s in ("correlation removal", "correlation_removal", "corr", "corr remv", "cr") for s in steps)):
        correlation_removal_flag = True

    # 3) Postprocessing (threshold tuning)
    postprocess_flag = False
    if(postprocess):
        postprocess_flag = True

    # Load data
    path = Path(dane)
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {path}")
    df = pd.read_csv(path, low_memory=False)

    # target: cancer
    X_df = df.iloc[:,1:-16]
    y_df = df.cancer

    X = X_df.to_numpy()
    y = y_df.to_numpy()
    
    # Base train/test split
    X_train = X[df["isTraining"] == 1]
    y_train = y[df["isTraining"] == 1]
    X_val = X[df["isValidation"] == 1]
    y_val = y[df["isValidation"] == 1]
    X_test = X[df["isTest"] == 1]
    y_test = y[df["isTest"] == 1]

    # FTO = for test only 
    print(f"Data sliced: BASE size X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Data sliced: BASE size X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"Data sliced: BASE size X_test: {X_test.shape}, y_test: {y_test.shape}")

    # MinMax for scaling to [0,1]
    if model_kind in ("Logistic Regression", "SVM"):
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)  
        X_val   = scaler.transform(X_val)        
        X_test  = scaler.transform(X_test)      

    corr_mask = None
    if correlation_removal_flag:
        X_train, X_test, corr_mask, corr_info = correlation_removal(
        X_train, X_test, threshold=0.90
    )
    if corr_mask is not None:
        X_val = X_val[:, corr_mask]

    # Feature selection
    fs_mask = None
    if feature_selection_flag:
        X_train, X_test, fs_mask, rfecv = feature_selection(
            steps=step,
            X_train=X_train, y_train=y_train, X_test=X_test,
            model_name=model_kind,
        )
    if fs_mask is not None:
        X_val = X_val[:, fs_mask]


#endregion

#region Model selection + training
    
    XAI_model = None  # "LIME" albo "SHAP"
    XAI_model_specific = None  # np. "TreeExplainer" / "KernelExplainer"

    if model_kind == "Decision Tree":    
        dt_defaults = {
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "class_weight": "balanced",      
            "random_state": RANDOM_STATE
        }
        dt_defaults.update(model_params or {})
        model = DecisionTreeClassifier(**dt_defaults)

        if XAI:
            XAI_model = "SHAP"
            XAI_model_specific = "TreeExplainer"

    elif model_kind == "Logistic Regression":

        max_iter = model_params.get("max_iter", 100)
        solver = model_params.get("solver", "lbfgs")
        penalty = model_params.get("penalty", "l2")
        C = model_params.get("C", 1.0)

        model = LogisticRegression(
            max_iter=max_iter, solver=solver, penalty=penalty, C=C, class_weight="balanced", random_state=RANDOM_STATE
        )
        
        # TODO: threshold tuning

        if XAI:
            XAI_model = "LIME"
            XAI_model_specific = "KernelExplainer"

    elif model_kind == "SVM":
            # SVM z sklearn – różne podejścia w zależności od kernela
            # LinearSVC jest szybszy dla dużych, rzadkich, wysokowymiarowych danych
            # ale nie ma predict_proba. Można to obejść kalibracją (CalibratedClassifierCV).
            # Dla nieliniowych kernelów (RBF, poly, sigmoid) używamy SVC z probability=True.
            # To jest wolniejsze przy wielu cechach i dużym N.
    

        svm_defaults = {
        "kernel": "linear",          # 'linear' | 'rbf' | 'poly' | 'sigmoid'
        "C": 1.0,
        "gamma": "scale",            # dla rbf/poly/sigmoid
        "degree": 3,                 # dla poly
        "class_weight": "balanced",  #zawsze
        "use_calibrated": True,      # tylko dla linear: LinearSVC + CalibratedClassifierCV
        "calibration_method": "sigmoid",  # 'sigmoid' | 'isotonic'
        "cv_calibration": 5,
        "probability": True          # dla SVC (nieliniowe kernele) – żeby mieć predict_proba
    }
        svm_defaults.update(model_params or {})

        kernel = svm_defaults["kernel"]

        if kernel == "linear" and svm_defaults["use_calibrated"]:
            # Szybka wersja do wysokowymiarowych danych: LinearSVC + kalibracja
            base = LinearSVC(
                C=svm_defaults["C"],
                class_weight=svm_defaults["class_weight"],
                # dual=True (domyślnie) – w wielu cechach i mniejszej liczbie próbek jest OK
                # random_state nie jest wymagany
            )
            model = CalibratedClassifierCV(
                estimator=base,
                method=svm_defaults["calibration_method"],
                cv=svm_defaults["cv_calibration"],
            )
        else:
            # Pełny SVC z proba=True (wolniejszy przy tysiącach cech i dużym N)
            model = SVC(
                kernel=kernel,
                C=svm_defaults["C"],
                gamma=svm_defaults["gamma"],
                degree=svm_defaults["degree"],
                class_weight=svm_defaults["class_weight"],
                probability=True,          # konieczne dla ROC/PR i tuningu progu
            )

        if XAI:
            # Dla SVM zwykle LIME/Kernel SHAP (czarna skrzynka)
            XAI_model = "LIME"
            XAI_model_specific = "KernelExplainer"

    else:
        raise ValueError("Nieznany model. Użyj: DecisionTree/DT lub LogisticRegression/LR lub SVM/SVC")

    print("Used X_train shape:", X_train.shape)
    print("Used X_test shape:", X_test.shape)
    
    model.fit(X_train, y_train)

#endregion

#region Post-processing = Threshold tuning
    if postprocess_flag:

        print("[POSTPROCESS] Threshold tuning na zbiorze walidacyjnym...")

        if X_val.shape[0] == 0:
            print("[POSTPROCESS][WARN] Zbiór walidacyjny jest pusty — pomijam tuning progu.")
            y_pred = model.predict(X_test)
        else:
            # scores on VAL
            if not hasattr(model, "predict_proba"):
                raise RuntimeError("Threshold tuning wymaga predict_proba (LR/DT/SVC(probability)/Calibrated).")

            y_score_val = model.predict_proba(X_val)[:, 1]
            thresholds = np.linspace(0, 1, 500)

            best_thr, best_score = 0.5, -1.0
            for thr in thresholds:
                y_pred_val = (y_score_val >= thr).astype(int)
                sc = f1_score(y_val, y_pred_val, zero_division=0)  # optimize for F1 (change if needed)
                if sc > best_score:
                    best_score, best_thr = sc, thr

            print(f"Wybrany threshold={best_thr:.3f} (F1@val={best_score:.3f})")

            # final decision on TEST with tuned threshold
            y_score_test = model.predict_proba(X_test)[:, 1]
            y_pred = (y_score_test >= best_thr).astype(int)
    else:
        y_pred = model.predict(X_test)

#endregion

#region Ewaluacja

    results = {}

    # flags (aliases)
    eval_lower = [m.lower() for m in (EVAL or [])]
    want_acc = any(m in ["accuracy", "ac"] for m in eval_lower)
    want_prec = any(m in ["precision", "prec", "p"] for m in eval_lower)
    want_rec = any(m in ["recall", "sensitivity", "r"] for m in eval_lower)
    want_f1 = any(m in ["f1", "f1-score", "f1score"] for m in eval_lower)
    want_cm = any(m in ["confusion matrix", "confusion_matrix", "cm"] for m in eval_lower)
    want_roc = any(m in ["auc roc", "auc_roc", "roc auc", "roc_auc"] for m in eval_lower)
    want_pr  = any(m in ["auc pr", "auc_pr", "pr auc", "pr_auc", "average precision", "ap"] for m in eval_lower)

    # point metrics (TEST)
    if want_acc:
        results["accuracy"] = accuracy_score(y_test, y_pred)
    if want_prec:
        results["precision"] = precision_score(y_test, y_pred, zero_division=0)
    if want_rec:
        results["recall"] = recall_score(y_test, y_pred, zero_division=0)
    if want_f1:
        results["f1"] = f1_score(y_test, y_pred, zero_division=0)

    # confusion matrix (TEST)
    if want_cm:
        cm = confusion_matrix(y_test, y_pred)
        TN, FP, FN, TP = cm.ravel()
        print(" Confusion matrix: TN", TN, "FP", FP, "FN", FN, "TP", TP)
        ConfusionMatrixDisplay(confusion_matrix=cm).plot()
        plt.title("Confusion Matrix — Test")
        plt.tight_layout()
        plt.show()
        results["Confusion matrix"] = cm.tolist()

    # ROC + PR on a single figure (1x2)
    if (want_roc or want_pr) and hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]

        # prepare figure
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        # ROC
        if want_roc:
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            ax[0].plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
            ax[0].plot([0, 1], [0, 1], "--", lw=1)
            ax[0].set_xlabel("False Positive Rate")
            ax[0].set_ylabel("True Positive Rate")
            ax[0].set_title("ROC Curve — Test")
            ax[0].legend(loc="lower right")
            results["AUC ROC"] = float(roc_auc)
        else:
            ax[0].axis("off")  # keep layout clean if only PR requested

        # PR
        if want_pr:
            precision, recall, _ = precision_recall_curve(y_test, y_score)
            pr_auc = auc(recall, precision)
            ap = average_precision_score(y_test, y_score)
            ax[1].plot(recall, precision, label=f"AUC = {pr_auc:.4f} (AP={ap:.4f})")
            ax[1].set_xlabel("Recall")
            ax[1].set_ylabel("Precision")
            ax[1].set_title("Precision-Recall — Test")
            ax[1].legend(loc="lower left")
            results["AUC PR"] = float(ap)  # store AP as in previous style
        else:
            ax[1].axis("off")

        plt.tight_layout()
        plt.show()
#endregion

#region XAI
    
    if XAI:
        if XAI_model == "LIME":
            # TODO: LIME (train/test)
            pass
        elif XAI_model == "SHAP":
            # TODO: SHAP (train/test)
            pass
        else:
            print("[XAI] Nieznana metoda XAI (użyj: LIME lub SHAP).", file=sys.stderr)

    # 8) Zwrócenie wyniku (na razie tylko szkic)
    return {
        "model": model.__class__.__name__,
        "metrics_requested": EVAL,
        "metrics": results,
        # "xai_method": XAI_model,
        # "xai_specific": XAI_model_specific,
    }
#endregion


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Bazowy szkielet pipeline ML (TODO w środku).")
    p.add_argument("--data", required=True, help="Ścieżka do pliku CSV.")
    p.add_argument(
        "--preprocess",
        default="[]",
        help="Lista kroków preprocessingu jako lista Pythona, np. \"['rfecv','corr']\""
    )
    p.add_argument(
        "--model",
        required=True,
        choices=["DecisionTree", "DT", "LogisticRegression", "LR", "SVM", "SVC"],
        help="Wybór modelu."
    )
    p.add_argument("--params", default="{}", help="Parametry modelu jako słownik Pythona, np. {'max_depth': 4}")
    p.add_argument("--postprocess", action="store_true", help="Włącz postprocessing i.e threshold tuning.")
    p.add_argument("--eval", default="['AUC ROC','accuracy','Confusion matrix']", help="Lista metryk jako lista Pythona.")
    p.add_argument("--xai", action="store_true", help="Włącz XAI.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Parsowanie
    try:
        params = ast.literal_eval(args.params)
        eval_list = ast.literal_eval(args.eval)
        preprocess_list = ast.literal_eval(args.preprocess)
        if not isinstance(preprocess_list, list):
            raise ValueError("--preprocess musi być listą Pythona (np. ['feature selection','corr'])")
        if not isinstance(params, dict):
            raise ValueError("--params musi być słownikiem Pythona (np. {'max_iter': 1000})")
        if not isinstance(eval_list, list):
            raise ValueError("--eval musi być listą Pythona (np. ['AUC ROC','accuracy'])")
    except Exception as e:
        print(f"Błąd parsowania --params/--eval: {e}", file=sys.stderr)
        sys.exit(2)


    out = pipeline(
        dane=args.data,
        preprocesing=preprocess_list,
        model_name=args.model,
        model_params=params,
        postprocess=args.postprocess,
        EVAL=eval_list,
        XAI=args.xai,
    )

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

