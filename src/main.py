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

RANDOM_STATE = 42  # dla powtarzalności wyników

def pipeline(
    dane: str,
    preprocesing: list,
    model_name: str,
    model_params: dict,
    EVAL: list,
    XAI: bool,
):


#region Data Prep

    # Flag innit
    feature_selection = False
    correlation_removal = False

    # 1) Identify model kind (string)
    model_name_norm = model_name.strip().lower()
    if model_name_norm in ("logisticregression", "lr"):
        model_kind = "Logistic Regression"
    elif model_name_norm in ("decisiontree", "dt"):
        model_kind = "Decision Tree"
    else:
        raise ValueError("Nieznany model. Użyj: DecisionTree/DT lub LogisticRegression/LR")

    # 2) Identify preprocessing steps (list of strings)
    steps = [str(s).strip().lower() for s in (preprocesing or [])]
    if(any(s in ("feature selection", "feature_selection", "fs" , "f s") for s in steps)):
        feature_selection = True
    if (any(s in ("correlation removal", "correlation_removal", "corr", "corr remv", "cr") for s in steps)):
        correlation_removal = True

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

    # 2) Opcjonalny preprocessing
    if correlation_removal:
        # TODO: correlation removal
        pass

    
    # Base train/test split
    X_train = X[df["isTraining"] == 1]
    y_train = y[df["isTraining"] == 1]
    X_test = X[df["isTest"] == 1]
    y_test = y[df["isTest"] == 1]

    # FTO = for test only 
    print(f"Size X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Size X_test: {X_test.shape}, y_test: {y_test.shape}")
#endregion

#region Model selection + training
    
    XAI_model = None  # "LIME" albo "SHAP"
    XAI_model_specific = None  # np. "TreeExplainer" / "KernelExplainer"

    if model_kind == "Decision Tree":
        # Parametry dla drzewa (np. {"max_depth": 4, "random_state": 42})
        model = DecisionTreeClassifier(**model_params)
        if XAI:
            XAI_model = "SHAP"
            XAI_model_specific = "TreeExplainer"

    elif model_kind == "Logistic Regression":

        # MinMax for scaling to [0,1]
        min_max_scaler = MinMaxScaler() 
        X_train = min_max_scaler.fit_transform(X_train)
        X_test  = min_max_scaler.transform(X_test)

        max_iter = model_params.get("max_iter", 100)
        solver = model_params.get("solver", "lbfgs")
        penalty = model_params.get("penalty", "l2")
        C = model_params.get("C", 1.0)

        model = LogisticRegression(
            max_iter=max_iter, solver=solver, penalty=penalty, C=C, class_weight="balanced", random_state=RANDOM_STATE
        )

        # feature selection
        if(feature_selection):

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            recall_scorer = make_scorer(recall_score, pos_label=1)

            rfecv = RFECV(
                estimator=model,
                step=50,                       
                scoring=recall_scorer,         
                cv=cv,
                min_features_to_select=20,
                n_jobs=-1
            )

            rfecv.fit(X_train, y_train)

            mask = rfecv.support_
            k_selected = rfecv.n_features_
            print(f"[RFECV balanced] Wybrane k (opt pod RECALL): {k_selected}")

            # przycięcie danych do wybranych cech
            X_train = X_train[:, mask]
            X_test  = X_test[:,  mask]

            #model
            model = LogisticRegression(
                max_iter=max_iter, solver=solver, penalty=penalty, C=C, class_weight="balanced", random_state=RANDOM_STATE
            )
        
        # TODO: threshold tuning

        if XAI:
            XAI_model = "LIME"
            XAI_model_specific = "KernelExplainer"

    else:
        raise ValueError("Nieznany model. Użyj: DecisionTree/DT lub LogisticRegression/LR")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

#endregion

#region Ewaluacja

    results = {}

    # Accuracy
    if any(m.lower() in ["accuracy", "ac"] for m in EVAL):

        results["accuracy"] = accuracy_score(y_test, y_pred)
    
    # Precision
    if any(m.lower() in ["precision", "prec", "p"] for m in EVAL):
    # Out of all predicted "cancer = 1" cases, how many are truly cancer.
    # High precision = few false positives (healthy people predicted as sick).

        results["precision"] = precision_score(y_test, y_pred, zero_division=0)

    # Recall (Sensitivity)
    if any(m.lower() in ["recall", "sensitivity", "r"] for m in EVAL):
    # Out of all true "cancer = 1" cases, how many did the model catch.
    # High recall = few false negatives (sick people missed by the model).

        results["recall"] = recall_score(y_test, y_pred, zero_division=0)

    # F1-score
    if any(m.lower() in ["f1", "f1-score", "f1score"] for m in EVAL):
    # Harmonic mean of precision and recall, balances both metrics.
    # In our context: good when we want both few false positives and few false negatives.
        results["f1"] = f1_score(y_test, y_pred, zero_division=0)

    # Confusion matrix
    if any(m.lower() in ["confusion matrix", "confusion_matrix", "cm"] for m in EVAL):

        cm = confusion_matrix(y_test, y_pred)

        # Bc in results the formating is bad
        TN, FP, FN, TP = cm.ravel()
        print(" Confusion matrix: TN",TN,"FP",FP,"FN",FN,"TP",TP)

        # plot
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.tight_layout()
        plt.show()

        results["Confusion matrix"] = cm.tolist()

    # AUC ROC
    if any(m.lower() in ["auc roc", "auc_roc", "roc auc", "roc_auc"] for m in EVAL):

        y_score = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        plt.subplot(1,2,1)
        plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
        plt.plot([0,1],[0,1],'--',lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

        results["AUC ROC"] = float(roc_auc_score(y_test, y_score))

    # Precision-Recall AUC
    if any(m.lower() in ["auc pr", "auc_pr", "pr auc", "pr_auc", "average precision", "ap"] for m in EVAL):
            
            y_score = model.predict_proba(X_test)[:, 1]
    
            precision, recall, _ = precision_recall_curve(y_test, y_score)
            pr_auc = auc(recall, precision)
            ap = average_precision_score(y_test, y_score)
    
            plt.subplot(1,2,2)
            plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f} (AP={ap:.4f})")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve")
            plt.legend(loc="lower left")
            plt.tight_layout()
            plt.show()
    
            results["AUC PR"] = float(ap)

#endregion

    # 7) XAI
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
        choices=["DecisionTree", "DT", "LogisticRegression", "LR"],
        help="Wybór modelu."
    )
    p.add_argument("--params", default="{}", help="Parametry modelu jako słownik Pythona, np. {'max_depth': 4}")
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
        EVAL=eval_list,
        XAI=args.xai,
    )

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


#j