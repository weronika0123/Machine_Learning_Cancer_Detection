import argparse
import json
import sys
from pathlib import Path
import ast
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge, Lasso 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    accuracy_score, roc_auc_score, roc_curve, auc, 
    precision_recall_curve, average_precision_score, fbeta_score,
    make_scorer, recall_score
)
import matplotlib.pyplot as plt

RANDOM_STATE = 42  # dla powtarzalności wyników

def pipeline(
    dane: str,
    preprocesing: bool,
    model_name: str,
    model_params: dict,
    EVAL: list,
    XAI: bool,
):

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
    if preprocesing:
        # TODO: imputacja/skaling/one-hot/ColumnTransformer/Pipeline
        pass

    
    # Base train/test split
    X_train = X[df["isTraining"] == 1]
    y_train = y[df["isTraining"] == 1]
    X_test = X[df["isTest"] == 1]
    y_test = y[df["isTest"] == 1]

    # FTO = for test only 
    print(f"Size X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Size X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Model selection
    model_name_norm = model_name.strip().lower()
    model = None
    XAI_model = None  # "LIME" albo "SHAP"
    XAI_model_specific = None  # np. "TreeExplainer" / "KernelExplainer"

    if model_name_norm in ("decisiontree", "dt"):
        # Parametry dla drzewa (np. {"max_depth": 4, "random_state": 42})
        model = DecisionTreeClassifier(**model_params)
        if XAI:
            XAI_model = "SHAP"
            XAI_model_specific = "TreeExplainer"

    elif model_name_norm in ("logisticregression", "lr"):

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

        # Opcjonalne kroki — zostawione jako TODO, bo nie są zdefiniowane w argumencie
        # TODO: feature selection
        # TODO: correlation removal
        # TODO: threshold tuning

        if XAI:
            XAI_model = "LIME"
            XAI_model_specific = "KernelExplainer"

    else:
        raise ValueError("Nieznany model. Użyj: DecisionTree/DT lub LogisticRegression/LR")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 6) Ewaluacja
    results = {}

    # Accuracy
    if any(m.lower() in ["accuracy", "ac"] for m in EVAL):

        results["accuracy"] = accuracy_score(y_test, y_pred)

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
    p.add_argument("--preprocess", action="store_true", help="Włącz preprocessing.")
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
        if not isinstance(params, dict):
            raise ValueError("--params musi być słownikiem Pythona (np. {'max_iter': 1000})")
        if not isinstance(eval_list, list):
            raise ValueError("--eval musi być listą Pythona (np. ['AUC ROC','accuracy'])")
    except Exception as e:
        print(f"Błąd parsowania --params/--eval: {e}", file=sys.stderr)
        sys.exit(2)


    out = pipeline(
        dane=args.data,
        preprocesing=args.preprocess,
        model_name=args.model,
        model_params=params,
        EVAL=eval_list,
        XAI=args.xai,
    )

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

