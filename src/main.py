import argparse
import json
import sys
from pathlib import Path
import ast
import pandas as pd
from sklearn.model_selection import train_test_split  # TODO: użyj w kroku 4
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge, Lasso  # Ridge/Lasso na przyszłość

RANDOM_STATE = 42  # dla powtarzalności wyników

def pipeline(
    dane: str,
    preprocesing: bool,
    model_name: str,
    model_params: dict,
    EVAL: list,
    XAI: bool,
):

    # 1) Wczytanie danych
    path = Path(dane)
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {path}")

    # Uwaga na separator/encoding: jeśli masz ; albo cp1250, użyj: pd.read_csv(path, sep=';', encoding='cp1250')
    df = pd.read_csv(path)

    # (opcjonalne) — drop ostatnich 16 kolumn
    # TODO: df = df.iloc[:, :-16]

    # 1a) TODO: wybór kolumn X i y (jeśli masz target)
    # X = ...
    # y = ...

    # 2) Opcjonalny preprocessing
    if preprocesing:
        # TODO: imputacja/skaling/one-hot/ColumnTransformer/Pipeline
        pass

    # 3) Wybór i inicjalizacja modelu
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
        # TODO: jeśli będziesz skalować: MinMax/StandardScaler tutaj lub w Pipeline

        # Wyciągamy parametry z dict
        max_iter = model_params.get("max_iter", 100)
        solver = model_params.get("solver", "lbfgs")
        penalty = model_params.get("penalty", "l2")
        C = model_params.get("C", 1.0)
        class_weight = model_params.get("class_weight", "balanced")  # domyślnie balanced, ale można nadpisać

        model = LogisticRegression(
            max_iter=max_iter, solver=solver, penalty=penalty, C=C, class_weight=class_weight
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

    # 4) Podział na train/test
    # TODO: jeśli to klasyfikacja i masz y: stratify=y
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # 5) Trening
    # TODO: model.fit(X_train, y_train)

    # 6) Ewaluacja
    # TODO: nalicz metryki z listy EVAL (np. accuracy, AUC ROC, Confusion matrix) i zapisz do results
    results = {
        # "accuracy": ...,
        # "AUC ROC": ...,
        # "Confusion matrix": ...,
    }

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

