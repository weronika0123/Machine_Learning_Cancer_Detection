import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import _tree, plot_tree
import numpy as np

def explain_lr_with_coeffs(model, feature_names, top_k=15):
    coefs = model.coef_[0]  # binary classification
    df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    df["abs_coef"] = df["coef"].abs()
    df = df.sort_values("abs_coef", ascending=False).head(top_k)

    # plot
    colors = df["coef"].apply(lambda x: "#ff9999" if x > 0 else "#99ccff")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(df["feature"], df["coef"], color=colors)
    ax.set_xlabel("Coefficient (β)")
    ax.set_title(f"Top {top_k} most important features (Logistic Regression)")
    ax.invert_yaxis()
    fig.tight_layout()
    plt.show()
    return df

def explain_dt_global_importance(model, feature_names, top_k=15):

    imp = model.feature_importances_  # Gini importance

    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    df = df[df["importance"] > 0]
    df = df.sort_values("importance", ascending=False).head(min(top_k, len(df)))

    # plot
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(df["feature"], df["importance"], color="#b3cde3")
    ax.set_xlabel("Gini importance")
    ax.set_title(f"Top {len(df)} features (Decision Tree)")
    ax.invert_yaxis()
    fig.tight_layout()
    plt.show()
    return df

def explain_dt_local_rules(model, X_val, feature_names, show_tree_depth=3):

    x_row_df = X_val.iloc[[0]]
    class_names = ['0', '1']

    x_vec = x_row_df.values.reshape(1, -1)
    tree = model.tree_
    feature = tree.feature
    threshold = tree.threshold

    # ścieżka i liść
    node_indicator = model.decision_path(x_vec)
    leaf_id = model.apply(x_vec)[0]

    # budujemy reguły IF-THEN
    rules = []
    for node_id in node_indicator.indices:
        if feature[node_id] != _tree.TREE_UNDEFINED:
            fname = feature_names[feature[node_id]]
            thr = threshold[node_id]
            val = x_vec[0, feature[node_id]]
            sign = "<=" if val <= thr else ">"
            rules.append(f"{fname} {sign} {thr:.4f}")

    # predykcja / proby
    proba_txt = ""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_vec)[0]
        if class_names is None and hasattr(model, "classes_"):
            class_names = [str(c) for c in model.classes_]
        if class_names is not None:
            top_idx = int(np.argmax(proba))
            proba_txt = f"  →  class={class_names[top_idx]} (p={proba[top_idx]:.3f})"
        else:
            proba_txt = f"  →  proba={proba}"

    # wypisz regułę
    rule_text = "IF " + " AND ".join(rules) + " THEN decision at leaf" + proba_txt
    print("[XAI][DT][LOCAL]\n" + rule_text)

    # mały podgląd drzewa (pierwsze poziomy) – czytelny, nie ucina się
    fig, ax = plt.subplots(figsize=(14, 8))
    plot_tree(
        model,
        max_depth=show_tree_depth,
        feature_names=feature_names,
        class_names=(class_names if class_names is not None else None),
        filled=True, rounded=True, impurity=True, proportion=True, fontsize=9, ax=ax
    )
    fig.tight_layout()
    plt.show()

    return {"rules": rules, "leaf_id": int(leaf_id)}

def auto_left_margin(labels):
    if not labels:
        return 0.18
    L = max(len(str(s)) for s in labels)
    return min(0.55, 0.18 + 0.012 * L)

def SHAP(mode, model, X_train, X_test, X_val):

    explainer = None
    if mode == "linear":
        explainer = shap.LinearExplainer(model, X_train)
    elif mode == "deep":
        explainer = shap.DeepExplainer(model, X_train)
    else:
        explainer = shap.KernelExplainer(model.predict, X_train)

    shap_values = explainer(X_val)

    # 1 sample explanation waterfall plot
    ax = shap.plots.waterfall(shap_values[0], show=False) 
    fig = plt.gcf()
    labels = list(X_val.columns)
    left = auto_left_margin(labels)
    fig.set_size_inches(13, 7)                       
    fig.subplots_adjust(left=left, right=0.98, top=0.95, bottom=0.08)
    fig.tight_layout()                               
    plt.show()


    # for global explenation beeswarm plot
    ax = shap.plots.beeswarm(shap_values, show=False)
    fig = plt.gcf()
    labels = list(X_val.columns)
    left = auto_left_margin(labels)
    fig.set_size_inches(12, 6)
    fig.subplots_adjust(left=left, right=0.98, top=0.95, bottom=0.08)
    fig.tight_layout()
    plt.show()
    
    return "Shap explainer output"

def run_xai(model_kind, model, feature_names, X_train, X_test, X_val=None):
    print(f"[XAI] Running explanations for model: {model_kind}")

    if X_val is None:
        X_val = X_test

    X_train = pd.DataFrame(X_train, columns=feature_names)    
    X_val = pd.DataFrame(X_val, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)


    # Logistic Regression
    if model_kind== "Logistic Regression":

        explain_lr_with_coeffs(model, feature_names, top_k=15)
        return "Success"

    # Decision Tree
    elif model_kind == "Decision Tree":

        explain_dt_global_importance(model, feature_names, top_k=15)
    
        explain_dt_local_rules(model,X_val, feature_names, show_tree_depth=3)
        return "DT"

    # SVM (linear)
    elif (model_kind == "SVM linear" or model_kind == "SVM linear calibrated"):

        if model_kind == "SVM linear calibrated":
            base_model = model.calibrated_classifiers_[0].estimator
            SHAP("linear", base_model, X_train, X_test, X_val)
        else:
            SHAP("linear", model, X_train, X_test, X_val)

        return "SHAP (linear) for SVM"

    # SVM (non-linear, e.g. RBF)
    elif model_kind == "SVM":
        SHAP("kernel", model, X_train, X_test, X_val)

        return "Kernel SHAP / LIME for SVM-RBF"

    # Deep Neural Network
    elif model_kind == "DNN":
        SHAP("deep", model, X_train, X_test, X_val)

        return "Deep SHAP for DNN"

    # Fallback
    else:
        print("[XAI] Model not recognized — using generic SHAP KernelExplainer")
        

        return "(generic fallback)"



