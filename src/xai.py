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
    top5_features = df["feature"].head(5).tolist()

    # plot
    colors = df["coef"].apply(lambda x: "#ff9999" if x > 0 else "#99ccff")
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df["feature"], df["coef"], color=colors)
    ax.set_xlabel("Coefficient (β)")
    ax.set_title(f"Top {top_k} most important features (Logistic Regression)")
    ax.invert_yaxis()
    for bar in bars:
        value = bar.get_width()
        ax.text(value, bar.get_y() + bar.get_height()/2, f"{value:.2f}", va="center", ha="left", fontsize=9, color="black")
    fig.tight_layout()
    plt.show()
    return top5_features

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
    top5_features = []

    if X_val is None:
        X_val = X_test

    X_train = pd.DataFrame(X_train, columns=feature_names)    
    X_val = pd.DataFrame(X_val, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)


    # Logistic Regression
    if model_kind== "Logistic Regression":

        top5_features = explain_lr_with_coeffs(model, feature_names, top_k=15)
        return ("Coefficient-based analysis", top5_features)

    # Decision Tree
    elif model_kind == "Decision Tree":

        explain_dt_global_importance(model, feature_names, top_k=15)
    
        explain_dt_local_rules(model,X_val, feature_names, show_tree_depth=3)
        return ("DT", top5_features)

    # SVM (linear)
    elif (model_kind == "SVM linear" or model_kind == "SVM linear calibrated"):

        if model_kind == "SVM linear calibrated":
            base_model = model.calibrated_classifiers_[0].estimator
            SHAP("linear", base_model, X_train, X_test, X_val)
        else:
            SHAP("linear", model, X_train, X_test, X_val)

        return ("SHAP (linear) for SVM", top5_features)

    # SVM (non-linear, e.g. RBF)
    elif model_kind == "SVM":
        SHAP("kernel", model, X_train, X_test, X_val)

        return ("Kernel SHAP / LIME for SVM-RBF",top5_features)

    # Deep Neural Network
    elif model_kind == "DNN":
        SHAP("deep", model, X_train, X_test, X_val)

        return ("Deep SHAP for DNN", top5_features)

    # Fallback
    else:
        print("[XAI] Model not recognized — using generic SHAP KernelExplainer")
        

        return (" generic fallback = SHAP KernelExplainer", top5_features)



