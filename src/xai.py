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


def auto_left_margin(labels):
    if not labels:
        return 0.18
    L = max(len(str(s)) for s in labels)
    return min(0.55, 0.18 + 0.012 * L)

def SHAP(explainer_type, model, X_train, X_test, X_val):

    explainer = None
    if explainer_type == "linear":
        explainer = shap.LinearExplainer(model, X_train)
    elif explainer_type == "deep":
        if hasattr(model, "get_keras_model"):
            model = model.get_keras_model()  # Use the underlying Keras model
        explainer = shap.DeepExplainer(model, X_train)
        shap_values = explainer.shap_values(X_test)
    elif explainer_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_val)

        # wybór klasy dodatniej przy binarce oraz absolute mean importances
        if len(shap_values.values.shape) == 3 and shap_values.values.shape[2] == 2:
            print("[XAI] Binary classification detected - using positive class (cancer=1)")
            shap_values_pos = shap_values[:, :, 1]
            feature_importance = np.abs(shap_values_pos.values).mean(axis=0)
        else:
            shap_values_pos = shap_values
            feature_importance = np.abs(shap_values.values).mean(axis=0)

        top_k = 15
        top_15_indices = np.argsort(feature_importance)[-top_k:][::-1]
        top_15_features = X_val.columns[top_15_indices]

        # Beeswarm plot for top 15 features
        shap_values_top_15 = shap_values_pos[:, top_15_indices]
        print(f"[XAI] Generating beeswarm plot (top {top_k} features)")
        shap.plots.beeswarm(shap_values_top_15, show=False, max_display=top_k)
        fig = plt.gcf()
        fig.set_size_inches(12, 6)
        plt.title(f"Global Feature Importance (Top {top_k}) - Decision Tree", fontsize=14, pad=10)
        fig.tight_layout()
        plt.show()

        top_5_features = top_15_features[:5].tolist()
        print(f"[XAI] Top 5 most important features: {top_5_features}")
        return top_5_features
    else:
        explainer = shap.KernelExplainer(model.predict, X_train)

    shap_values = explainer(X_val)

    # Extract top 15 features directly from SHAP values
    feature_importance = np.abs(shap_values.values).mean(axis=0)
    top_15_indices = np.argsort(feature_importance)[-15:][::-1]
    top_15_features = X_val.columns[top_15_indices]

    # Beeswarm plot for top 15 features
    shap_values_top_15 = shap_values[:, top_15_indices]
    ax = shap.plots.beeswarm(shap_values_top_15, show=False, max_display=15)  # Ensure 15 features are displayed
    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    fig.tight_layout()
    plt.show()

    # Extract top 5 features
    top_5_features = top_15_features[:5].tolist()


    return top_5_features

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
        top5_features = SHAP("tree", model, X_train, X_test, X_val)
        return ("SHAP TreeExplainer for Decision Tree", top5_features)

    # SVM (linear)
    elif (model_kind == "SVM linear" or model_kind == "SVM linear calibrated"):

        if model_kind == "SVM linear calibrated":
            base_model = model.calibrated_classifiers_[0].estimator
            top5_features=SHAP("linear", base_model, X_train, X_test, X_val)
        else:
            top5_features=SHAP("linear", model, X_train, X_test, X_val)

        return ("SHAP (linear) for SVM", top5_features)

    # SVM (non-linear, e.g. RBF)
    elif model_kind == "SVM":
        top5_features=SHAP("kernel", model, X_train, X_test, X_val)

        return ("Kernel SHAP / LIME for SVM-RBF",top5_features)

    # Deep Neural Network
    elif model_kind == "DNN":
        top5_features = SHAP("deep", model, X_train, X_test, X_val)
        
        return ("Deep SHAP for DNN", top5_features)

    # Fallback
    else:
        print("[XAI] Model not recognized — using generic SHAP KernelExplainer")
        

        return (" generic fallback = SHAP KernelExplainer", top5_features)