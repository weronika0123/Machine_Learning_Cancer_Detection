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

def SHAP(explainer_type, model, X_train, X_test, X_val, feature_names):

    explainer = None
    if explainer_type == "linear":
        explainer = shap.LinearExplainer(model, X_train)
        shap_values = explainer(X_val)
    elif explainer_type == "deep":

        # unwrap the Keras model 
        keras_model = model.get_keras_model()

        # Convert background and validation data to NumPy arrays (2D tabular format)
        X_bg = X_train.values if isinstance(X_train, pd.DataFrame) else np.asarray(X_train)
        X_val_np = X_val.values if isinstance(X_val, pd.DataFrame) else np.asarray(X_val)

        # Subsample background (≤200) for computational efficiency
        if X_bg.shape[0] > 200:
            rng = np.random.default_rng(42)
            X_bg = X_bg[rng.choice(X_bg.shape[0], size=200, replace=False)]

        # SHAP values
        explainer = shap.DeepExplainer(keras_model, X_bg)
        shap_values_raw = explainer.shap_values(X_val_np)
        expected_value = explainer.expected_value

        # Normalize output to 2D (n_samples, n_features)
        shap_values_arr = shap_values_raw[0] if isinstance(shap_values_raw, list) else shap_values_raw
        shap_values_arr = np.asarray(shap_values_arr)
        if shap_values_arr.ndim == 3 and shap_values_arr.shape[-1] == 1:
            shap_values_arr = shap_values_arr[..., 0]

        # Validate dimensional consistency
        assert shap_values_arr.ndim == 2
        assert X_val_np.ndim == 2
        assert shap_values_arr.shape[1] == X_val_np.shape[1]

        # base values 
        ev = np.array(expected_value).ravel()
        base_vals = ev if ev.shape[0] == X_val_np.shape[0] else np.full(X_val_np.shape[0], ev.mean(), dtype=float)

        feat_names_vec = list(feature_names)
        expl = shap.Explanation(
            values=shap_values_arr,
            base_values=base_vals,
            data=X_val_np,
            feature_names=feat_names_vec
        )

        # global feature importance
        feature_importance = np.abs(expl.values).mean(axis=0)

        top_k = 15
        top_15_indices = np.argsort(feature_importance)[-top_k:][::-1]
        top_15_indices = np.asarray(top_15_indices, dtype=int).ravel().tolist()
        top_15_features = np.array(expl.feature_names)[top_15_indices]

        # Beeswarm plot 
        shap_values_top_15 = expl[:, top_15_indices]
        print(f"[XAI] Generating beeswarm plot (top {top_k} features) for DNN (sigmoid)..")
        shap.plots.beeswarm(shap_values_top_15, show=False, max_display=top_k)
        fig = plt.gcf()
        fig.set_size_inches(12, 6)
        plt.title(f"Global Feature Importance (Top {top_k}) - DNN (sigmoid)", fontsize=14, pad=10)
        fig.tight_layout()
        plt.show()

        top_5_features = list(top_15_features[:5])
        print(f"[XAI] Top 5 most important features (absolute mean SHAP): {top_5_features}")
        return top_5_features
    
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

    if model_kind != "DNN":
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
        top5_features = SHAP("deep", model, X_train, X_test, X_val, feature_names)
        
        return ("Deep SHAP for DNN", top5_features)

    # Fallback
    else:
        print("[XAI] Model not recognized — using generic SHAP KernelExplainer")
        

        return (" generic fallback = SHAP KernelExplainer", top5_features)