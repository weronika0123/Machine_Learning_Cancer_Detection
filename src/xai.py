import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import _tree, plot_tree
import numpy as np
from pathlib import Path

def _save_fig(output_dir, filename):
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    full = outdir / filename
    plt.savefig(full, bbox_inches="tight", dpi=150)
    plt.show()
    return str(full)

def explain_lr_with_coeffs(model, feature_names, top_k=15, output_dir=None):
    coefs = model.coef_[0]  #Binary classification
    df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    df["abs_coef"] = df["coef"].abs()
    df = df.sort_values("abs_coef", ascending=False).head(top_k)
    top5_features = df["feature"].head(5).tolist()

    #Plot
    colors = df["coef"].apply(lambda x: "#06A77D" if x > 0 else "#D8504D")
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(df["feature"], df["coef"], color=colors)
    ax.set_xlabel("Coefficient (β)", fontsize=11)
    ax.set_title(f"Top {top_k} Most Important Features (Logistic Regression)", fontsize=14)
    ax.invert_yaxis()
    for bar in bars:
        value = bar.get_width()
        ax.text(value, bar.get_y() + bar.get_height()/2, f"{value:.2f}",
                va="center", ha="left", fontsize=10, color="black")
    ax.grid(True, ls=':', alpha=0.6, axis='x')
    fig.tight_layout()

    #Save (if requested) and/or show
    saved = _save_fig(output_dir, "logistic_regression_coeff_xai_plot.png")
    print(f"[XAI] Coefficients plot saved to {saved}")

    return top5_features


def auto_left_margin(labels):
    if not labels:
        return 0.18
    L = max(len(str(s)) for s in labels)
    return min(0.55, 0.18 + 0.012 * L)


def SHAP(explainer_type, model, X_train, X_val, feature_names=None, output_dir=None, xai_sample=None):
    explainer = None
#region Linear SHAP
    if explainer_type == "linear":
        explainer = shap.LinearExplainer(model, X_train)
        shap_values = explainer(X_val)

        # Extract top 15 features directly from SHAP values
        feature_importance = np.abs(shap_values.values).mean(axis=0)
        top_15_indices = np.argsort(feature_importance)[-15:][::-1]
        top_15_features = X_val.columns[top_15_indices]

        # Beeswarm plot
        shap_values_top_15 = shap_values[:, top_15_indices]
        print(f"[XAI] Generating beeswarm plot (top 15 features) for LinearExplainer")
        plt.figure(figsize=(12, 6))  
        shap.plots.beeswarm(shap_values_top_15, show=False, max_display=15)
        fig = plt.gcf()
        plt.title("Global Feature Importance (Top 15) - Linear SHAP", fontsize=14, pad=10)
        fig.tight_layout()
        saved = _save_fig(output_dir, "shap_beeswarm_linear.png")
        if saved is not None:
            print(f"[XAI] SHAP beeswarm plot saved to {saved}")
        else:
            plt.show()

        # Generate waterfall plot for the specified feature
        if xai_sample is not None:
            sample_idx = int(xai_sample)  
            if 0 <= sample_idx < len(shap_values):
                print(f"[XAI] Generating waterfall plot (top 15 features) for sample index: {sample_idx}")

                # Get SHAP values for that single sample
                sample_values = shap_values[sample_idx].values
                sample_base = shap_values[sample_idx].base_values
                sample_data = shap_values[sample_idx].data
                feature_names = shap_values.feature_names

                # Sort features by absolute importance for this sample and keep top 15
                top_k = 15
                top_idx = np.argsort(np.abs(sample_values))[-top_k:][::-1]

                sample_expl = shap.Explanation(
                    values=sample_values[top_idx],
                    base_values=sample_base,
                    data=np.array(sample_data)[top_idx],
                    feature_names=np.array(feature_names)[top_idx]
                )

                plt.figure(figsize=(12, 6))
                shap.plots.waterfall(sample_expl, max_display=top_k, show=False)
                plt.title(f"Waterfall Plot (Top {top_k}) – Sample #{sample_idx}", fontsize=14, pad=10)
                plt.tight_layout()
                saved = _save_fig(output_dir, f"shap_waterfall_top{top_k}_sample_{sample_idx}.png")
                print(f"[XAI] SHAP waterfall plot (top {top_k}) saved to {saved}")

            else:
                print(f"[XAI] Invalid sample index: {sample_idx}. Must be between 0 and {len(shap_values)-1}.")
        else:
            print("[XAI] No sample index provided for waterfall plot.")

        top_5_features = top_15_features[:5].tolist()
        return top_5_features
#endregion 
#region Deep SHAP
    elif explainer_type == "deep":
        #Unwrap the Keras model if a wrapper is used
        keras_model = model.get_keras_model() if hasattr(model, "get_keras_model") else model

        #Convert background and validation data to NumPy arrays (2D tabular format)
        X_bg = X_train.values if isinstance(X_train, pd.DataFrame) else np.asarray(X_train)
        X_val_np = X_val.values if isinstance(X_val, pd.DataFrame) else np.asarray(X_val)

        #Subsample background (≤200) for computational efficiency
        if X_bg.shape[0] > 200:
            rng = np.random.default_rng(42)
            X_bg = X_bg[rng.choice(X_bg.shape[0], size=200, replace=False)]

        #Calculate SHAP values
        explainer = shap.DeepExplainer(keras_model, X_bg)
        shap_values_raw = explainer.shap_values(X_val_np)
        expected_value = explainer.expected_value

        #Normalize output to 2D (n_samples, n_features)
        shap_values_arr = shap_values_raw[0] if isinstance(shap_values_raw, list) else shap_values_raw
        shap_values_arr = np.asarray(shap_values_arr)
        if shap_values_arr.ndim == 3 and shap_values_arr.shape[-1] == 1:
            shap_values_arr = shap_values_arr[..., 0]

        #Validate dimensional consistency
        assert shap_values_arr.ndim == 2
        assert X_val_np.ndim == 2
        assert shap_values_arr.shape[1] == X_val_np.shape[1]

        #Base values
        ev = np.array(expected_value).ravel()
        base_vals = ev if ev.shape[0] == X_val_np.shape[0] else np.full(X_val_np.shape[0], ev.mean(), dtype=float)

        #Feature names fallback if None
        if feature_names is None:
            feat_names_vec = [f"gene_{i}" for i in range(X_val_np.shape[1])]
        else:
            feat_names_vec = list(feature_names)

        expl = shap.Explanation(
            values=shap_values_arr,
            base_values=base_vals,
            data=X_val_np,
            feature_names=feat_names_vec
        )

        #Global feature importance
        feature_importance = np.abs(expl.values).mean(axis=0)
        top_k = 15
        top_15_indices = np.argsort(feature_importance)[-top_k:][::-1]
        top_15_indices = np.asarray(top_15_indices, dtype=int).ravel().tolist()
        top_15_features = np.array(expl.feature_names)[top_15_indices]

        #Beeswarm plot
        shap_values_top_15 = expl[:, top_15_indices]
        print(f"[XAI] Generating beeswarm plot (top {top_k} features) for DNN")
        shap.plots.beeswarm(shap_values_top_15, show=False, max_display=top_k)
        fig = plt.gcf()
        fig.set_size_inches(12, 6)
        plt.title(f"Global Feature Importance (Top {top_k}) - Deep Neural Network", fontsize=14, pad=10)
        fig.tight_layout()
        saved = _save_fig(output_dir, "shap_beeswarm_dnn.png")
        print(f"[XAI] SHAP beeswarm plot saved to {saved}")


        #  TOP 15 genes for picked sample 
        if xai_sample is not None:
            sample_idx = int(xai_sample)
            if 0 <= sample_idx < len(expl):
                print(f"[XAI] Generating DNN waterfall (top 15) for sample #{sample_idx}")

                # SHAP for the picked sample
                sample_values = expl.values[sample_idx]      # (n_features,)
                sample_base   = expl.base_values[sample_idx] # skalar
                sample_data   = expl.data[sample_idx]        # (n_features,)
                feat_names    = np.array(expl.feature_names)

                # TOP-15 based on |SHAP| for the picked sample
                top_k  = 15
                top_ix = np.argsort(np.abs(sample_values))[-top_k:][::-1]

                # Temporary Explanation 
                sample_expl = shap.Explanation(
                    values       = sample_values[top_ix],
                    base_values  = sample_base,
                    data         = np.asarray(sample_data)[top_ix],
                    feature_names= feat_names[top_ix]
                )

                plt.figure(figsize=(12, 6))  
                shap.plots.waterfall(sample_expl, max_display=top_k, show=False)
                plt.title(f"Waterfall (Top {top_k}) – DNN – Sample #{sample_idx}", fontsize=14, pad=10)
                plt.tight_layout()
                saved = _save_fig(output_dir, f"shap_waterfall_dnn_top{top_k}_sample_{sample_idx}.png")
                print(f"[XAI] SHAP waterfall plot (top {top_k}) saved to {saved}")
            else:
                print(f"[XAI] Invalid sample index: {sample_idx}. Must be between 0 and {len(expl)-1}.")
        else:
            print("[XAI] No sample index provided for DNN waterfall.")


        top_5_features = list(top_15_features[:5])
        print(f"[XAI] Top 5 most important features (absolute mean SHAP): {top_5_features}")
        return top_5_features
#endregion
#region Tree SHAP
    elif explainer_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_val)

        #Select positive class for binary classification and calculate absolute mean importances
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

        if xai_sample is not None:
            sample_idx = int(xai_sample)
            if 0 <= sample_idx < len(shap_values_pos):
                print(f"[XAI] Generating Decision Tree waterfall (top 15) for sample #{sample_idx}")

                # SHAP dla tej próbki
                sample_values = shap_values_pos.values[sample_idx]     # (n_features,)
                sample_base   = shap_values_pos.base_values[sample_idx] # skalar
                sample_data   = shap_values_pos.data[sample_idx]        # (n_features,)
                feat_names    = np.array(shap_values_pos.feature_names)

                # TOP-15 wg |SHAP| dla TEJ próbki
                top_k  = 15
                top_ix = np.argsort(np.abs(sample_values))[-top_k:][::-1]

                # tymczasowy obiekt Explanation tylko dla top 15
                sample_expl = shap.Explanation(
                    values        = sample_values[top_ix],
                    base_values   = sample_base,
                    data          = np.asarray(sample_data)[top_ix],
                    feature_names = feat_names[top_ix]
                )

                plt.figure(figsize=(12, 6))
                shap.plots.waterfall(sample_expl, max_display=top_k, show=False)
                plt.title(f"Waterfall (Top {top_k}) – Decision Tree – Sample #{sample_idx}", fontsize=14, pad=10)
                plt.tight_layout()
                saved = _save_fig(output_dir, f"shap_waterfall_tree_top{top_k}_sample_{sample_idx}.png")
                if saved is None:
                    plt.show(); plt.close()
            else:
                print(f"[XAI] Invalid sample index: {sample_idx}. Must be between 0 and {len(shap_values_pos)-1}.")
        else:
            print("[XAI] No sample index provided for Tree waterfall.")

        #Beeswarm plot for top 15 features
        shap_values_top_15 = shap_values_pos[:, top_15_indices]
        print(f"[XAI] Generating beeswarm plot (top {top_k} features)")
        shap.plots.beeswarm(shap_values_top_15, show=False, max_display=top_k)
        fig = plt.gcf()
        fig.set_size_inches(12, 6)
        plt.title(f"Global Feature Importance (Top {top_k}) - Decision Tree", fontsize=14, pad=10)
        fig.tight_layout()
        saved = _save_fig(output_dir, "shap_beeswarm_tree.png")
        print(f"[XAI] SHAP waterfall plot (top {top_k}) saved to {saved}")

        top_5_features = top_15_features[:5].tolist()
        print(f"[XAI] Top 5 most important features: {top_5_features}")
        return top_5_features
#endregion
#region Kernel SHAP
    else:
        #Kernel SHAP fallback (e.g., non-linear SVM without Tree/Deep explainers)
        #Prefer proba if available (better for binary classification)
        f = (lambda X: model.predict_proba(X)[:, 1]) if hasattr(model, "predict_proba") else model.predict
        background = X_train
        if X_train.shape[0] > 1000:
            background = shap.sample(X_train, 100, random_state=42)
        explainer = shap.KernelExplainer(f, background)
        shap_values = explainer(X_val)

        #Extract top 15 features directly from SHAP values
        feature_importance = np.abs(shap_values.values).mean(axis=0)
        top_15_indices = np.argsort(feature_importance)[-15:][::-1]
        top_15_features = X_val.columns[top_15_indices]

        #Beeswarm plot for top 15 features
        shap_values_top_15 = shap_values[:, top_15_indices]
        print(f"[XAI] Generating beeswarm plot (top 15 features) for Kernel SHAP")
        shap.plots.beeswarm(shap_values_top_15, show=False, max_display=15)
        fig = plt.gcf()
        fig.set_size_inches(12, 6)
        plt.title("Global Feature Importance (Top 15) - Kernel SHAP", fontsize=14, pad=10)
        fig.tight_layout()
        saved = _save_fig(output_dir, "shap_beeswarm_kernel.png")
        if saved is not None:
            print(f"[XAI] SHAP beeswarm plot saved to {saved}")
        else:
            plt.show()

        # Generate waterfall plot for the specified feature
        if xai_sample is not None:
            sample_idx = int(xai_sample)  
            if 0 <= sample_idx < len(shap_values):
                print(f"[XAI] Generating waterfall plot (top 15 features) for sample index: {sample_idx}")

                # Get SHAP values for that single sample
                sample_values = shap_values[sample_idx].values
                sample_base = shap_values[sample_idx].base_values
                sample_data = shap_values[sample_idx].data
                feature_names = shap_values.feature_names

                # Sort features by absolute importance for this sample and keep top 15
                top_k = 15
                top_idx = np.argsort(np.abs(sample_values))[-top_k:][::-1]

                sample_expl = shap.Explanation(
                    values=sample_values[top_idx],
                    base_values=sample_base,
                    data=np.array(sample_data)[top_idx],
                    feature_names=np.array(feature_names)[top_idx]
                )

                plt.figure(figsize=(12, 6))
                shap.plots.waterfall(sample_expl, max_display=top_k, show=False)
                plt.title(f"Waterfall Plot (Top {top_k}) – Sample #{sample_idx}", fontsize=14, pad=10)
                plt.tight_layout()
                saved = _save_fig(output_dir, f"shap_waterfall_top{top_k}_sample_{sample_idx}.png")
                print(f"[XAI] SHAP waterfall plot (top {top_k}) saved to {saved}")

            else:
                print(f"[XAI] Invalid sample index: {sample_idx}. Must be between 0 and {len(shap_values)-1}.")
        else:
            print("[XAI] No sample index provided for waterfall plot.")
        #Extract top 5 features
        top_5_features = top_15_features[:5].tolist()
        return top_5_features
#endregion

def run_xai(model_kind, model, feature_names, X_train, X_test, X_val=None, output_dir=None, xai_sample=None):
    print(f"[XAI] Running explanations for model: {model_kind}")
    top5_features = []

    if X_val is None or (hasattr(X_val, "shape") and X_val.shape[0] == 0):
        X_val = X_test

    if model_kind != "DNN":
        X_train = pd.DataFrame(X_train, columns=feature_names)
        X_val = pd.DataFrame(X_val, columns=feature_names)


    #Logistic Regression
    if model_kind == "Logistic Regression":
        top5_features = explain_lr_with_coeffs(model, feature_names, top_k=15, output_dir=output_dir)
        if xai_sample is not None:
            SHAP("linear", model, X_train, X_val, feature_names, output_dir=output_dir, xai_sample=xai_sample)
        return ("Coefficient-based analysis", top5_features)

    #Decision Tree
    elif model_kind == "Decision Tree":
        top5_features = SHAP("tree", model, X_train, X_val, output_dir=output_dir, xai_sample=xai_sample)
        return ("SHAP TreeExplainer for Decision Tree", top5_features)

    #SVM (linear)
    elif (model_kind == "SVM linear" or model_kind == "SVM linear calibrated"):
        if model_kind == "SVM linear calibrated":
            base_model = model.calibrated_classifiers_[0].estimator
            top5_features = SHAP("linear", base_model, X_train, X_val, feature_names, output_dir=output_dir, xai_sample=xai_sample)
        else:
            top5_features = SHAP("linear", model, X_train, X_val,feature_names,  output_dir=output_dir, xai_sample=xai_sample)
        return ("SHAP (linear) for SVM", top5_features)

    #SVM (non-linear, e.g., RBF)
    elif model_kind == "SVM":
        top5_features = SHAP("kernel", model, X_train, X_val, feature_names, output_dir=output_dir,  xai_sample=xai_sample)
        return ("Kernel SHAP / LIME for SVM-RBF", top5_features)

    #Deep Neural Network
    elif model_kind == "DNN":
        top5_features = SHAP("deep", model, X_train, X_val, feature_names, output_dir=output_dir,  xai_sample=xai_sample)
        return ("Deep SHAP for DNN", top5_features)

    #Fallback
    else:
        print("[XAI] Model not recognized - using generic SHAP KernelExplainer")