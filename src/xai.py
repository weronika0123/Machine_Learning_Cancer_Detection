import shap
import pandas as pd
import matplotlib.pyplot as plt


def auto_left_margin(labels):
    if not labels:
        return 0.18
    L = max(len(str(s)) for s in labels)
    return min(0.55, 0.18 + 0.012 * L)

def SHAP(mode, model, X_train, X_test, X_val):
    # Placeholder for SHAP explanation logic

    explainer = None
    if mode == "linear":
        explainer = shap.LinearExplainer(model, X_train)
    elif mode == "tree":
        explainer = shap.TreeExplainer(model)
    elif mode == "deep":
        explainer = shap.DeepExplainer(model, X_train)
    else:
        explainer = shap.KernelExplainer(model.predict, X_train)

    shap_values = explainer(X_val)

    # 1 feature explanation waterfall plot
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
    labels = list(X_val.columns) if isinstance(X_val, pd.DataFrame) else []
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
        print("[XAI] Using SHAP LinearExplainer for Logistic Regression")
        SHAP("linear", model, X_train, X_test, X_val)

        return "Success"

    # Decision Tree
    elif model_kind == "Decision Tree":
        print("[XAI] Using LIME for Decision Tree (simple and local explanation)")


        return "LIME for DT"

    # SVM (linear)
    elif model_kind == "SVM linear":
        print("[XAI] Using SHAP LinearExplainer for linear SVM")


        return "SHAP (linear) for SVM"

    # SVM (non-linear, e.g. RBF)
    elif model_kind == "SVM":
        print("[XAI] Using Kernel SHAP or LIME (model-agnostic) for kernel SVM")


        return "Kernel SHAP / LIME for SVM-RBF"

    # Deep Neural Network
    elif model_kind == "DNN":
        print("[XAI] Using SHAP (DeepExplainer) or Integrated Gradients for DNN")


        return "Deep SHAP / Integrated Gradients for DNN"

    # Fallback
    else:
        print("[XAI] Model not recognized — using generic Permutation Importance")


        return "Permutation Importance (generic fallback)"



