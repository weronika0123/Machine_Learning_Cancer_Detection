

def run_xai(model_name, model, X_train, X_test):
    print(f"[XAI] Running explanations for model: {model_name}")

    # Logistic Regression
    if model_name.lower() in ["lr", "logistic_regression"]:
        print("[XAI] Using SHAP LinearExplainer for Logistic Regression")
        return "SHAP (linear) for LR"

    # Decision Tree
    elif model_name.lower() in ["dt", "decision_tree"]:
        print("[XAI] Using LIME for Decision Tree (simple and local explanation)")
        return "LIME for DT"

    # SVM (linear)
    elif model_name.lower() in ["svm_linear", "linear_svm"]:
        print("[XAI] Using SHAP LinearExplainer for linear SVM")
        return "SHAP (linear) for SVM"

    # SVM (non-linear, e.g. RBF)
    elif model_name.lower() in ["svm_rbf", "svm_kernel"]:
        print("[XAI] Using Kernel SHAP or LIME (model-agnostic) for kernel SVM")
        return "Kernel SHAP / LIME for SVM-RBF"

    # Deep Neural Network
    elif model_name.lower() in ["dnn", "mlp", "neural_network"]:
        print("[XAI] Using SHAP (DeepExplainer) or Integrated Gradients for DNN")
        return "Deep SHAP / Integrated Gradients for DNN"

    # Fallback
    else:
        print("[XAI] Model not recognized — using generic Permutation Importance")
        return "Permutation Importance (generic fallback)"
