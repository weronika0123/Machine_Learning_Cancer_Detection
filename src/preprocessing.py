import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, recall_score
from sklearn.svm import LinearSVC

RANDOM_STATE = 42

def correlation_removal(X_train, X_test, threshold=0.90):
    
    print(f"[CORR] Starting correlation-based feature removal with threshold={threshold}...")

    n_features = X_train.shape[1]

    #  Drop constant features (zero variance)
    variances = X_train.var(axis=0)
    nonconst_mask = variances > 0.0
    dropped_const = int((~nonconst_mask).sum())

    X_train_nc = X_train[:, nonconst_mask]
    X_test_nc  = X_test[:, nonconst_mask]

    #  Correlation matrix (train only)
    corr = np.corrcoef(X_train_nc, rowvar=False)
    m = corr.shape[0]
    keep = np.ones(m, dtype=bool)

    # Scan upper triangle
    for i in range(m):
        if not keep[i]:
            continue
        high_corr = np.where(np.abs(corr[i, (i+1):]) > threshold)[0]
        if high_corr.size > 0:
            drop_idx = (i+1) + high_corr
            keep[drop_idx] = False

    # Final mask relative to original feature set
    final_mask = np.zeros(n_features, dtype=bool)
    final_mask[nonconst_mask] = keep

    X_train_red = X_train[:, final_mask]
    X_test_red  = X_test[:,  final_mask]

    info = {
        "initial_features": n_features,
        "dropped_constant": dropped_const,
        "dropped_corr": int(nonconst_mask.sum() - keep.sum()),
        "kept": int(final_mask.sum()),
        "threshold": threshold,
    }
    
    print(f"[CORR] threshold={threshold} | "
          f"kept={info['kept']} out of {info['initial_features']} "
          f"(dropped_const={info['dropped_constant']}, "
          f"dropped_corr={info['dropped_corr']})")
    
    # Plot correlation heatmap (train only, after dropping constants)
    #plt.figure(figsize=(8, 6))
    #sns.heatmap(corr, cmap="coolwarm", center=0,
    #                xticklabels=False, yticklabels=False)
    #plt.title(f"Correlation matrix (train, after dropping constants)\nThreshold = {threshold}")
    #plt.show()

    return X_train_red, X_test_red, final_mask, info

def feature_selection(steps:int, X_train, y_train, X_test, model_name):

    print("[RFECV] Starting RFECV... It might take a while, but worry not, it's working well :)")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    recall_scorer = make_scorer(recall_score, pos_label=1)
    fs_estimator = None

    if(model_name == "Logistic Regression"):
        fs_estimator = LogisticRegression(
            max_iter=2000, solver="lbfgs", penalty="l2", C=1.0, class_weight="balanced", random_state=RANDOM_STATE
        )
    elif(model_name == "Decision Tree"):
        fs_estimator = DecisionTreeClassifier(
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",      
            random_state=RANDOM_STATE
        )
    elif model_name == "SVM":
        # Używamy linearnego SVM do RFECV, bo ma coef_.
        fs_estimator = LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=20000,   # stabilniejsza zbieżność w high-dim
            tol=1e-3,
            dual=True,
        )
    else:
        # Bezpieczny fallback – LogisticRegression
        print("[RFECV] [WARN] Nieznany model_name dla FS – używam LogisticRegression jako fallback.")
        fs_estimator = LogisticRegression(
            max_iter=2000, solver="lbfgs", penalty="l2", C=1.0,
            class_weight="balanced", random_state=RANDOM_STATE
        )
        
    rfecv = RFECV(
        estimator=fs_estimator,
        step=steps,                       # usuń co 'step' cech na iterację
        scoring=recall_scorer,
        cv=cv,
        min_features_to_select=20,
        n_jobs=-1
    )
    rfecv.fit(X_train, y_train)
    mask = rfecv.support_
    k_selected = rfecv.n_features_
    print(f"[RFECV] Wybrane k (opt pod RECALL): {k_selected}")

    X_train_sel = X_train[:, mask]
    X_test_sel  = X_test[:,  mask]

     # recall vs liczba cech
    scores = np.asarray(rfecv.cv_results_["mean_test_score"]).ravel()
    n_features_list = np.asarray(rfecv.cv_results_["n_features"]).ravel()
    plt.figure(figsize=(7,4))
    plt.plot(n_features_list, scores, marker="o", linewidth=1)
    plt.axvline(rfecv.n_features_, ls="--", color="red", label=f"Optymalne k ={rfecv.n_features_}")
    plt.xlabel("Liczba cech (k) — TRAIN (CV)")
    plt.ylabel("Średni recall — TRAIN (CV)")
    14
    plt.title("RFECV (balanced): recall (CV) vs liczba cech")
    plt.gca().invert_xaxis()
    plt.grid(True, ls=":")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return X_train_sel, X_test_sel, mask, rfecv