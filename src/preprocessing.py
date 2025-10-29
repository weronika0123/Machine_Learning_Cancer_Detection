import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, recall_score
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import LinearSVC
import warnings

RANDOM_STATE = 42


def correlation_removal(X_train, X_test, threshold, full_mask):
    
    print(f"[PREPROCESS] Starting correlation-based feature removal with threshold={threshold}...")

    #Initial number of features
    n_features = X_train.shape[1]

    #Step 1: Dropping constant features (zero variance) 
    variances = X_train.var(axis=0)
    nonconst_mask = variances > 0.0  # True = keep
    dropped_const = (~nonconst_mask).sum()

    #Step 2: Using training set only for correlation matrix
    X_train_non_const = X_train[:, nonconst_mask]

    #Step 3: Computing Pearson's Correlation matrix 
    corr = np.corrcoef(X_train_non_const, rowvar=False)
    candidate_nr = corr.shape[0]
    start_mask = np.ones(candidate_nr, dtype=bool)  # True = keep

    #Scanning upper triangle only
    for i in range(candidate_nr):
        if not start_mask[i]:
            continue

        column_mask = np.abs(corr[i, (i+1):]) > threshold  # TRUE = drop
        relative_indexes = np.where(column_mask)[0]

        if relative_indexes.size > 0:
            drop_idx = (i+1) + relative_indexes
            start_mask[drop_idx] = False

    #Final mask relative to original feature set
    final_mask = np.zeros(n_features, dtype=bool)  # True = keep
    final_mask[nonconst_mask] = start_mask

    X_train_red = X_train[:, final_mask]
    X_test_red  = X_test[:,  final_mask]

    full_mask &= final_mask

    info = {
        "initial_features": n_features,
        "dropped_constant": dropped_const,
        "dropped_corr": int(nonconst_mask.sum() - start_mask.sum()),
        "kept": int(final_mask.sum()),
        "threshold": threshold,
    }
    
    print(f"[PREPROCESS] Correlation removal: threshold={threshold} | "
          f"kept={info['kept']} out of {info['initial_features']} "
          f"(dropped_const={info['dropped_constant']}, "
          f"dropped_corr={info['dropped_corr']})")
        

    return X_train_red, X_test_red, full_mask, info

def prefilter_select_kbest(X_train, y_train, X_test, full_mask, k=1500):
    
    #Validation - k must not be greater than number of features
    k = min(k, X_train.shape[1])

    #SelectKBest with ANOVA F-value
    selector = SelectKBest(score_func=f_classif, k=k)

    #Calculate F-scores and keep top-k features
    X_train_k = selector.fit_transform(X_train, y_train)

    #Apply the same top-k selection to test set
    X_test_k  = selector.transform(X_test)

    mask_k = selector.get_support() 
    print(f"[FEATURE_SEL] Selected top-{X_train_k.shape[1]} features (out of {X_train.shape[1]}) using SelectKBest (f_classif)")

    #Map kbest mask back to the original feature space
    idx_alive = np.where(full_mask)[0]  #Indices of features that survived so far
    new_full_mask = np.zeros_like(full_mask)
    new_full_mask[idx_alive] = mask_k
    full_mask = new_full_mask

    return X_train_k, X_test_k, full_mask

def estimator(model_name: str):

    if model_name == "Logistic Regression" or model_name == "DNN":
        fs_estimator = LogisticRegression(
            max_iter=1000, 
            class_weight="balanced", 
            random_state=RANDOM_STATE
        )
    elif model_name == "Decision Tree":
        fs_estimator = DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    elif model_name == "SVM":
        fs_estimator = LinearSVC(
            max_iter=100000,
            class_weight="balanced", 
            random_state=RANDOM_STATE
        )

    else:
        print("[FEATURE_SEL][WARN] Model name unknown - fallback: LogisticRegression")
        fs_estimator = LogisticRegression(
            max_iter=1000, 
            class_weight="balanced", 
            random_state=RANDOM_STATE
        )
    return fs_estimator

def rfecv(steps: int, X_train, y_train, X_test, model_name, full_mask, output_dir):

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    #Scorer based on recall for the positive class (label=1)
    recall_scorer = make_scorer(recall_score, pos_label=1)

    #Base estimator for RFECV
    fs_estimator = estimator(model_name)

    rfecv_model = RFECV(
        estimator=fs_estimator,
        step=steps,
        scoring=recall_scorer,
        cv=cv,
        min_features_to_select=50,
        n_jobs=-1
    )
    #Fit RFECV (on training set only)
    rfecv_model.fit(X_train, y_train)

    mask_rfe_full = rfecv_model.support_
    print(f"[FEATURE_SEL] Optimal k (recall-optimized): {rfecv_model.n_features_}")

    X_train_sel = X_train[:, mask_rfe_full]
    X_test_sel  = X_test[:,  mask_rfe_full]

    #CV history: mean recall per iteration and the corresponding number of features
    scores = np.asarray(rfecv_model.cv_results_["mean_test_score"]).ravel()
    n_features_list = np.asarray(rfecv_model.cv_results_["n_features"]).ravel()

    #Plot: RFECV - recall vs number of features (k)
    plt.figure(figsize=(12, 6))
    plt.plot(n_features_list, scores, marker="o", linewidth=2, color='#2E86AB')
    #Mark the optimal k
    plt.axvline(rfecv_model.n_features_, ls="--", color="#D8504D", linewidth=2, label=f"Optimal k = {rfecv_model.n_features_}")
    best_idx = int(np.argmax(scores))
    plt.scatter([n_features_list[best_idx]], [scores[best_idx]], s=80, color='#D8504D', zorder=3)

    plt.xlabel("Number of Features (k) - Training Set (CV)", fontsize=11)
    plt.ylabel("Mean Recall - Training Set (CV)", fontsize=11)
    plt.title("RFECV: Recall vs Number of Features", fontsize=14)
    plt.gca().invert_xaxis()
    plt.grid(True, ls=":", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plot_path = output_dir / "rfecv_plot.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"[FEATURE_SEL] RFECV plot saved to {plot_path}")

    #Map selection back to the original feature space
    idx_alive = np.where(full_mask)[0]
    new_full_mask = np.zeros_like(full_mask)
    #Inject into previous full_mask
    new_full_mask[idx_alive] = mask_rfe_full
    full_mask = new_full_mask

    return X_train_sel, X_test_sel, full_mask, rfecv_model

def feature_selection(steps: int, X_train, y_train, X_test, 
                      model_name: str, fs_methods: list, prefilter_k: int=1500, corr_threshold: float=0.95, output_dir=None):

    print("[FEATURE_SEL] Starting feature selection...")

    n0 = X_train.shape[1]  # Original number of features
    mask_best = np.ones(n0, dtype=bool)  # Full mask relative to original feature set

    if "corr" in fs_methods:
        X_train, X_test, mask_best, corr_info = correlation_removal(X_train, X_test, corr_threshold, mask_best)

    if "kbest" in fs_methods:
        X_train, X_test, mask_best = prefilter_select_kbest(X_train, y_train, X_test, mask_best, k=prefilter_k)

    if "rfecv" in fs_methods:
        X_train, X_test, mask_best, rfecv_model = rfecv(steps, X_train, y_train, X_test, model_name, mask_best, output_dir)
    
    return X_train, X_test, mask_best
