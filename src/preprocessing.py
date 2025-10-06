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
    
    print(f"[CORR] Starting correlation-based feature removal with threshold={threshold}...")

    # initial number of features
    n_features = X_train.shape[1]

    # [* corr1] Dropping constant features (zero variance) 
    variances = X_train.var(axis=0)
    nonconst_mask = variances > 0.0 # True = keep
    dropped_const = (~nonconst_mask).sum()

    # [*corr2] Only train set involved
    X_train_non_const = X_train[:, nonconst_mask]

    # [*corr3] Pearsons Correlation matrix 
    corr = np.corrcoef(X_train_non_const, rowvar=False)
    candidate_nr = corr.shape[0]
    start_mask = np.ones(candidate_nr, dtype=bool) # True = keep

    # Scaning upper triangle only
    for i in range(candidate_nr):
        if not start_mask[i]:
            continue

        column_mask = np.abs(corr[i, (i+1):]) > threshold # TRUE = drop
        relative_indexes = np.where(column_mask)[0]

        if relative_indexes.size > 0:
            drop_idx = (i+1) + relative_indexes
            start_mask[drop_idx] = False

    # Final mask relative to original feature set
    final_mask = np.zeros(n_features, dtype=bool) # True = keep
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
    
    print(f"[CORR] threshold={threshold} | "
          f"kept={info['kept']} out of {info['initial_features']} "
          f"(dropped_const={info['dropped_constant']}, "
          f"dropped_corr={info['dropped_corr']})")
        

    return X_train_red, X_test_red, full_mask, info

def prefilter_select_kbest(X_train, y_train, X_test, full_mask,k=1500):
    
    # Validation - k must not be grater than number of features
    k = min(k, X_train.shape[1])

    # [*sel1] SelectKBest with ANOVA F-value
    selector = SelectKBest(score_func=f_classif, k=k)

    # calculating F-score scores, keeping top-k
    X_train_k = selector.fit_transform(X_train, y_train)

    # Applies the same top-k 
    X_test_k  = selector.transform(X_test)

    mask_k = selector.get_support() 
    print(f"[KBest] Wybrano top-{X_train_k.shape[1]} cech (z {X_train.shape[1]}) wg f_classif.")

     # wstaw maskę kbest w miejsca ŻYWYCH kolumn wg full_mask
    idx_alive = np.where(full_mask)[0]    # indeksy kolumn, które przetrwały dotąd
    new_full_mask = np.zeros_like(full_mask)
    new_full_mask[idx_alive] = mask_k
    full_mask = new_full_mask

    return X_train_k, X_test_k, full_mask

def estimator(model_name:str):
    # 2) Dobór estymatora do RFECV (gołe modele – bez Pipeline)
    if model_name == "Logistic Regression":
        fs_estimator = LogisticRegression(
            max_iter=3000, solver="lbfgs", penalty="l2",
            C=0.5, class_weight="balanced", random_state=RANDOM_STATE
        )
    elif model_name == "Decision Tree":
        fs_estimator = DecisionTreeClassifier(
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    elif model_name == "SVM":
        fs_estimator = LinearSVC(
            C=0.5, class_weight="balanced", max_iter=100_000, tol=5e-3, dual=True
        )
    else:
        print("[RFECV][WARN] Nieznany model_name — fallback: LogisticRegression.")
        fs_estimator = LogisticRegression(
            max_iter=3000, solver="lbfgs", penalty="l2",
            C=0.5, class_weight="balanced", random_state=RANDOM_STATE
        )
    return fs_estimator

def rfecv(steps:int, X_train, y_train, X_test, model_name, full_mask):

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    recall_scorer = make_scorer(recall_score, pos_label=1)

    fs_estimator = estimator(model_name)
    print("[FS] Tryb: czysty RFECV (bez prefiltracji)")
    rfecv_model = RFECV(
        estimator=fs_estimator,
        step=steps,
        scoring=recall_scorer,
        cv=cv,
        min_features_to_select=50,
        n_jobs=-1
    )
    rfecv_model.fit(X_train, y_train)

    mask_rfe_full = rfecv_model.support_
    print(f"[RFECV] Wybrane k (opt pod RECALL): {rfecv_model.n_features_}")

    X_train_sel = X_train[:, mask_rfe_full]
    X_test_sel  = X_test[:,  mask_rfe_full]

    scores = np.asarray(rfecv_model.cv_results_["mean_test_score"]).ravel()
    n_features_list = np.asarray(rfecv_model.cv_results_["n_features"]).ravel()
    plt.figure(figsize=(7,4))
    plt.plot(n_features_list, scores, marker="o", linewidth=1)
    plt.axvline(rfecv_model.n_features_, ls="--", color="red", label=f"Optymalne k = {rfecv_model.n_features_}")
    best_idx = int(np.argmax(scores))
    plt.scatter([n_features_list[best_idx]], [scores[best_idx]], s=60, zorder=3)
    plt.xlabel("Liczba cech (k) — TRAIN (CV)")
    plt.ylabel("Średni recall — TRAIN (CV)")
    plt.title("RFECV (balanced): recall (CV) vs liczba cech")
    plt.gca().invert_xaxis()
    plt.grid(True, ls=":")
    plt.legend()
    plt.tight_layout()
    plt.show()

    idx_alive = np.where(full_mask)[0]
    new_full_mask = np.zeros_like(full_mask)
    new_full_mask[idx_alive] = mask_rfe_full
    full_mask = new_full_mask

    return X_train_sel, X_test_sel, full_mask, rfecv_model

def feature_selection(steps:int, X_train, y_train, X_test, 
                      model_name:str, fs_methods:list, prefilter_k:int=1500, corr_threshold:float=0.95):

    print("[FS] Start feature_selection...")

    n0 = X_train.shape[1]                     # liczba cech w oryginale (np. 5277)
    mask_best = np.ones(n0, dtype=bool)       # pełna maska wzgledem oryginału

    if "corr" in fs_methods:
        
        X_train, X_test, mask_best, corr_info = correlation_removal(X_train, X_test, corr_threshold, mask_best)
        # corr_mask JEST już względem oryginału (tak budujesz w correlation_removal)
        
        print(f"[DEBUG CORR] full_mask sum={mask_best.sum()}")
        print(f"[DEBUG CORR] full_mask shape={mask_best.shape}")

    if "kbest" in fs_methods:

        X_train, X_test, mask_best = prefilter_select_kbest(X_train, y_train, X_test,mask_best, k=prefilter_k)
        print(f"[DEBUG KBest] full_mask sum={mask_best.sum()}")
        print(f"[DEBUG KBest] full_mask shape={mask_best.shape}")

    if "rfecv" in fs_methods:
        
        X_train, X_test, mask_best, rfecv_model = rfecv(steps, X_train, y_train, X_test, model_name, mask_best)
        print(f"[DEBUG RFECV] full_mask sum={mask_best.sum()}")
        print(f"[DEBUG RFECV] full_mask shape={mask_best.shape}")
    
    return X_train, X_test, mask_best
