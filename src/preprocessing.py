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

def prefilter_select_kbest(X_train, y_train, X_test, k=1500):
    
    #Zwraca X_train/X_test zredukowane do top-k cech wg ANOVA F (f_classif),
    #oraz maskę bool (względem wejścia), które cechy zostały zachowane.
    
    k = min(k, X_train.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_k = selector.fit_transform(X_train, y_train)
    X_test_k  = selector.transform(X_test)
    mask_k = selector.get_support()  # bool mask względem WEJŚCIA (po corr)
    print(f"[KBest] Wybrano top-{X_train_k.shape[1]} cech (z {X_train.shape[1]}) wg f_classif.")
    return X_train_k, X_test_k, mask_k

def feature_selection(steps:int, X_train, y_train, X_test,
                      model_name:str, fs_method:str="rfecv", prefilter_k:int=1500):

    print("[FS] Start feature_selection...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    recall_scorer = make_scorer(recall_score, pos_label=1)

    # 1) TYLKO KBEST -> pełna maska (po corr), zgodna z X_val
    if fs_method.lower() == "kbest":
        X_train_k, X_test_k, mask_kbest_full = prefilter_select_kbest(X_train, y_train, X_test, k=prefilter_k)
        print(f"[FS] Tryb: tylko SelectKBest (k={prefilter_k}) — pomijam RFECV.")
        return X_train_k, X_test_k, mask_kbest_full, None

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

    # 3) KBEST+RFECV -> RFECV na zredukowanym X, a potem składanie pełnej maski
    if fs_method.lower() == "kbest+rfecv":
        print(f"[FS] Tryb: SelectKBest (k={prefilter_k}) + RFECV")
        n_features_full = X_train.shape[1]               # = 5098 po corr
        X_train_k, X_test_k, mask_kbest_full = prefilter_select_kbest(X_train, y_train, X_test, k=prefilter_k)

        print("[RFECV] Odpalam RFECV na danych po KBest...")
        rfecv = RFECV(
            estimator=fs_estimator,
            step=steps,
            scoring=recall_scorer,
            cv=cv,
            min_features_to_select=50,
            n_jobs=-1
        )
        rfecv.fit(X_train_k, y_train)

        # maska RFECV w PRZESTRZENI KBEST (długość = prefilter_k)
        mask_rfe_on_k = rfecv.support_
        print(f"[RFECV] Wybrane k (opt pod RECALL): {rfecv.n_features_} z {X_train_k.shape[1]} (po KBest)")

        # --- SKŁADANIE MASKI PEŁNEJ (względem 5098 kolumn po corr) ---
        full_mask = np.zeros(n_features_full, dtype=bool)
        idx_kbest = np.flatnonzero(mask_kbest_full)   # indeksy cech, które przeszły KBest
        selected_idx_full = idx_kbest[mask_rfe_on_k]  # wybór tych, które przeszły także RFECV
        full_mask[selected_idx_full] = True

        # projekcja na oryginalne X (po corr)
        X_train_sel = X_train[:, full_mask]
        X_test_sel  = X_test[:,  full_mask]

        # wykres (opcjonalnie) — nadal pokazujemy krzywą RFECV
        scores = np.asarray(rfecv.cv_results_["mean_test_score"]).ravel()
        n_features_list = np.asarray(rfecv.cv_results_["n_features"]).ravel()
        plt.figure(figsize=(7,4))
        plt.plot(n_features_list, scores, marker="o", linewidth=1)
        plt.axvline(rfecv.n_features_, ls="--", color="red", label=f"Optymalne k = {rfecv.n_features_}")
        best_idx = int(np.argmax(scores))
        plt.scatter([n_features_list[best_idx]], [scores[best_idx]], s=60, zorder=3)
        plt.xlabel("Liczba cech (k) — TRAIN (CV) [po KBest]")
        plt.ylabel("Średni recall — TRAIN (CV)")
        plt.title("RFECV (po KBest): recall (CV) vs liczba cech")
        plt.gca().invert_xaxis()
        plt.grid(True, ls=":")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # ZWRACAMY pełną maskę (5098), więc w main.py:
        # X_val = X_val[:, fs_mask] będzie działać poprawnie.
        return X_train_sel, X_test_sel, full_mask, rfecv

    # 4) Czysty RFECV (bez KBest) – maska od razu jest w pełnej przestrzeni
    print("[FS] Tryb: czysty RFECV (bez prefiltracji)")
    rfecv = RFECV(
        estimator=fs_estimator,
        step=steps,
        scoring=recall_scorer,
        cv=cv,
        min_features_to_select=50,
        n_jobs=-1
    )
    rfecv.fit(X_train, y_train)

    mask_rfe_full = rfecv.support_
    print(f"[RFECV] Wybrane k (opt pod RECALL): {rfecv.n_features_}")

    X_train_sel = X_train[:, mask_rfe_full]
    X_test_sel  = X_test[:,  mask_rfe_full]

    scores = np.asarray(rfecv.cv_results_["mean_test_score"]).ravel()
    n_features_list = np.asarray(rfecv.cv_results_["n_features"]).ravel()
    plt.figure(figsize=(7,4))
    plt.plot(n_features_list, scores, marker="o", linewidth=1)
    plt.axvline(rfecv.n_features_, ls="--", color="red", label=f"Optymalne k = {rfecv.n_features_}")
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

    return X_train_sel, X_test_sel, mask_rfe_full, rfecv