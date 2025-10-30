# tuning_experiments.py
# Ulepszone eksperymenty pod A/B/C: corr+kbest, L1 LR, RFECV, cost-tuned threshold, mała siatka C dla SVM.
# Zależności: korzysta bezpośrednio z funkcji pipeline() z main.py

import json
import time
from pathlib import Path
from copy import deepcopy

from main import pipeline  # <-- używamy gotowego pipeline’u

OUT_DIR = Path("output") / "tuning_experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA = r"data_sources\liquid_biopsy_data.csv"

# Zestaw metryk wspólny
EVAL = ["AUC ROC", "accuracy", "F1", "recall", "precision"]

def run_cfg(cfg: dict):
    """Uruchamia pojedynczy eksperyment przez main.pipeline i zwraca (wynik, czas)."""
    t0 = time.time()
    res = pipeline(
        dane=cfg["data"],
        use_validation=cfg.get("use_validation", "separate"),
        preprocesing=cfg.get("preprocess", []),
        model_name=cfg["model"],
        model_params=cfg.get("model_params", {}),
        preprocess_params=cfg.get("preprocess_params", {}),
        postprocess_params=cfg.get("postprocess_params", {}),
        postprocess=cfg.get("postprocess", False),
        EVAL=cfg.get("eval", EVAL),
        XAI=cfg.get("xai", False),
    )
    dt = time.time() - t0
    return res, dt

def save_result(name: str, cfg: dict, res: dict, runtime_sec: float):
    """Zapisuje wynik do plików JSON + konsoliduje do CSV (append)."""
    # JSON
    payload = {
        "name": name,
        "cfg": cfg,
        "result": res,
        "runtime_sec": round(runtime_sec, 2),
    }
    (OUT_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2))

    # CSV (append)
    # cols: name, model, accuracy, f1, recall, precision, auc_roc, runtime
    row = {
        "name": name,
        "model": res.get("model"),
        "accuracy": res["metrics"].get("accuracy"),
        "f1": res["metrics"].get("f1"),
        "recall": res["metrics"].get("recall"),
        "precision": res["metrics"].get("precision"),
        "auc_roc": res["metrics"].get("AUC ROC"),
        "runtime_sec": round(runtime_sec, 2),
    }
    csv_path = OUT_DIR / "summary.csv"
    header = "name,model,accuracy,f1,recall,precision,auc_roc,runtime_sec\n"
    line = ",".join(str(row[k]) for k in ["name","model","accuracy","f1","recall","precision","auc_roc","runtime_sec"]) + "\n"
    if not csv_path.exists():
        csv_path.write_text(header + line)
    else:
        with csv_path.open("a", encoding="utf-8") as f:
            f.write(line)

def main():
    experiments = []

    # ===== Use Case A: F1 – ulepszenia =====
    # A1: LR (L1) + corr+kbest, tuning F1
    experiments.append({
        "name": "A1_LR_L1_corr_k1000_f1",
        "cfg": {
            "data": DATA,
            "use_validation": "separate",
            "preprocess": ["corr", "kbest"],
            "preprocess_params": {"corr_threshold": 0.90, "prefilter_k": 1000},
            "model": "LR",
            "model_params": {"solver": "liblinear", "penalty": "l1", "C": 0.3, "max_iter": 3000},
            "postprocess": True,
            "postprocess_params": {"tuning_method": "f1", "show_tuning_plots": False},
            "eval": EVAL,
            "xai": True
        }
    })

    # A2: LR (L2) + corr+kbest+RFECV, tuning F1
    experiments.append({
        "name": "A2_LR_L2_corr_k800_rfecv_f1",
        "cfg": {
            "data": DATA,
            "use_validation": "separate",
            "preprocess": ["corr", "kbest", "rfecv"],
            "preprocess_params": {"corr_threshold": 0.90, "prefilter_k": 800},
            "model": "LR",
            "model_params": {"solver": "liblinear", "penalty": "l2", "C": 0.3, "max_iter": 2000},
            "postprocess": True,
            "postprocess_params": {"tuning_method": "f1", "show_tuning_plots": False},
            "eval": EVAL,
            "xai": True
        }
    })

    # ===== Use Case B: Recall – podejście cost-sensitive =====
    # B1: LR (L2), corr+kbest, tuning 'cost' (waż FN)
    experiments.append({
        "name": "B1_LR_cost_corr_k1000_recall_mode",
        "cfg": {
            "data": DATA,
            "use_validation": "separate",
            "preprocess": ["corr", "kbest"],
            "preprocess_params": {"corr_threshold": 0.90, "prefilter_k": 1000},
            "model": "LR",
            "model_params": {"solver": "lbfgs", "penalty": "l2", "C": 1.0, "max_iter": 2000},
            "postprocess": True,
            "postprocess_params": {"tuning_method": "cost", "cost_fn": 5.0, "cost_fp": 1.0, "show_tuning_plots": False},
            "eval": EVAL + ["Confusion Matrix"],
            "xai": False
        }
    })

    # ===== Use Case C: Accuracy – SVM linear calibrated, mały grid po C =====
    C_GRID = [0.1, 0.3, 1.0, 3.0, 4.5, 6.0]
    for C in C_GRID:
        experiments.append({
            "name": f"C_SVM_lin_cal_corr_only_C{C}".replace(".", "p"),
            "cfg": {
                "data": DATA,
                "use_validation": "merge_train_test",
                "preprocess": ["corr"],  # szybciej; kbest opcjonalnie w kolejnym rzucie
                "preprocess_params": {"corr_threshold": 0.92},
                "model": "SVM",
                "model_params": {
                    "kernel": "linear",
                    "C": C,
                    "class_weight": "balanced",
                    "use_calibrated": True,
                    "calibration_method": "sigmoid",
                    "cv_calibration": 3
                },
                "postprocess": False,
                "postprocess_params": {},
                "eval": EVAL + ["AUC PR","Confusion Matrix"],
                "xai": True
            }
        })

    # ===== Uruchomienie wszystkich biegów =====
    best_by_obj = {
        "A_F1": {"name": None, "score": -1, "payload": None},
        "B_Recall": {"name": None, "score": -1, "payload": None},
        "C_Accuracy": {"name": None, "score": -1, "payload": None},
    }

    for exp in experiments:
        name = exp["name"]
        cfg = exp["cfg"]
        print(f"\n=== Running: {name} ===")
        try:
            res, dt = run_cfg(cfg)
            save_result(name, cfg, res, dt)

            # track bests
            acc = res["metrics"].get("accuracy", float("nan"))
            f1  = res["metrics"].get("f1", float("nan"))
            rec = res["metrics"].get("recall", float("nan"))

            # UC-A: maks F1 (use_validation=separate)
            if cfg.get("use_validation", "") == "separate" and f1 == f1 and f1 > best_by_obj["A_F1"]["score"]:
                best_by_obj["A_F1"] = {"name": name, "score": f1, "payload": {"cfg": cfg, "res": res}}

            # UC-B: maks Recall (use_validation=separate)
            if cfg.get("use_validation", "") == "separate" and rec == rec and rec > best_by_obj["B_Recall"]["score"]:
                best_by_obj["B_Recall"] = {"name": name, "score": rec, "payload": {"cfg": cfg, "res": res}}

            # UC-C: maks Accuracy (merge_train_test)
            if cfg.get("use_validation", "") == "merge_train_test" and acc == acc and acc > best_by_obj["C_Accuracy"]["score"]:
                best_by_obj["C_Accuracy"] = {"name": name, "score": acc, "payload": {"cfg": cfg, "res": res}}

        except Exception as e:
            # zapisujemy błąd zamiast wyniku
            err_payload = {"name": name, "cfg": cfg, "error": str(e)}
            (OUT_DIR / f"{name}_ERROR.json").write_text(json.dumps(err_payload, indent=2))
            print(f"[ERROR] {name}: {e}")

    # Podsumowanie „best of” do JSON
    (OUT_DIR / "best_of_summary.json").write_text(json.dumps(best_by_obj, indent=2))
    print("\n=== DONE ===")
    print("Summary CSV:", OUT_DIR / "summary.csv")
    print("Best-of JSON:", OUT_DIR / "best_of_summary.json")

if __name__ == "__main__":
    main()
