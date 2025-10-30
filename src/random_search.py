import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from copy import deepcopy

# Importujemy tylko pipeline – nic nie zmieniamy w Twoim kodzie
from main import pipeline

RNG = random.Random(42)


def loguniform(rng, low, high):
    """Próbkowanie log-uniform (np. C, lr)."""
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def choice(rng, seq):
    return rng.choice(seq)


def randint(rng, a, b):
    return rng.randint(a, b)


def sample_preprocess_for_linear(rng):
    """Preprocessing dla LR/SVM: zawsze 'corr', opcjonalnie 'kbest'."""
    steps = ["corr"]
    params = {
        "corr_threshold": round(rng.uniform(0.85, 0.95), 2),
    }
    use_kbest = rng.random() < 0.7  # często warto odfiltrować
    if use_kbest:
        steps.append("kbest")
        params["prefilter_k"] = choice(rng, [200, 300, 500, 800, 1000, 1500])
    return steps, params


def sample_preprocess_for_tree(rng):
    """Preprocessing dla drzew: corr + (opcjonalnie) kbest."""
    steps = ["corr"]
    params = {
        "corr_threshold": round(rng.uniform(0.85, 0.95), 2),
    }
    if rng.random() < 0.5:
        steps.append("kbest")
        params["prefilter_k"] = choice(rng, [500, 800, 1000, 1500])
    return steps, params


def sample_lr_params(rng):
    # Solver ograniczamy do sprawdzonych na wielu cechach
    solver = choice(rng, ["lbfgs", "liblinear"])
    penalty = "l2"  # bezpiecznie i stabilnie
    C = loguniform(rng, 0.05, 5.0)
    return {"max_iter": 2000, "solver": solver, "penalty": penalty, "C": round(C, 6)}


def sample_svm_linear_params(rng, calibrated=True):
    C = loguniform(rng, 0.05, 5.0)
    params = {
        "kernel": "linear",
        "C": round(C, 6),
        "class_weight": "balanced",
    }
    if calibrated:
        params.update({"use_calibrated": True, "cv_calibration": choice(rng, [3, 5]), "calibration_method": "sigmoid"})
    else:
        params.update({"use_calibrated": False, "probability": True})
    return params


def sample_svm_rbf_params(rng):
    C = loguniform(rng, 0.05, 5.0)
    gamma = loguniform(rng, 1e-4, 1e-1)
    return {
        "kernel": "rbf",
        "C": round(C, 6),
        "gamma": round(gamma, 8),
        "class_weight": "balanced",
        "probability": True,
    }


def sample_dt_params(rng):
    return {
        "criterion": choice(rng, ["gini", "entropy"]),
        "max_depth": choice(rng, [None, 4, 6, 8, 10, 14]),
        "min_samples_leaf": choice(rng, [1, 2, 3, 5, 8, 12]),
        "class_weight": "balanced",
    }


def sample_dnn_params(rng, n_features_hint=None):
    layers_choices = [
        [128, 64],
        [256, 128],
        [256, 128, 64],
        [512, 256, 128],
    ]
    lr = loguniform(rng, 5e-5, 5e-3)
    return {
        "hidden_layers": choice(rng, layers_choices),
        "activation": "relu",
        "dropout_rate": round(rng.uniform(0.15, 0.4), 2),
        "learning_rate": round(lr, 6),
        "epochs": choice(rng, [20, 30, 40, 50, 60]),
        "batch_size": choice(rng, [32, 48, 64, 96]),
    }


def objective_key(use_case):
    if use_case == "A":
        return "f1"
    if use_case == "B":
        return "recall"
    if use_case == "C":
        return "accuracy"
    raise ValueError("use_case must be one of: A, B, C")


def build_eval_list(use_case):
    base = ["AUC ROC", "accuracy", "F1", "recall", "precision"]
    if use_case in ("B", "C"):
        base.append("Confusion Matrix")
    return base


def run_trial(
    data_path: str,
    use_case: str,
    model_name: str,
    model_params: dict,
    preprocess_steps: list,
    preprocess_params: dict,
    rng: random.Random,
):
    """
    Składa pojedynczy bieg pipeline pod dany use-case.
    Zwraca (score, metrics, full_cfg).
    """
    # Ustawienia per use-case
    if use_case in ("A", "B"):
        use_validation = "separate"
        postprocess = True
        tuning_method = "f1" if use_case == "A" else "recall"
        postprocess_params = {"tuning_method": tuning_method, "show_tuning_plots": False}
    elif use_case == "C":
        use_validation = "merge_train_test"
        postprocess = False
        postprocess_params = {}
    else:
        raise ValueError("Unknown use_case")

    # XAI: tylko gdy ma sens i nie spowalnia
    xai = model_name in ("LR", "SVM") and use_case != "B"

    # Wywołanie Twojego pipeline
    out = pipeline(
        dane=data_path,
        use_validation=use_validation,
        preprocesing=preprocess_steps,
        model_name=model_name,
        model_params=model_params,
        preprocess_params=preprocess_params,
        postprocess_params=postprocess_params,
        postprocess=postprocess,
        EVAL=build_eval_list(use_case),
        XAI=xai,
    )

    metrics = out.get("metrics", {})
    key = objective_key(use_case)
    score = float(metrics.get(key, float("-inf")))
    return score, metrics, {
        "data": data_path,
        "use_case": use_case,
        "use_validation": use_validation,
        "preprocess": preprocess_steps,
        "preprocess_params": preprocess_params,
        "model": model_name,
        "model_params": model_params,
        "postprocess": postprocess,
        "postprocess_params": postprocess_params,
        "eval": build_eval_list(use_case),
        "xai": xai,
    }


def search_space(use_case, rng):
    """
    Generator konfiguracji (model + preproc + hiperparametry) z przewagą
    dla wariantów sensownych w danym use-case.
    """
    # Priorytety modeli per use-case
    if use_case == "A":  # F1
        model_order = ["LR", "SVM", "DT", "DNN"]
    elif use_case == "B":  # Recall
        model_order = ["LR", "SVM", "DT", "DNN"]
    else:  # "C" Accuracy
        model_order = ["SVM", "LR", "DT", "DNN"]

    while True:
        model_name = choice(rng, model_order)

        if model_name == "LR":
            preproc, preproc_params = sample_preprocess_for_linear(rng)
            mparams = sample_lr_params(rng)

        elif model_name == "SVM":
            preproc, preproc_params = sample_preprocess_for_linear(rng)
            # W use-case C faworyzujemy linear+calibration, w A/B dopuszczamy też RBF
            if use_case == "C" or rng.random() < 0.7:
                mparams = sample_svm_linear_params(rng, calibrated=True)
            else:
                mparams = sample_svm_rbf_params(rng)

        elif model_name == "DT":
            preproc, preproc_params = sample_preprocess_for_tree(rng)
            mparams = sample_dt_params(rng)

        elif model_name == "DNN":
            # dla DNN zwykle dobry jest kbest żeby zredukować p
            preproc = ["corr", "kbest"]
            preproc_params = {
                "corr_threshold": round(rng.uniform(0.85, 0.95), 2),
                "prefilter_k": choice(rng, [100, 200, 300, 500]),
            }
            mparams = sample_dnn_params(rng)

        else:
            # fallback na LR
            preproc, preproc_params = sample_preprocess_for_linear(rng)
            mparams = sample_lr_params(rng)
            model_name = "LR"

        yield model_name, mparams, preproc, preproc_params


def main():
    parser = argparse.ArgumentParser(description="Lightweight random search wrapper (no code changes needed).")
    parser.add_argument("--data", required=True, help="Ścieżka do CSV (np. data_sources\\liquid_biopsy_data.csv)")
    parser.add_argument("--use_case", choices=["A", "B", "C"], required=True,
                        help="A=max F1, B=max Recall, C=max Accuracy")
    parser.add_argument("--trials", type=int, default=20, help="Liczba losowych prób")
    parser.add_argument("--seed", type=int, default=42, help="Seed RNG")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    outdir = Path("output")
    outdir.mkdir(exist_ok=True, parents=True)
    csv_path = outdir / f"random_search_log_{args.use_case}.csv"
    best_path = outdir / f"random_search_best_{args.use_case}.json"

    # nagłówek CSV
    if not csv_path.exists():
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("trial,use_case,model,score,accuracy,f1,recall,precision,auc_roc,details_json\n")

    best = {"score": float("-inf"), "cfg": None, "metrics": None}
    gen = search_space(args.use_case, rng)

    for t in range(1, args.trials + 1):
        model_name, model_params, preproc, preproc_params = next(gen)
        t0 = time.time()
        try:
            score, metrics, cfg = run_trial(
                data_path=args.data,
                use_case=args.use_case,
                model_name=model_name,
                model_params=model_params,
                preprocess_steps=preproc,
                preprocess_params=preproc_params,
                rng=rng,
            )
        except Exception as e:
            # zapis błędu do loga (jako wiersz z NaN score)
            with open(csv_path, "a", encoding="utf-8") as f:
                err = {"error": str(e), "model": model_name, "model_params": model_params,
                       "preprocess": preproc, "preprocess_params": preproc_params}
                f.write(f'{t},{args.use_case},{model_name},NaN,NaN,NaN,NaN,NaN,NaN,{json.dumps(err)}\n')
            print(f"[TRIAL {t}] ERROR: {e}")
            continue

        dt = time.time() - t0
        print(f"[TRIAL {t}] {model_name} | {objective_key(args.use_case)}={score:.4f} | {dt:.1f}s")

        # zapis do CSV
        with open(csv_path, "a", encoding="utf-8") as f:
            row = {
                "trial": t,
                "use_case": args.use_case,
                "model": model_name,
                "score": score,
                "accuracy": metrics.get("accuracy"),
                "f1": metrics.get("f1"),
                "recall": metrics.get("recall"),
                "precision": metrics.get("precision"),
                "auc_roc": metrics.get("AUC ROC", metrics.get("AUC ROC".lower())),
                "details_json": {
                    "cfg": cfg,
                    "metrics": metrics,
                    "runtime_sec": round(dt, 2),
                }
            }
            f.write(
                f'{row["trial"]},{row["use_case"]},{row["model"]},{row["score"]},'
                f'{row["accuracy"]},{row["f1"]},{row["recall"]},{row["precision"]},{row["auc_roc"]},'
                f'{json.dumps(row["details_json"])}\n'
            )

        # update best
        if score > best["score"]:
            best["score"] = score
            best["cfg"] = deepcopy(cfg)
            best["metrics"] = deepcopy(metrics)
            with open(best_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "use_case": args.use_case,
                        "objective": objective_key(args.use_case),
                        "best_score": best["score"],
                        "best_config": best["cfg"],
                        "best_metrics": best["metrics"],
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            print(f"[BEST] New best for UC-{args.use_case}: {best['score']:.4f} saved to {best_path}")

    print("\n=== DONE ===")
    if best["cfg"] is None:
        print("No successful trials. Check random_search_log for errors.")
    else:
        print(f"Best {objective_key(args.use_case)} = {best['score']:.4f}")
        print(f"Config: {json.dumps(best['cfg'], indent=2)}")
        print(f"Metrics: {json.dumps(best['metrics'], indent=2)}")


if __name__ == "__main__":
    main()
