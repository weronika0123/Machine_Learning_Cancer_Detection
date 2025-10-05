import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TunedThresholdClassifierCV

def plot_threshold_curve(tuned_model):
    thresholds = tuned_model.cv_results_["thresholds"]
    scores = tuned_model.cv_results_["scores"]

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, scores, label="Score vs threshold")
    plt.plot(
        tuned_model.best_threshold_,
        tuned_model.best_score_,
        "o",
        markersize=10,
        color="tab:orange",
        label=f"Best thr={tuned_model.best_threshold_:.3f}"
    )
    plt.xlabel("Threshold")
    plt.ylabel("Score (CV)")
    plt.title("Threshold tuning curve")
    plt.legend()
    plt.grid(ls=":")
    plt.tight_layout()
    plt.show()


def tune_threshold(
    model,
    X_val,
    y_val,
    X_test,
    scoring="f1",
    thresholds=500,
    plot_curve=False
):
    print("[POSTPROCESS] Threshold tuning on validation set...")
    
    tuning_info = {
        'best_threshold': None,
        'best_score': None,
        'tuned_model': None,
        'tuning_performed': False
    }
    
    #Check if validation set is empty
    if X_val.shape[0] == 0:
        print("[POSTPROCESS][WARN] Validation set is empty — skipping threshold tuning.")
        y_pred = model.predict(X_test)
        return y_pred, tuning_info
    
    #Determine response method
    if hasattr(model, "predict_proba"):
        response_method = "predict_proba"
    elif hasattr(model, "decision_function"):
        response_method = "decision_function"
    else:
        print("[POSTPROCESS][WARN] Model doesn't have predict_proba or decision_function — skipping threshold tuning.")
        y_pred = model.predict(X_test)
        return y_pred, tuning_info
    
    #Tune threshold on validation set
    tuned = TunedThresholdClassifierCV(
        estimator=model,                
        cv="prefit",                   
        scoring=scoring,               
        thresholds=thresholds,         
        response_method=response_method,
        refit=False                    
    )
    
    tuned.fit(X_val, y_val)
    print(f"[POSTPROCESS] Selected threshold={tuned.best_threshold_:.3f} (score={tuned.best_score_:.3f})")
    
    tuning_info['best_threshold'] = tuned.best_threshold_
    tuning_info['best_score'] = tuned.best_score_
    tuning_info['tuned_model'] = tuned
    tuning_info['tuning_performed'] = True
    
    if plot_curve:
        plot_threshold_curve(tuned)
    
    y_pred = tuned.predict(X_test) 
    return y_pred, tuning_info
