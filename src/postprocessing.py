import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    recall_score, precision_score, f1_score, fbeta_score
)
import warnings

#region Threshold Optimization Core

def threshold_tuning(
    model,
    X_val,
    y_val,
    X_test,
    y_test,
    method='recall',
    cost_fn=10.0,
    cost_fp=1.0,
    thresholds=500,
    show_plots=True,
    output_dir=None
):
    
    print("[POSTPROCESS] Starting threshold tuning...")
    
    #Define supported methods
    METHODS_WITH_PLOTS = ['recall', 'f1', 'f2', 'youden']
    METHODS_WITHOUT_PLOTS = ['cost']
    ALL_SUPPORTED_METHODS = METHODS_WITH_PLOTS + METHODS_WITHOUT_PLOTS
    
    #Validate method is supported
    if method not in ALL_SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported tuning method: '{method}'. "
            f"Supported methods are: {', '.join(ALL_SUPPORTED_METHODS)}"
        )
    
    #Validate visualization compatibility
    if method in METHODS_WITHOUT_PLOTS and show_plots:
        raise ValueError(
            f"Method '{method}' does not support visualization. "
            f"Please set 'show_tuning_plots': False in --params, or use one of these methods with plots: "
            f"{', '.join(METHODS_WITH_PLOTS)}"
        )
    
    #Validate thresholds parameter
    if not isinstance(thresholds, int) or thresholds < 10:
        raise ValueError("Parameter 'thresholds' must be an integer >= 10")
    
    #Validate cost parameters (only for cost method)
    if method == 'cost':
        if cost_fn <= 0 or cost_fp <= 0:
            raise ValueError("Parameters 'cost_fn' and 'cost_fp' must be positive values")
    
    info = {
        'best_threshold': None,
        'best_score': None,
        'method_used': method,
        'metrics_before': {},
        'metrics_after': {},
        'tuning_performed': False
    }
    
    #Validate inputs
    validation_error = _validate_inputs(X_val, y_val, model)
    if validation_error:
        print(f"[POSTPROCESS][WARN] {validation_error} — skipping threshold tuning.")
        y_pred = model.predict(X_test)
        return y_pred, info
    
    y_val_proba = _get_probabilities(model, X_val)
    y_test_proba = _get_probabilities(model, X_test)
    
    if y_val_proba is None or y_test_proba is None:
        print("[POSTPROCESS][WARN] Cannot extract probabilities-skipping threshold tuning.")
        y_pred = model.predict(X_test)
        return y_pred, info
    
    #Calculate metrics before tuning (with default threshold 0.5)
    y_test_pred_default = (y_test_proba >= 0.5).astype(int)
    info['metrics_before'] = _calculate_metrics(y_test, y_test_pred_default)
    
    threshold_range = np.linspace(0.01, 0.99, thresholds)
    
    if method == 'recall':
        best_threshold, best_score = _optimize_recall(
            y_val, y_val_proba, threshold_range
        )
    elif method == 'f1':
        best_threshold, best_score = _optimize_fbeta(
            y_val, y_val_proba, threshold_range, beta=1.0
        )
    elif method == 'f2':
        best_threshold, best_score = _optimize_fbeta(
            y_val, y_val_proba, threshold_range, beta=2.0
        )
    elif method == 'youden':
        best_threshold, best_score = _optimize_youden(
            y_val, y_val_proba
        )
    elif method == 'cost':
        best_threshold, best_score = _optimize_cost_sensitive(
            y_val, y_val_proba, threshold_range, cost_fn, cost_fp
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'recall', 'f1', 'f2', 'youden', or 'cost'.")
    
    print(f"[POSTPROCESS] Method: {method} | Best threshold: {best_threshold:.3f} | Score: {best_score:.3f}")
    
    #Apply optimal threshold to test set
    y_pred = (y_test_proba >= best_threshold).astype(int)
    
    #Calculate metrics after tuning
    info['metrics_after'] = _calculate_metrics(y_test, y_pred)
    info['best_threshold'] = float(best_threshold)
    info['best_score'] = float(best_score)
    info['tuning_performed'] = True
    
    #Display improvement
    _print_improvement(info['metrics_before'], info['metrics_after'])
    
    #Visualizations
    if show_plots and output_dir is not None:
        _create_visualizations(
            y_val, y_val_proba, y_test, y_test_proba,
            best_threshold, threshold_range, method, output_dir
        )
    
    return y_pred, info


def _optimize_recall(y_true, y_proba, threshold_range):
    scores = []
    for thresh in threshold_range:
        y_pred = (y_proba >= thresh).astype(int)
        score = recall_score(y_true, y_pred, zero_division=0)
        scores.append(score)
    scores = np.array(scores)
    best_idx = np.argmax(scores)
    return threshold_range[best_idx], scores[best_idx]


def _optimize_fbeta(y_true, y_proba, threshold_range, beta=1.0):
    scores = []
    for thresh in threshold_range:
        y_pred = (y_proba >= thresh).astype(int)
        score = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        scores.append(score)
    scores = np.array(scores)
    best_idx = np.argmax(scores)
    return threshold_range[best_idx], scores[best_idx]


def _optimize_youden(y_true, y_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx], j_scores[best_idx]

def _optimize_cost_sensitive(y_true, y_proba, threshold_range, cost_fn, cost_fp):
    costs = []
    for thresh in threshold_range:
        y_pred = (y_proba >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        total_cost = fn * cost_fn + fp * cost_fp
        costs.append(total_cost)
    
    costs = np.array(costs)
    best_idx = np.argmin(costs)
    return threshold_range[best_idx], -costs[best_idx] 

#region Visualization
def _create_visualizations(y_val, y_val_proba, y_test, y_test_proba, 
                          best_threshold, threshold_range, method, output_dir):
    """Create comprehensive visualization plots."""
    fig = plt.figure(figsize=(16, 5))
    
    # Subplot 1: Threshold vs Metric Curve
    ax1 = plt.subplot(1, 3, 1)
    _plot_threshold_curve(ax1, y_val, y_val_proba, threshold_range, best_threshold, method)
    # Subplot 2: ROC Curve with Optimal Point
    ax2 = plt.subplot(1, 3, 2)
    _plot_roc_with_optimal(ax2, y_test, y_test_proba, best_threshold)
    # Subplot 3: Precision-Recall Curve with Optimal Point
    ax3 = plt.subplot(1, 3, 3)
    _plot_pr_with_optimal(ax3, y_test, y_test_proba, best_threshold)
    
    plt.tight_layout()
    plot_path = output_dir / "threshold_tuning_plots.png"
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    print(f"[POSTPROCESS] Threshold tuning plots saved to {plot_path}")

def _plot_threshold_curve(ax, y_true, y_proba, threshold_range, best_threshold, method):
    #Calculate scores for all thresholds
    if method in ['recall', 'f1', 'f2']:
        if method == 'recall':
            metric_name = 'Recall'
            scores = [recall_score(y_true, (y_proba >= t).astype(int), zero_division=0) 
                     for t in threshold_range]
        elif method == 'f1':
            metric_name = 'F1-Score'
            scores = [f1_score(y_true, (y_proba >= t).astype(int), zero_division=0) 
                     for t in threshold_range]
        elif method == 'f2':
            metric_name = 'F2-Score'
            scores = [fbeta_score(y_true, (y_proba >= t).astype(int), beta=2.0, zero_division=0) 
                     for t in threshold_range]
        
        ax.plot(threshold_range, scores, linewidth=2.5, color='#2E86AB', label=f'{metric_name}')
    elif method == 'youden':
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        j_scores = tpr - fpr
        ax.plot(thresholds, j_scores, linewidth=2.5, color='#2E86AB', label="Youden's J")
    
    #Mark optimal point
    if method != 'youden':
        best_idx = np.argmin(np.abs(threshold_range - best_threshold))
        ax.scatter([best_threshold], [scores[best_idx]], s=60, c='#D8504D', 
                  marker='o', zorder=3,
                  label=f'Optimal: {best_threshold:.3f}')
    
    ax.axvline(best_threshold, ls='--', color='#D8504D', linewidth=2, alpha=0.7)
    ax.set_xlabel('Threshold', fontsize=11)
    ax.set_ylabel('Score (Validation)', fontsize=11)
    ax.set_title(f'{method.capitalize()} Optimization Curve', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, ls=':', alpha=0.6)

def _plot_roc_with_optimal(ax, y_true, y_proba, best_threshold):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    best_idx = np.argmin(np.abs(thresholds - best_threshold))
    
    ax.plot(fpr, tpr, linewidth=2.5, color='#2E86AB', label=f'ROC (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random Classifier')
    ax.scatter([fpr[best_idx]], [tpr[best_idx]], s=60, c='#D8504D', marker='o', 
              zorder=3,
              label=f'Optimal (thr={best_threshold:.3f})')
    
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve with Optimal Point', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, ls=':', alpha=0.6)

def _plot_pr_with_optimal(ax, y_true, y_proba, best_threshold):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    best_idx = np.argmin(np.abs(thresholds - best_threshold))
    
    ax.plot(recall, precision, linewidth=2.5, color='#2E86AB', label=f'PR (AUC={pr_auc:.3f})')
    ax.scatter([recall[best_idx]], [precision[best_idx]], s=60, c='#D8504D', marker='o', 
              zorder=3,
              label=f'Optimal (thr={best_threshold:.3f})')
    
    ax.set_xlabel('Recall (Sensitivity)', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision-Recall Curve with Optimal Point', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, ls=':', alpha=0.6)

#region Utilities

def _validate_inputs(X_val, y_val, model):
    """Validate input data and model for threshold tuning."""
    if X_val.shape[0] == 0:
        return "Validation set is empty"
    if not (hasattr(model, 'predict_proba') or hasattr(model, 'decision_function')):
        return "Model doesn't have predict_proba or decision_function"
    return None

def _get_probabilities(model, X):
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            #Return probability of positive class
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba[:, 1]
            else:
                return proba.ravel()
        elif hasattr(model, 'decision_function'):
            #For SVM decision_function, normalize to [0, 1] range
            decision = model.decision_function(X)
            #Simple normalization
            return (decision - decision.min()) / (decision.max() - decision.min() + 1e-10)
    except Exception as e:
        warnings.warn(f"Error extracting probabilities: {e}")
        return None

    return None

def _calculate_metrics(y_true, y_pred):
    return {
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'f2': float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))
    }

def _print_improvement(metrics_before, metrics_after):
    print("\n" + "="*70)
    print("[POSTPROCESS] Performance Improvement (Test Set)")
    print("="*70)
    print(f"{'Metric':<12} | {'Before (0.5)':>12} | {'After (tuned)':>14} | {'Change':>10}")
    print("-"*70)
    
    for metric in ['recall', 'precision', 'f1', 'f2']:
        before = metrics_before[metric]
        after = metrics_after[metric]
        change = after - before
        sign = '+' if change >= 0 else ''
        print(f"{metric.upper():<12} | {before:>12.4f} | {after:>14.4f} | {sign}{change:>9.4f}")
    
    print("="*70 + "\n")
