import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

def classification_metrics(y_true, y_pred, y_prob=None):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred)
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics['roc_auc'] = float('nan')
        try:
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        except Exception:
            metrics['pr_auc'] = float('nan')
    else:
        metrics['roc_auc'] = float('nan')
        metrics['pr_auc'] = float('nan')
    return metrics

def find_global_threshold(y_true, y_prob, n_thresholds=100):
    """
    Tìm threshold tối ưu (global) trên tập validation để phân loại nhị phân.
    Trả về threshold cho accuracy cao nhất.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    best_acc = 0.0
    best_t = 0.5
    min_prob, max_prob = y_prob.min(), y_prob.max()
    thresholds = np.linspace(min_prob, max_prob, n_thresholds)
    for t in thresholds:
        y_pred = (y_prob > t).astype(int)
        acc = (y_pred == y_true).mean()
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return best_t

