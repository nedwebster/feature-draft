from sklearn import metrics


METRICS_CONFIG = {
    metrics.roc_auc_score: max,
    metrics.mean_squared_error: min,
}
