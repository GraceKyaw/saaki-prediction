"""
Cluster-aware bootstrap confidence intervals -- for the temporal-feature
FFNN only, where multiple windows come from the same patient.

bootstrap_ci and bootstrap_metrics_report (already in evaluation_utils.py)
resample individual rows, which assumes each row is independent. That
assumption does not hold here: windows from the same patient share
correlated risk scores, so resampling rows independently understates the
true uncertainty. These functions resample whole patients instead, so a
patient's windows move together in each resample rather than being
split apart as if they were unrelated.

Same principle as the other evaluation functions: only needs (true
labels, predicted probabilities, already computed) plus a patient ID per
row. Nothing is refit.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    recall_score, precision_score, f1_score,
)


def cluster_bootstrap_ci(y_true, y_score, patient_ids, metric_func,
                          n_bootstrap=1000, confidence=0.95, random_state=42):
    """
    Bootstrap confidence interval, resampling whole patients rather than
    individual rows.

    y_true      : true binary labels, one per window
    y_score     : predicted probabilities or predictions, one per window
    patient_ids : which patient each window belongs to
    metric_func : a function taking (y_true, y_score) and returning one
                  number

    Returns a dict with the point estimate, the interval bounds, and the
    bootstrap scores.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    patient_ids = np.asarray(patient_ids)
    n = len(y_true)

    if not (len(y_score) == n and len(patient_ids) == n):
        raise ValueError("y_true, y_score, and patient_ids must all be the same length")

    point_estimate = float(metric_func(y_true, y_score))

    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)

    # built once, so each bootstrap iteration looks up rows instead of
    # re-searching the full array every time
    patient_to_rows = {p: np.where(patient_ids == p)[0] for p in unique_patients}

    rng = np.random.default_rng(random_state)
    scores = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sampled_patients = rng.choice(unique_patients, size=n_patients, replace=True)
        idx = np.concatenate([patient_to_rows[p] for p in sampled_patients])
        scores[i] = metric_func(y_true[idx], y_score[idx])

    alpha = 1 - confidence
    ci_lower = float(np.percentile(scores, 100 * alpha / 2))
    ci_upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))

    return {
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
        "bootstrap_scores": scores,
    }


def cluster_bootstrap_metrics_report(y_true, y_proba, patient_ids, threshold=0.5,
                                      n_bootstrap=1000, confidence=0.95,
                                      random_state=42, label="model"):
    """
    Cluster-aware confidence intervals for AUPRC, AUROC, recall,
    specificity, precision and F1.

    Same six metrics and same printed/returned shape as
    bootstrap_metrics_report, but resampling whole patients instead of
    individual windows. Takes patient_ids in addition to labels and
    probabilities.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    patient_ids = np.asarray(patient_ids)
    y_pred = (y_proba >= threshold).astype(int)

    def specificity_score(yt, yp):
        return recall_score(yt, yp, pos_label=0)

    metrics = {
        "AUPRC": (y_proba, average_precision_score),
        "AUROC": (y_proba, roc_auc_score),
        "recall_AKI": (y_pred, recall_score),
        "specificity_non_AKI": (y_pred, specificity_score),
        "precision": (y_pred, precision_score),
        "F1": (y_pred, f1_score),
    }

    rows = []
    print(f"Cluster-aware bootstrap {int(confidence*100)}% confidence intervals: {label}")
    print(f"  ({n_bootstrap} resamples, {len(np.unique(patient_ids)):,} patients, "
          f"{len(y_true):,} windows)")
    for name, (scores, func) in metrics.items():
        result = cluster_bootstrap_ci(y_true, scores, patient_ids, func,
                                       n_bootstrap=n_bootstrap, confidence=confidence,
                                       random_state=random_state)
        rows.append({
            "metric": name,
            "point_estimate": round(result["point_estimate"], 4),
            "ci_lower": round(result["ci_lower"], 4),
            "ci_upper": round(result["ci_upper"], 4),
        })
        print(f"  {name:20s} {result['point_estimate']:.4f}  "
              f"[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    print()

    return pd.DataFrame(rows)


def cluster_subgroup_report(y_true, y_proba, group_values, patient_ids, threshold=0.5,
                             n_bootstrap=1000, confidence=0.95, random_state=42,
                             label="model"):
    """
    Runs cluster_bootstrap_metrics_report separately for each unique
    value in group_values.

    Same shape as subgroup_report, but cluster-aware throughout: calls
    cluster_bootstrap_metrics_report instead of bootstrap_metrics_report,
    so subgroup confidence intervals also account for correlated windows
    from the same patient, not just the whole-dataset result.
    """
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_proba = pd.Series(y_proba).reset_index(drop=True)
    group_values = pd.Series(group_values).reset_index(drop=True)
    patient_ids = pd.Series(patient_ids).reset_index(drop=True)

    if not (len(y_true) == len(y_proba) == len(group_values) == len(patient_ids)):
        raise ValueError("y_true, y_proba, group_values, and patient_ids must all be the same length")

    print(f"Subgroup performance (cluster-aware): {label}")
    print(f"  groups: {sorted(group_values.unique())}")
    print()

    all_results = []
    for group in sorted(group_values.unique()):
        mask = (group_values == group)
        group_n = int(mask.sum())

        result = cluster_bootstrap_metrics_report(
            y_true[mask].to_numpy(), y_proba[mask].to_numpy(), patient_ids[mask].to_numpy(),
            threshold=threshold, n_bootstrap=n_bootstrap,
            confidence=confidence, random_state=random_state,
            label=f"{label} -- {group} (n={group_n})",
        )
        result["subgroup"] = group
        result["group_n"] = group_n
        all_results.append(result)

    return pd.concat(all_results, ignore_index=True)
