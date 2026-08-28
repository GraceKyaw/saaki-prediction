"""
Evaluation helpers for the training notebooks.

Each function takes true labels and already-computed predictions. Nothing here
fits, refits or resamples a model.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score,
    precision_score, recall_score, roc_auc_score,
)

# Colour order for chart series. Index position maps to series position.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def calibration_report(y_true, y_proba, label="model", n_bins=10, strategy="quantile"):
    """
    Bin predicted probabilities against observed outcomes.

    Bins by "quantile" (equal count) or "uniform" (equal width). Prints Brier,
    ECE and MCE, and returns one row per bin with count, mean predicted,
    observed rate and gap.
    """
    y_true = np.asarray(y_true).astype(float)
    y_proba = np.asarray(y_proba, dtype=float)

    if y_true.shape != y_proba.shape:
        raise ValueError(
            f"y_true and y_proba must be the same length, "
            f"got {y_true.shape} and {y_proba.shape}"
        )
    if not ((y_proba >= 0) & (y_proba <= 1)).all():
        raise ValueError("y_proba must contain probabilities in [0, 1]")

    frame = pd.DataFrame({"y": y_true, "p": y_proba})

    if strategy == "quantile":
        frame["bin"] = pd.qcut(frame["p"], q=n_bins, duplicates="drop")
    elif strategy == "uniform":
        frame["bin"] = pd.cut(frame["p"], bins=np.linspace(0, 1, n_bins + 1),
                              include_lowest=True)
    else:
        raise ValueError(f"strategy must be 'quantile' or 'uniform', got {strategy!r}")

    bins = (
        frame.groupby("bin", observed=True)
             .agg(n=("y", "size"), mean_predicted=("p", "mean"), observed=("y", "mean"))
             .reset_index(drop=True)
    )
    bins["gap"] = bins["observed"] - bins["mean_predicted"]
    bins.attrs["label"] = label

    # ECE weights each bin by its share of cases; MCE takes the largest gap
    weights = bins["n"] / bins["n"].sum()
    ece = float((weights * bins["gap"].abs()).sum())
    mce = float(bins["gap"].abs().max())
    brier = float(brier_score_loss(y_true, y_proba))

    print(f"calibration: {label}")
    print(f"  Brier score {brier:.4f}   ECE {ece:.4f}   MCE {mce:.4f}")
    print(f"  base rate {y_true.mean():.4f}   mean predicted {y_proba.mean():.4f}")
    print(f"  {len(bins)} bins ({strategy}), n = {int(bins['n'].sum()):,}")
    print(bins.round(4).to_string(index=False))
    print()

    bins.attrs.update({"brier": brier, "ece": ece, "mce": mce})
    return bins


def plot_calibration(series, title="Calibration on the held-out test set",
                     figsize=(6.4, 6.0)):
    """
    Draw a reliability diagram from calibration_report output.

    Takes a list of (bins, label) pairs. Displays the chart and returns the
    figure.
    """
    if not series:
        raise ValueError("series is empty: pass at least one (bins, label) pair")
    if len(series) > len(SERIES_COLORS):
        raise ValueError(
            f"{len(series)} series requested but only {len(SERIES_COLORS)} "
            f"colours are defined. Add an explicit colour to SERIES_COLORS "
            f"rather than cycling or generating one."
        )

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray",
            label="perfect calibration")

    for i, (bins, label) in enumerate(series):
        ax.plot(bins["mean_predicted"], bins["observed"],
                color=SERIES_COLORS[i], marker="o", label=label)

    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed positive rate")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    plt.show()
    return fig


# ---------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------


def bootstrap_ci(y_true, y_score, metric_func, n_bootstrap=1000,
                 confidence=0.95, random_state=42):
    """
    Bootstrap confidence interval for one metric.

    Takes labels, scores and a metric function of (y_true, y_score). Returns a
    dict with the point estimate, the interval bounds and the bootstrap scores.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)

    if len(y_score) != n:
        raise ValueError(f"y_true and y_score must be the same length, got {n} and {len(y_score)}")

    point_estimate = float(metric_func(y_true, y_score))

    rng = np.random.default_rng(random_state)
    scores = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)  # resample with replacement
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


def bootstrap_metrics_report(y_true, y_proba, threshold=0.5, n_bootstrap=1000,
                             confidence=0.95, random_state=42, label="model"):
    """
    Bootstrap intervals for AUPRC, AUROC, recall, specificity, precision and F1.

    Takes labels, probabilities and the threshold used for the threshold-based
    metrics. Prints a summary and returns one row per metric.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
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
    print(f"Bootstrap {int(confidence*100)}% confidence intervals: {label}")
    print(f"  ({n_bootstrap} resamples, n = {len(y_true):,} per resample)")
    for name, (scores, func) in metrics.items():
        result = bootstrap_ci(y_true, scores, func, n_bootstrap=n_bootstrap,
                              confidence=confidence, random_state=random_state)
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


# ---------------------------------------------------------------------
# Subgroup performance
# ---------------------------------------------------------------------


def subgroup_report(y_true, y_proba, group_values, threshold=0.5,
                    n_bootstrap=1000, confidence=0.95, random_state=42,
                    label="model"):
    """
    Run bootstrap_metrics_report separately for each value in group_values.

    Takes labels, probabilities and a subgroup label per patient. Returns the
    combined DataFrame with added "subgroup" and "group_n" columns.
    """
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_proba = pd.Series(y_proba).reset_index(drop=True)
    group_values = pd.Series(group_values).reset_index(drop=True)

    if not (len(y_true) == len(y_proba) == len(group_values)):
        raise ValueError("y_true, y_proba, and group_values must all be the same length")

    print(f"Subgroup performance: {label}")
    print(f"  groups: {sorted(group_values.unique(), key=str)}")
    print()

    all_results = []
    for group in sorted(group_values.unique(), key=str):
        mask = (group_values == group)
        group_n = int(mask.sum())

        result = bootstrap_metrics_report(
            y_true[mask].to_numpy(), y_proba[mask].to_numpy(),
            threshold=threshold, n_bootstrap=n_bootstrap,
            confidence=confidence, random_state=random_state,
            label=f"{label} -- {group} (n={group_n})",
        )
        result["subgroup"] = group
        result["group_n"] = group_n
        all_results.append(result)

    return pd.concat(all_results, ignore_index=True)


# ---------------------------------------------------------------------
# Decision curve analysis
# ---------------------------------------------------------------------


def decision_curve(y_true, y_proba, thresholds=None):
    """
    Net benefit at each threshold, for the model and for treating everyone.

    Takes labels, probabilities and optional thresholds (default 0.01 to 0.99).
    Returns one row per threshold.
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    prevalence = y_true.mean()

    net_benefit_model = []
    net_benefit_all = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        true_positives = ((y_pred == 1) & (y_true == 1)).sum()
        false_positives = ((y_pred == 1) & (y_true == 0)).sum()

        # Vickers & Elkin (2006): TP/n - (FP/n) * (t / (1 - t))
        nb_model = (true_positives / n) - (false_positives / n) * (t / (1 - t))
        nb_all = prevalence - (1 - prevalence) * (t / (1 - t))

        net_benefit_model.append(nb_model)
        net_benefit_all.append(nb_all)

    return pd.DataFrame({
        "threshold": thresholds,
        "net_benefit_model": net_benefit_model,
        "net_benefit_all": net_benefit_all,
        "net_benefit_none": 0.0,
    })


def plot_decision_curve(curves, title="Decision Curve Analysis"):
    """
    Plot decision curves with treat-everyone and treat-no-one reference lines.

    Takes a list of (curve, label) pairs from decision_curve. Displays the
    chart and returns the figure.
    """
    fig, ax = plt.subplots(figsize=(6.4, 6.0))

    # reference lines are model-independent, so take them from the first curve
    first_curve = curves[0][0]
    ax.plot(first_curve["threshold"], first_curve["net_benefit_all"],
            linestyle="--", color="gray", label="Treat everyone")
    ax.plot(first_curve["threshold"], first_curve["net_benefit_none"],
            linestyle=":", color="black", label="Treat no one")

    for i, (curve_df, label) in enumerate(curves):
        ax.plot(curve_df["threshold"], curve_df["net_benefit_model"],
                color=SERIES_COLORS[i], marker="", label=label)

    # cap the y-axis just above the highest curve value
    all_model_values = np.concatenate([c["net_benefit_model"].to_numpy() for c, _ in curves])
    y_max = max(all_model_values.max(), first_curve["net_benefit_all"].max()) * 1.15
    ax.set_ylim(-0.05, y_max)

    ax.set_xlabel("threshold probability")
    ax.set_ylabel("net benefit")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    plt.show()
    return fig


# ---------------------------------------------------------------------
# Cluster-aware bootstrap confidence intervals
# ---------------------------------------------------------------------
#
# For representations with several rows per patient. bootstrap_ci
# resamples individual rows, which assumes each row is independent. These
# functions resample whole patients instead, so a patient's rows move
# together in each resample.


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


# ---------------------------------------------------------------------
# Permutation feature importance
# ---------------------------------------------------------------------


def permutation_importance(model, X_test, y_test, feature_names, metric_func,
                           n_repeats=10, random_state=42, predict_fn=None):
    """
    For each feature, shuffle its values n_repeats times, measure the drop in
    metric_func's score each time, and return the average drop as that
    feature's importance. Larger drop means more important.

    X_test may be 2D (rows by features) or 3D (rows by timesteps by features).
    For 3D input the feature's whole per-row block is permuted as one unit, so
    the entire trajectory moves together. predict_fn overrides how predictions
    are obtained, for models taking more than one input.
    """
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    if X_test.ndim not in (2, 3):
        raise ValueError(f"X_test must be 2D or 3D, got {X_test.ndim}D")
    if X_test.shape[-1] != len(feature_names):
        raise ValueError(
            f"feature_names has {len(feature_names)} entries but X_test has "
            f"{X_test.shape[-1]} features"
        )

    if predict_fn is None:
        def predict_fn(X):
            return np.asarray(model.predict(X, verbose=0)).ravel()

    baseline = float(metric_func(y_test, predict_fn(X_test)))
    rng = np.random.default_rng(random_state)
    n_rows = X_test.shape[0]

    rows = []
    for j, name in enumerate(feature_names):
        drops = np.empty(n_repeats)
        for r in range(n_repeats):
            X_perm = X_test.copy()
            order = rng.permutation(n_rows)
            if X_test.ndim == 2:
                X_perm[:, j] = X_test[:, j][order]
            else:
                X_perm[:, :, j] = X_test[:, :, j][order]
            drops[r] = baseline - float(metric_func(y_test, predict_fn(X_perm)))
        rows.append({
            "feature": name,
            "importance": float(drops.mean()),
            "importance_std": float(drops.std()),
        })

    result = pd.DataFrame(rows).sort_values("importance", ascending=False)
    result.attrs["baseline"] = baseline
    result.attrs["n_repeats"] = n_repeats
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------
# Bootstrap confidence interval for a median
# ---------------------------------------------------------------------


def bootstrap_median_ci(values, n_bootstrap=1000, confidence=0.95, random_state=42):
    """
    Bootstrap confidence interval for the median of one array.

    Takes the values, resamples them with replacement, and returns a dict with
    the median, the interval bounds, and the number of values it rests on. A
    small n gives a wide interval, so n is returned rather than left implicit.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)

    if n == 0:
        return {
            "median": float("nan"), "ci_lower": float("nan"),
            "ci_upper": float("nan"), "n": 0,
            "confidence": confidence, "n_bootstrap": n_bootstrap,
            "bootstrap_medians": np.empty(0),
        }

    rng = np.random.default_rng(random_state)
    medians = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)  # resample with replacement
        medians[i] = np.median(values[idx])

    alpha = 1 - confidence
    return {
        "median": float(np.median(values)),
        "ci_lower": float(np.percentile(medians, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(medians, 100 * (1 - alpha / 2))),
        "n": int(n),
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
        "bootstrap_medians": medians,
    }


# ---------------------------------------------------------------------
# Feature importance and subgroup charts
# ---------------------------------------------------------------------


def plot_feature_importance(series, top_n=10, title="Feature importance",
                            figsize=(9, 6)):
    """
    Horizontal bar chart of the highest-importance features.

    Takes a list of (feature_names, importance_values, label) tuples. Features
    are ranked by the first series and the same features are shown for each
    series, so the bars line up. Displays the chart and returns the figure.
    """
    if not series:
        raise ValueError("series is empty: pass at least one "
                         "(feature_names, importance_values, label) tuple")
    if len(series) > len(SERIES_COLORS):
        raise ValueError(
            f"{len(series)} series requested but only {len(SERIES_COLORS)} "
            f"colours are defined. Add an explicit colour to SERIES_COLORS."
        )

    lookups = []
    for names, values, label in series:
        if len(names) != len(values):
            raise ValueError(f"{label}: {len(names)} names but {len(values)} values")
        lookups.append((dict(zip(names, values)), label))

    ranking = sorted(lookups[0][0].items(), key=lambda kv: kv[1], reverse=True)
    features = [name for name, _ in ranking[:top_n]][::-1]

    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(len(features))
    height = 0.8 / len(lookups)

    for i, (lookup, label) in enumerate(lookups):
        offset = (i - (len(lookups) - 1) / 2) * height
        ax.barh(positions + offset, [lookup.get(f, 0.0) for f in features],
                height, color=SERIES_COLORS[i], label=label)

    ax.set_yticks(positions)
    ax.set_yticklabels(features)
    ax.set_xlabel("importance")
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    plt.show()
    return fig


def plot_subgroup_specificity(series, metric="specificity_non_AKI",
                              title="Specificity by age band", figsize=(9, 6)):
    """
    Line chart of one metric across ordered subgroups, one line per model.

    Takes a list of (subgroup_labels, values, label) tuples, subgroups given in
    the order they should appear. Displays the chart and returns the figure.
    """
    if not series:
        raise ValueError("series is empty: pass at least one "
                         "(subgroups, values, label) tuple")
    if len(series) > len(SERIES_COLORS):
        raise ValueError(
            f"{len(series)} series requested but only {len(SERIES_COLORS)} "
            f"colours are defined. Add an explicit colour to SERIES_COLORS."
        )

    fig, ax = plt.subplots(figsize=figsize)
    for i, (subgroups, values, label) in enumerate(series):
        ax.plot(list(subgroups), list(values), marker="o",
                color=SERIES_COLORS[i], label=label)

    ax.set_xlabel("subgroup")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    plt.show()
    return fig


def plot_feature_importance_grid(series, top_n=10, title="Feature importance",
                                 ncols=3, panel_size=(6.0, 4.2)):
    """
    One small panel per model, each ranked and labelled on its own axis.

    Takes a list of (feature_names, importance_values, label) tuples. Use this
    where the series do not share a feature vocabulary, so a single shared
    category axis would leave most series with nothing to plot.
    Displays the chart and returns the figure.
    """
    if not series:
        raise ValueError("series is empty: pass at least one "
                         "(feature_names, importance_values, label) tuple")
    if len(series) > len(SERIES_COLORS):
        raise ValueError(
            f"{len(series)} series requested but only {len(SERIES_COLORS)} "
            f"colours are defined. Add an explicit colour to SERIES_COLORS."
        )

    nrows = int(np.ceil(len(series) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
    flat = np.atleast_1d(axes).ravel()

    for i, (names, values, label) in enumerate(series):
        if len(names) != len(values):
            raise ValueError(f"{label}: {len(names)} names but {len(values)} values")
        ax = flat[i]
        ranked = sorted(zip(names, values), key=lambda kv: kv[1], reverse=True)[:top_n]
        features = [name for name, _ in ranked][::-1]
        heights = [value for _, value in ranked][::-1]

        ax.barh(np.arange(len(features)), heights, color=SERIES_COLORS[i])
        ax.set_yticks(np.arange(len(features)))
        ax.set_yticklabels(features, fontsize=13)
        ax.set_xlabel("importance", fontsize=14)
        ax.set_title(label, fontsize=17)
        ax.tick_params(axis="x", labelsize=12)

    for ax in flat[len(series):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=17)
    fig.tight_layout()
    # tight_layout leaves a wide gap between rows once the panel titles are
    # enlarged; pull the rows together without touching the horizontal gap,
    # which the long feature labels need
    fig.subplots_adjust(hspace=0.45)
    plt.show()
    return fig
