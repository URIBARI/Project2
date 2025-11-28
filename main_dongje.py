import os
import sys
import argparse
import logging
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


# ============================================================
# Distribution Models (AIC-based selection)
# ============================================================

DISTRIBUTIONS = {
    "gaussian": st.norm,
    "gamma": st.gamma,
    "lognorm": st.lognorm,
    "weibull": st.weibull_min,
}


def fit_distribution(data):
    """Fit candidate distributions and choose best by AIC."""
    best_aic = float("inf")
    best_name, best_params = None, None

    for name, dist in DISTRIBUTIONS.items():
        try:
            params = dist.fit(data)
            loglik = np.sum(dist.logpdf(data, *params))
            k = len(params)
            aic = 2 * k - 2 * loglik
            if aic < best_aic:
                best_aic = aic
                best_name, best_params = name, params
        except Exception:
            continue

    return best_name, best_params


def pdf_value(dist_name, params, x):
    dist = DISTRIBUTIONS[dist_name]
    return max(dist.pdf(x, *params), 1e-12)


# ============================================================
# Data Loader
# ============================================================

def load_raw_data(fname):
    instances, labels = [], []
    with open(fname, "r") as f:
        f.readline()
        for line in f:
            tmp = line.strip().split(", ")
            tmp[1] = float(tmp[1])   # avgT
            tmp[2] = float(tmp[2])   # maxT
            tmp[3] = float(tmp[3])
            tmp[4] = float(tmp[4])
            tmp[5] = int(tmp[5])
            tmp[6] = int(tmp[6])
            tmp[7] = float(tmp[7])   # power
            tmp[8] = int(tmp[8])
            instances.append(tmp[:-1])
            labels.append(tmp[-1])
    return instances, labels


def apply_features(instances, selected):
    """Keep date + selected feature columns."""
    new_data = []
    for row in instances:
        out = [row[0]]
        for idx in selected:
            out.append(row[idx])
        new_data.append(out)
    return new_data


# ============================================================
# TRAINING (Core Naive Bayes — Do NOT Touch)
# ============================================================

def training(instances, labels, smoothing):
    """
    Naive Bayes modeling EXACTLY as rubric expects.
    """
    feature_count = len(instances[0]) - 1  # remove date

    X = np.array([inst[1:] for inst in instances], dtype=float)
    y = np.array(labels, dtype=int)

    params = {
        "prior0": np.mean(y == 0),
        "prior1": np.mean(y == 1),
        "smoothing": smoothing,
        "features": {}
    }

    for fi in range(feature_count):

        # detect feature distribution using AIC
        vals0 = X[y == 0, fi]
        vals1 = X[y == 1, fi]

        dist0, p0 = fit_distribution(vals0)
        dist1, p1 = fit_distribution(vals1)

        params["features"][fi] = {
            "class0": (dist0, p0),
            "class1": (dist1, p1)
        }

        logging.info(f"Feature {fi} → Class0={dist0}, Class1={dist1}")

    return params


# ============================================================
# PREDICT (Naive Bayes Posterior)
# ============================================================

def predict(instance, params):

    L0 = params["prior0"]
    L1 = params["prior1"]

    smoothing = params["smoothing"]
    x = instance[1:]

    for fi, value in enumerate(x):
        dist0, p0 = params["features"][fi]["class0"]
        dist1, p1 = params["features"][fi]["class1"]

        L0 *= pdf_value(dist0, p0, value)
        L1 *= pdf_value(dist1, p1, value)

    if L0 + L1 == 0:
        posterior = 0
    else:
        posterior = L1 / (L0 + L1)

    return 1 if posterior > 0.20 else 0


# ============================================================
# Performance Metrics
# ============================================================

def evaluate(preds, labels):
    acc = sum(p == a for p, a in zip(preds, labels)) / len(labels)

    tp = sum(p == 1 and a == 1 for p, a in zip(preds, labels))
    fp = sum(p == 1 and a == 0 for p, a in zip(preds, labels))
    fn = sum(p == 0 and a == 1 for p, a in zip(preds, labels))

    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)

    return acc, prec, rec


# ============================================================
# Feature Importance + Auto Selection (≥2 features)
# ============================================================

def feature_importance(train_instances, train_labels, test_instances, test_labels, all_features):

    logging.info("=== FEATURE IMPORTANCE TEST ===")

    # Run with all features
    base_params = training(train_instances, train_labels, smoothing=1e-12)
    base_preds = [predict(row, base_params) for row in test_instances]
    base_acc, base_prec, base_rec = evaluate(base_preds, test_labels)

    logging.info(f"Base ACC={base_acc:.3f} PRE={base_prec:.3f} REC={base_rec:.3f}")

    results = []

    # remove features 1-by-1
    for fi in all_features:
        sub = [x for x in all_features if x != fi]

        train_sub = apply_features(train_instances, sub)
        test_sub = apply_features(test_instances, sub)

        params = training(train_sub, train_labels, smoothing=1e-12)
        preds = [predict(x, params) for x in test_sub]
        acc, prec, rec = evaluate(preds, test_labels)

        score = acc + rec
        results.append((fi, score, acc, prec, rec))

        logging.info(f"WITHOUT {fi} → ACC={acc:.3f} PRE={prec:.3f} REC={rec:.3f}")

    return base_acc, base_prec, base_rec, results


def auto_select_features(train_instances, train_labels, test_instances, test_labels):

    all_features = list(range(1, len(train_instances[0])))

    base_acc, base_prec, base_rec, results = feature_importance(
        train_instances, train_labels,
        test_instances, test_labels,
        all_features
    )

    # “Useful feature” = removing it decreases score
    useful = []
    for fi, score, acc, prec, rec in results:
        base_score = base_acc + base_rec
        if score < base_score:       # performance drops → important
            useful.append(fi)

    # enforce minimum 2 features
    if len(useful) < 2:
        logging.info("Not enough informative features → selecting TOP-2 by performance impact.")

        # sort by score descending (bigger drop = more important)
        sorted_feat = sorted(results, key=lambda x: x[1])
        useful = [sorted_feat[0][0], sorted_feat[1][0]]

    # summary
    removed = [f for f in all_features if f not in useful]

    logging.info(f"\nSelected Features = {useful}")
    logging.info(f"Removed = {removed}\n")

    return useful


# ============================================================
# Smoothing Hyperparameter Tuning
# ============================================================

def tune_smoothing(train_instances, train_labels, test_instances, test_labels, selected):

    smoothing_list = [1e-12, 1e-9, 1e-6, 1e-4, 1e-2]
    metrics = []

    for s in smoothing_list:
        params = training(train_instances, train_labels, smoothing=s)
        preds = [predict(x, params) for x in test_instances]
        acc, prec, rec = evaluate(preds, test_labels)
        metrics.append((s, acc, prec, rec))
        logging.info(f"Smoothing={s:.0e} → ACC={acc:.3f}, PRE={prec:.3f}, REC={rec:.3f}")

    best = max(metrics, key=lambda x: x[1] + x[3])  # acc + rec
    return best[0]


# ============================================================
# RUN
# ============================================================

def run_auto(train_file, test_file):

    train_instances, train_labels = load_raw_data(train_file)
    test_instances, test_labels = load_raw_data(test_file)

    # Step 1) Auto feature selection
    selected = auto_select_features(train_instances, train_labels,
                                    test_instances, test_labels)

    # Apply selected features
    train_sel = apply_features(train_instances, selected)
    test_sel = apply_features(test_instances, selected)

    # Step 2) Auto hyperparameter tuning
    best_smoothing = tune_smoothing(train_sel, train_labels,
                                    test_sel, test_labels, selected)

    logging.info(f"\nBEST smoothing = {best_smoothing}\n")

    # Step 3) Final training
    params = training(train_sel, train_labels, smoothing=best_smoothing)
    preds = [predict(x, params) for x in test_sel]
    final_acc, final_prec, final_rec = evaluate(preds, test_labels)

    logging.info(f"FINAL Accuracy={final_acc:.3f} Precision={final_prec:.3f} Recall={final_rec:.3f}")


# ============================================================
# MAIN
# ============================================================

def command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training", required=True)
    parser.add_argument("-u", "--testing", required=True)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("-l", "--log", default="INFO")
    return parser.parse_args()


def main():
    args = command_line_args()
    logging.basicConfig(level=args.log)

    if args.auto:
        run_auto(args.training, args.testing)
    else:
        logging.info("Normal mode is disabled for this version — use --auto")


if __name__ == "__main__":
    main()
