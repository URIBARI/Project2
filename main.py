import os
import sys
import argparse
import logging
import math
import numpy as np
import scipy.stats as st


# 0. Candidate Distributions
DISTRIBUTIONS = {
    "gaussian": st.norm,
    "gamma": st.gamma,
    "lognorm": st.lognorm,
    "weibull": st.weibull_min,
}


# 1. Fit distribution and choose best one (AIC)
def fit_distribution(data):
    best_aic = float("inf")
    best_name = None
    best_params = None

    for name, dist in DISTRIBUTIONS.items():
        try:
            params = dist.fit(data)
            loglik = np.sum(dist.logpdf(data, *params))
            k = len(params)
            aic = 2 * k - 2 * loglik
            if aic < best_aic:
                best_aic = aic
                best_name = name
                best_params = params
        except Exception:
            continue

    return best_name, best_params


# 2. PDF evaluation
def pdf_value(dist_name, params, x):
    dist = DISTRIBUTIONS[dist_name]
    return dist.pdf(x, *params)


# 3. Load data
def load_raw_data(fname):
    instances = []
    labels = []
    with open(fname, "r") as f:
        f.readline()
        for line in f:
            tmp = line.strip().split(", ")
            tmp[1] = float(tmp[1])
            tmp[2] = float(tmp[2])
            tmp[3] = float(tmp[3])
            tmp[4] = float(tmp[4])
            tmp[5] = int(tmp[5])
            tmp[6] = int(tmp[6])
            tmp[7] = float(tmp[7])
            tmp[8] = int(tmp[8])
            instances.append(tmp[:-1])
            labels.append(tmp[-1])
    return instances, labels


# Helper: apply feature subset
def apply_feature_subset(instances, selected):
    new_data = []
    for inst in instances:
        row = [inst[0]]  # keep date
        for idx in selected:
            row.append(inst[idx + 1])  # +1 offset
        new_data.append(row)
    return new_data


# 4. Training
def training(instances, labels):
    normal_data = [inst for inst, y in zip(instances, labels) if y == 0]

    if len(normal_data) == 0:
        logging.error("No normal (label=0) data in training set.")
        sys.exit(1)

    X = np.array([x[1:] for x in normal_data], dtype=float)

    # Z-score scaling
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1e-6
    X_scaled = (X - means) / stds

    # Automatic distribution selection
    dist_info = []
    for i in range(X_scaled.shape[1]):
        name, params = fit_distribution(X_scaled[:, i])
        dist_info.append((name, params))
        logging.info(f"Feature {i}: best distribution = {name}")

    return {"scaler": (means, stds), "dists": dist_info}


# 5. Predict (return log-likelihood)
def predict(instance, parameters):
    means, stds = parameters["scaler"]
    x = np.array(instance[1:], dtype=float)
    x_scaled = (x - means) / stds

    log_likelihood = 0.0
    for val, (dist_name, params) in zip(x_scaled, parameters["dists"]):
        p = pdf_value(dist_name, params, val)
        log_likelihood += math.log(p + 1e-12)

    return log_likelihood


# 6. Report
def report(predictions, answers):
    correct = sum(p == a for p, a in zip(predictions, answers))
    accuracy = round(correct / len(answers), 3)

    tp = sum(p == 1 and a == 1 for p, a in zip(predictions, answers))
    fp = sum(p == 1 and a == 0 for p, a in zip(predictions, answers))
    fn = sum(p == 0 and a == 1 for p, a in zip(predictions, answers))

    precision = round(tp / (tp + fp + 1e-9), 3)
    recall = round(tp / (tp + fn + 1e-9), 3)

    logging.info("===== PERFORMANCE REPORT =====")
    logging.info(f"accuracy: {accuracy}")
    logging.info(f"precision: {precision}")
    logging.info(f"recall: {recall}")

    return accuracy, precision, recall


# 7. Full run on selected feature set
def run_with_features(train_file, test_file, selected_features):
    train_instances, train_labels = load_raw_data(train_file)
    test_instances, test_labels = load_raw_data(test_file)

    train_mod = apply_feature_subset(train_instances, selected_features)
    test_mod = apply_feature_subset(test_instances, selected_features)

    params = training(train_mod, train_labels)

    # threshold using percentile (5%)
    logLs = [predict(inst, params) for inst, y in zip(train_mod, train_labels) if y == 0]
    threshold = np.percentile(logLs, 5)
    params["threshold"] = threshold
    logging.info(f"Threshold = {threshold}")

    preds = []
    for inst in test_mod:
        ll = predict(inst, params)
        preds.append(0 if ll >= threshold else 1)

    return report(preds, test_labels)


# 8. FEATURE IMPORTANCE ANALYSIS
def evaluate_feature_importance(train_file, test_file):
    logging.info("===== FEATURE IMPORTANCE EVALUATION =====")

    base_acc, base_prec, base_rec = run_with_features(
        train_file, test_file, list(range(7))
    )

    results = []

    for remove_idx in range(7):
        remaining = [i for i in range(7) if i != remove_idx]
        logging.info(f"\n--- Testing WITHOUT feature {remove_idx} ---")
        acc, prec, rec = run_with_features(train_file, test_file, remaining)
        results.append((remove_idx, acc, prec, rec, remaining))

    return base_prec, base_rec, results


# 9. AUTO FEATURE SELECTION
def auto_select_features(train_file, test_file):
    base_prec, base_rec, results = evaluate_feature_importance(train_file, test_file)

    selected = []
    removed = []

    for idx, acc, prec, rec, remaining in results:
        if prec > base_prec or rec > base_rec:
            removed.append(idx)
        else:
            selected.append(idx)

    logging.info("===== AUTO FEATURE SELECTION RESULT =====")
    logging.info(f"Selected Features: {selected}")
    logging.info(f"Removed Noise Features: {removed}")

    return selected


# 10. MAIN
def command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training", required=True)
    parser.add_argument("-u", "--testing", required=True)
    parser.add_argument("--auto", action="store_true",
                        help="Run fully automated feature selection + training")
    parser.add_argument("-l", "--log", default="INFO")
    return parser.parse_args()


def main():
    args = command_line_args()
    logging.basicConfig(level=args.log)

    train_file = args.training
    test_file = args.testing

    if args.auto:
        selected = auto_select_features(train_file, test_file)
        logging.info("===== FINAL TRAINING WITH SELECTED FEATURES =====")
        run_with_features(train_file, test_file, selected)
    else:
        logging.info("===== RUNNING WITH ALL FEATURES =====")
        run_with_features(train_file, test_file, list(range(7)))


if __name__ == "__main__":
    main()
