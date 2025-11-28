import os
import sys
import argparse
import logging
import math
import numpy as np
import scipy.stats as st


# =========================================================
# 0. Candidate Distributions
# =========================================================
DISTRIBUTIONS = {
    "gaussian": st.norm,
    "gamma": st.gamma,
    "lognorm": st.lognorm,
    "weibull": st.weibull_min,
}


# =========================================================
# 1. Fit distribution and choose best one (AIC)
# =========================================================
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


# =========================================================
# 2. PDF evaluation
# =========================================================
def pdf_value(dist_name, params, x):
    dist = DISTRIBUTIONS[dist_name]
    return dist.pdf(x, *params)


# =========================================================
# 3. Load data
# =========================================================
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


# =========================================================
# 4. Training
# =========================================================
def training(instances, labels):
    normal_data = [inst for inst, y in zip(instances, labels) if y == 0]

    if len(normal_data) == 0:
        logging.error("No normal (label=0) data in training set.")
        sys.exit(1)

    X = np.array([x[1:] for x in normal_data], dtype=float)

    # 1) Z-score scaling
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1e-6
    X_scaled = (X - means) / stds

    # 2) Automatic distribution selection for each feature
    dist_info = []
    for i in range(X_scaled.shape[1]):
        name, params = fit_distribution(X_scaled[:, i])
        dist_info.append((name, params))
        logging.info(f"Feature {i}: best distribution = {name}")

    logging.info("Training completed (auto distribution selection + scaling).")

    return {"scaler": (means, stds), "dists": dist_info}


# =========================================================
# 5. Predict (return log-likelihood)
# =========================================================
def predict(instance, parameters):
    means, stds = parameters["scaler"]
    x = np.array(instance[1:], dtype=float)

    x_scaled = (x - means) / stds

    log_likelihood = 0.0
    for val, (dist_name, params) in zip(x_scaled, parameters["dists"]):
        p = pdf_value(dist_name, params, val)
        log_likelihood += math.log(p + 1e-12)

    return log_likelihood


# =========================================================
# 6. Report
# =========================================================
def report(predictions, answers):
    correct = sum(p == a for p, a in zip(predictions, answers))
    accuracy = round(correct / len(answers), 2) * 100

    tp = sum(p == 1 and a == 1 for p, a in zip(predictions, answers))
    fp = sum(p == 1 and a == 0 for p, a in zip(predictions, answers))
    fn = sum(p == 0 and a == 1 for p, a in zip(predictions, answers))

    precision = round(tp / (tp + fp + 1e-9), 2) * 100
    recall = round(tp / (tp + fn + 1e-9), 2) * 100

    logging.info("===== PERFORMANCE REPORT =====")
    logging.info(f"accuracy: {accuracy}%")
    logging.info(f"precision: {precision}%")
    logging.info(f"recall: {recall}%")


# =========================================================
# 7. Run
# =========================================================
def run(train_file, test_file):
    inst_train, lab_train = load_raw_data(train_file)
    params = training(inst_train, lab_train)

    # ----------------------------------------------
    # Compute threshold using percentile (5%)
    # ----------------------------------------------
    logLs = []
    for inst, y in zip(inst_train, lab_train):
        if y == 0:
            ll = predict(inst, params)
            logLs.append(ll)

    threshold = np.percentile(logLs, 5)   # Outlier threshold (recommended)
    params["threshold"] = threshold

    logging.info(f"Auto threshold set to 5th percentile: {threshold}")

    # ----------------------------------------------
    # Testing
    # ----------------------------------------------
    inst_test, lab_test = load_raw_data(test_file)
    preds = []

    for inst in inst_test:
        ll = predict(inst, params)
        preds.append(0 if ll >= threshold else 1)

    report(preds, lab_test)


# =========================================================
# 8. Argparse
# =========================================================
def command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training", required=True)
    parser.add_argument("-u", "--testing", required=True)
    parser.add_argument("-l", "--log", default="INFO")
    return parser.parse_args()


def main():
    args = command_line_args()
    logging.basicConfig(level=args.log)

    if not os.path.exists(args.training):
        logging.error("Training file not found")
        sys.exit(1)

    if not os.path.exists(args.testing):
        logging.error("Testing file not found")
        sys.exit(1)

    run(args.training, args.testing)


if __name__ == "__main__":
    main()
