import os
import sys
import argparse
import logging
import math
import numpy as np
import scipy.stats as st


# 1. SELECTED FEATURES (avg temp, max temp, power)
SELECTED_FEATURES = [1, 2, 7]   # column indexes to use


# 2. Candidate distributions to test
DISTRIBUTIONS = {
    "gaussian": st.norm,
    "gamma": st.gamma,
    "lognorm": st.lognorm,
    "weibull": st.weibull_min,
}


# 3. Fit distribution

def fit_distribution(data):
    best_dist = None
    best_aic = np.inf
    best_params = None

    for name, dist in DISTRIBUTIONS.items():
        try:
            params = dist.fit(data)
            loglik = np.sum(dist.logpdf(data, *params))
            k = len(params)
            aic = 2 * k - 2 * loglik

            if aic < best_aic:
                best_aic = aic
                best_dist = name
                best_params = params

        except Exception:
            continue

    return best_dist, best_params


# 4. Training (estimate distribution of NORMAL data)
def training(instances, labels):
    normal_instances = [x for x, y in zip(instances, labels) if y == 0]

    if len(normal_instances) == 0:
        logging.error("No normal data found (label=0).")
        sys.exit(1)

    selected_data = [[row[i] for i in SELECTED_FEATURES] for row in normal_instances]
    features_t = list(zip(*selected_data))

    feature_info = []

    for i, col in zip(SELECTED_FEATURES, features_t):
        data = np.array(col)

        # scaling
        mean = np.mean(data)
        std = np.std(data) + 1e-6
        scaled = (data - mean) / std

        best_dist, params = fit_distribution(scaled)

        feature_info.append({
            "feature_index": i,
            "mean": mean,
            "std": std,
            "dist": best_dist,
            "params": params
        })

        logging.info(f"Feature {i}: best distribution = {best_dist}")

    return {"feature_info": feature_info}


# 5. Likelihood calculation
def log_likelihood(instance, params):
    x_raw = [instance[i] for i in SELECTED_FEATURES]

    ll = 0.0
    for x, finfo in zip(x_raw, params["feature_info"]):
        scaled = (x - finfo["mean"]) / finfo["std"]
        dist = DISTRIBUTIONS[finfo["dist"]]
        p = dist.pdf(scaled, *finfo["params"])
        ll += math.log(p + 1e-12)

    return ll


# 6. Prediction
def predict(inst, params, threshold):
    ll = log_likelihood(inst, params)
    return 1 if ll < threshold else 0


# 7. Report
def report(preds, labels):
    correct = sum(p == y for p, y in zip(preds, labels))
    accuracy = correct / len(labels)

    tp = sum(p==1 and y==1 for p,y in zip(preds, labels))
    fp = sum(p==1 and y==0 for p,y in zip(preds, labels))
    fn = sum(p==0 and y==1 for p,y in zip(preds, labels))

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)

    logging.info("===== PERFORMANCE REPORT =====")
    logging.info(f"accuracy: {accuracy:.3f}")
    logging.info(f"precision: {precision:.3f}")
    logging.info(f"recall: {recall:.3f}")


# 8. Load data
def load_raw_data(fname):
    inst, lab = [], []
    with open(fname, "r") as f:
        f.readline()
        for line in f:
            tmp = line.strip().split(", ")
            row = tmp[:-1]
            row[1] = float(row[1])
            row[2] = float(row[2])
            row[3] = float(row[3])
            row[4] = float(row[4])
            row[5] = int(row[5])
            row[6] = int(row[6])
            row[7] = float(row[7])
            lab.append(int(tmp[-1]))
            inst.append(row)
    return inst, lab


# 9. Run
def run(train_file, test_file):
    train_x, train_y = load_raw_data(train_file)
    params = training(train_x, train_y)

    # compute threshold
    ll_list = [log_likelihood(x, params) for x, y in zip(train_x, train_y) if y == 0]
    ll_list.sort()
    threshold = ll_list[int(len(ll_list)*0.05)]   # bottom 5%

    logging.info(f"Threshold = {threshold}")

    # test
    test_x, test_y = load_raw_data(test_file)
    preds = [predict(x, params, threshold) for x in test_x]

    report(preds, test_y)


# 10. CLI
def command_line_args():
    p = argparse.ArgumentParser()
    p.add_argument("-t", "--training", required=True)
    p.add_argument("-u", "--testing", required=True)
    p.add_argument("-l", "--log", default="INFO")
    return p.parse_args()


def main():
    args = command_line_args()
    logging.basicConfig(level=args.log)
    run(args.training, args.testing)


if __name__ == "__main__":
    main()
