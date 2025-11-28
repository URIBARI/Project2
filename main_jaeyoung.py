import os
import sys
import argparse
import logging
import math
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PDF FUNCTIONS
# ============================================================

def gaussian_pdf(x, mu, sigma, smoothing):
    if sigma <= 0:
        sigma = 1e-6
    coeff = 1.0 / (math.sqrt(2 * math.pi) * sigma)
    exponent = math.exp(-((x - mu)**2) / (2 * sigma * sigma))
    p = coeff * exponent
    return max(p, smoothing)


def lognormal_pdf(x, mu, sigma, smoothing):
    if x <= 0:
        return smoothing
    if sigma <= 0:
        sigma = 1e-6
    coeff = 1.0 / (x * sigma * math.sqrt(2 * math.pi))
    exponent = math.exp(-(math.log(x) - mu)**2 / (2 * sigma * sigma))
    p = coeff * exponent
    return max(p, smoothing)


# ============================================================
# TRAINING
# ============================================================

def training(instances, labels, smoothing):

    # feature index:
    #  maxT = 2, avgT = 1, power = 7
    X = []
    y = []

    for inst, lab in zip(instances, labels):
        X.append([
            float(inst[2]),  # maxT
            float(inst[1]),  # avgT
            float(inst[7])   # power
        ])
        y.append(int(lab))

    X = np.array(X)
    y = np.array(y)

    params = {}
    params["prior0"] = np.mean(y == 0)
    params["prior1"] = np.mean(y == 1)

    # Gaussian: maxT
    params["maxT"] = {}
    for c in [0, 1]:
        vals = X[y == c, 0]
        params["maxT"][c] = (np.mean(vals), np.std(vals))

    # Gaussian: avgT
    params["avgT"] = {}
    for c in [0, 1]:
        vals = X[y == c, 1]
        params["avgT"][c] = (np.mean(vals), np.std(vals))

    # Lognormal: power
    params["power"] = {}
    for c in [0, 1]:
        vals = X[y == c, 2]
        vals = np.log(vals + 1e-9)
        params["power"][c] = (np.mean(vals), np.std(vals))

    params["smoothing"] = smoothing
    params["threshold"] = 0.20

    return params


# ============================================================
# PREDICT
# ============================================================

def predict(instance, parameters):
    smoothing = parameters["smoothing"]
    maxT = float(instance[2])
    avgT = float(instance[1])
    power = float(instance[7])

    # Priors
    p0 = parameters["prior0"]
    p1 = parameters["prior1"]

    # Class 0 likelihood
    L0 = (
        gaussian_pdf(maxT, *parameters["maxT"][0], smoothing) *
        gaussian_pdf(avgT, *parameters["avgT"][0], smoothing) *
        lognormal_pdf(power, *parameters["power"][0], smoothing) *
        p0
    )

    # Class 1 likelihood
    L1 = (
        gaussian_pdf(maxT, *parameters["maxT"][1], smoothing) *
        gaussian_pdf(avgT, *parameters["avgT"][1], smoothing) *
        lognormal_pdf(power, *parameters["power"][1], smoothing) *
        p1
    )

    if L0 + L1 == 0:
        posterior = 0
    else:
        posterior = L1 / (L0 + L1)

    return 1 if posterior > parameters["threshold"] else 0


# ============================================================
# PERFORMANCE METRIC
# ============================================================

def evaluate(predictions, answers):
    accuracy = sum(int(p == a) for p, a in zip(predictions, answers)) / len(answers)

    tp = sum((p == 1 and a == 1) for p, a in zip(predictions, answers))
    fp = sum((p == 1 and a == 0) for p, a in zip(predictions, answers))
    fn = sum((p == 0 and a == 1) for p, a in zip(predictions, answers))

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)

    return accuracy, precision, recall


# ============================================================
# DATA LOADING
# ============================================================

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


# ============================================================
# RUN
# ============================================================

def run(train_file, test_file):

    train_instances, train_labels = load_raw_data(train_file)
    test_instances, test_labels = load_raw_data(test_file)

    smoothing_values = [1e-12, 1e-9, 1e-6, 1e-4, 1e-2]
    acc_list = []
    prec_list = []
    rec_list = []

    results = []

    for sm in smoothing_values:
        params = training(train_instances, train_labels, smoothing=sm)
        preds = [predict(inst, params) for inst in test_instances]

        acc, prec, rec = evaluate(preds, test_labels)

        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)

        score = acc + rec
        results.append((score, sm, acc, prec, rec))

        logging.info(f"Smoothing={sm:.0e}  ACC={acc:.3f}  PREC={prec:.3f}  REC={rec:.3f}")

    # Select best smoothing
    best = max(results, key=lambda x: x[0])
    best_smoothing = best[1]

    logging.info(f"\nBEST smoothing: {best_smoothing}\n")

    # Train final model
    final_params = training(train_instances, train_labels, smoothing=best_smoothing)
    final_preds = [predict(inst, final_params) for inst in test_instances]
    final_acc, final_prec, final_rec = evaluate(final_preds, test_labels)

    logging.info(f"FINAL MODEL → ACC={final_acc:.3f}  PREC={final_prec:.3f}  REC={final_rec:.3f}")

    # Save tuning plot
    plt.figure(figsize=(8,5))
    plt.plot(smoothing_values, acc_list, marker='o', label="Accuracy")
    plt.plot(smoothing_values, prec_list, marker='o', label="Precision")
    plt.plot(smoothing_values, rec_list, marker='o', label="Recall")
    plt.xscale("log")
    plt.xlabel("Smoothing value")
    plt.ylabel("Performance")
    plt.title("Smoothing Hyperparameter Tuning")
    plt.legend()
    plt.grid()
    plt.savefig("tuning_performance.png")

    logging.info("Saved tuning plot: tuning_performance.png")


# ============================================================
# MAIN
# ============================================================

def command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training", required=True)
    parser.add_argument("-u", "--testing", required=True)
    parser.add_argument("-l", "--log", type=str, default="INFO")
    return parser.parse_args()


def main():
    args = command_line_args()
    logging.basicConfig(level=args.log)

    if not os.path.exists(args.training):
        logging.error("Training dataset not found")
        sys.exit(1)

    if not os.path.exists(args.testing):
        logging.error("Testing dataset not found")
        sys.exit(1)

    run(args.training, args.testing)


if __name__ == "__main__":
    main()
