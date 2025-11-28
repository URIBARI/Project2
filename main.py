import os
import sys
import argparse
import logging
import math


# =========================================================
# 1. Gaussian PDF
# =========================================================
def gaussian_pdf(x, mean, var):
    if var == 0:
        var = 1e-6  # smoothing
    coeff = 1.0 / math.sqrt(2 * math.pi * var)
    exponent = math.exp(-((x - mean) ** 2) / (2 * var))
    return coeff * exponent


# =========================================================
# 2. Training: Learn NORMAL distribution only
# =========================================================
def training(instances, labels):
    """
    Outlier Detection Naïve Bayes
    - 정상(0) 데이터만 사용하여 feature 분포(mean, var) 학습
    - label=1 (outlier)은 학습에 사용하지 않음
    """
    normal_data = [inst for inst, y in zip(instances, labels) if y == 0]

    if len(normal_data) == 0:
        logging.error("No normal data (label 0) found. Cannot train outlier detector.")
        sys.exit(1)

    # feature-wise mean & variance
    features_t = list(zip(*[x[1:] for x in normal_data]))  # remove date

    means = []
    variances = []

    for col in features_t:
        mean = sum(col) / len(col)
        var = sum((v - mean)**2 for v in col) / len(col)
        if var == 0:
            var = 1e-6
        means.append(mean)
        variances.append(var)

    logging.info("Training completed using NORMAL-class data only.")
    logging.info("Learned means: {}".format(means))
    logging.info("Learned variances: {}".format(variances))

    return {
        "means": means,
        "variances": variances,
    }


# =========================================================
# 3. Prediction: compute log-likelihood → threshold
# =========================================================
def predict(instance, parameters):
    """
    Computes log-likelihood of NORMAL distribution.
    If log-likelihood < threshold → OUTLIER(1)
    """
    x = instance[1:]  # remove date column
    means = parameters["means"]
    variances = parameters["variances"]

    # calculate log-likelihood
    log_likelihood = 0.0
    for v, mean, var in zip(x, means, variances):
        p = gaussian_pdf(v, mean, var)
        log_likelihood += math.log(p + 1e-12)

    # threshold = mean(logL) - 2*std(logL)  (robust heuristic)
    threshold = parameters["threshold"]

    return 0 if log_likelihood >= threshold else 1


# =========================================================
# 4. Report metrics
# =========================================================
def report(predictions, answers):
    if len(predictions) != len(answers):
        logging.error("The lengths of two arguments should be same")
        sys.exit(1)

    # accuracy
    correct = 0
    for idx in range(len(predictions)):
        if predictions[idx] == answers[idx]:
            correct += 1
    accuracy = round(correct / len(answers), 2) * 100

    # precision
    tp = 0
    fp = 0
    for idx in range(len(predictions)):
        if predictions[idx] == 1:
            if answers[idx] == 1:
                tp += 1
            else:
                fp += 1
    precision = round(tp / (tp + fp + 1e-9), 2) * 100

    # recall
    tp = 0
    fn = 0
    for idx in range(len(answers)):
        if answers[idx] == 1:
            if predictions[idx] == 1:
                tp += 1
            else:
                fn += 1
    recall = round(tp / (tp + fn + 1e-9), 2) * 100

    logging.info("===== PERFORMANCE REPORT =====")
    logging.info("accuracy: {}%".format(accuracy))
    logging.info("precision: {}%".format(precision))
    logging.info("recall: {}%".format(recall))


# =========================================================
# 5. Load Data
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
# 6. FULL RUN
# =========================================================
def run(train_file, test_file):
    # training phase
    train_instances, train_labels = load_raw_data(train_file)
    parameters = training(train_instances, train_labels)

    # compute training log-likelihood for threshold setting
    log_ls = []
    for inst, y in zip(train_instances, train_labels):
        if y == 0:
            x = inst[1:]
            ll = 0
            for v, mean, var in zip(x, parameters["means"], parameters["variances"]):
                ll += math.log(gaussian_pdf(v, mean, var) + 1e-12)
            log_ls.append(ll)

    # automatic threshold selection (mean − 2 * std)
    mu = sum(log_ls) / len(log_ls)
    sigma = (sum((l - mu)**2 for l in log_ls) / len(log_ls))**0.5
    threshold = mu - 2 * sigma

    parameters["threshold"] = threshold
    logging.info(f"Threshold (auto): {threshold:.4f}")

    # testing phase
    test_instances, test_labels = load_raw_data(test_file)
    predictions = []
    for instance in test_instances:
        result = predict(instance, parameters)
        predictions.append(result)

    # report
    report(predictions, test_labels)


# =========================================================
# 7. Argument parser
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
        logging.error("Training dataset not found")
        sys.exit(1)

    if not os.path.exists(args.testing):
        logging.error("Testing dataset not found")
        sys.exit(1)

    run(args.training, args.testing)


if __name__ == "__main__":
    main()
