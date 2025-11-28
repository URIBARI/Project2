import os
import sys
import argparse
import logging
import math
import matplotlib.pyplot as plt

# ---------------------------
# 1. 자동 Feature Selection
# ---------------------------

def automatic_feature_selection(instances, threshold=5.0):
    """
    Variance threshold 기반 자동 feature selection
    date column 제외, feature 1~7 사용
    """
    features = [x[1:] for x in instances]  # remove date
    features_t = list(zip(*features))

    selected_idx = []
    for i, col in enumerate(features_t):
        mean = sum(col) / len(col)
        var = sum((v - mean)**2 for v in col) / len(col)

        # Variance threshold 자동 적용
        if var >= threshold:
            selected_idx.append(i)

    logging.info(f"[Auto-Feature-Selection] selected features idx: {selected_idx}")
    return selected_idx


# ---------------------------
# 2. Gaussian Naive Bayes Training
# ---------------------------

def training(instances, labels, feature_idx, smoothing):
    parameters = {
        "priors": {},
        "mean": {},
        "var": {},
        "features": feature_idx,
        "smoothing": smoothing
    }

    class_data = {0: [], 1: []}

    for x, y in zip(instances, labels):
        class_data[y].append([x[i+1] for i in feature_idx])  # +1 to skip date

    total = len(labels)
    parameters["priors"][0] = len(class_data[0]) / total
    parameters["priors"][1] = len(class_data[1]) / total

    for c in [0,1]:
        features = list(zip(*class_data[c]))
        means = []
        variances = []

        for fvals in features:
            mean = sum(fvals) / len(fvals)
            var = sum((v - mean) ** 2 for v in fvals) / len(fvals)
            if var == 0:
                var = smoothing   # smoothing 적용

            means.append(mean)
            variances.append(var)

        parameters["mean"][c] = means
        parameters["var"][c] = variances

    return parameters


# ---------------------------
# 3. Gaussian PDF
# ---------------------------

def gaussian_prob(x, mean, var):
    coeff = 1.0 / math.sqrt(2 * math.pi * var)
    exponent = math.exp(-((x - mean) ** 2) / (2 * var))
    return coeff * exponent


# ---------------------------
# 4. Prediction
# ---------------------------

def predict(instance, parameters):
    feature_idx = parameters["features"]
    smoothing = parameters["smoothing"]

    x = [instance[i+1] for i in feature_idx]

    priors = parameters["priors"]
    means = parameters["mean"]
    vars_ = parameters["var"]

    posteriors = {}

    for c in [0,1]:
        log_prob = math.log(priors[c] + 1e-9)

        for i in range(len(x)):
            p = gaussian_prob(x[i], means[c][i], vars_[c][i])
            log_prob += math.log(p + 1e-9)

        posteriors[c] = log_prob

    return 0 if posteriors[0] > posteriors[1] else 1


# ---------------------------
# 5. Metrics
# ---------------------------

def compute_metrics(predictions, answers):
    correct = sum(p==a for p,a in zip(predictions, answers))
    accuracy = correct / len(answers)

    tp = sum(p==1 and a==1 for p,a in zip(predictions, answers))
    fp = sum(p==1 and a==0 for p,a in zip(predictions, answers))
    fn = sum(p==0 and a==1 for p,a in zip(predictions, answers))

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)

    return accuracy, precision, recall


# ---------------------------
# 6. Data Loading
# ---------------------------

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


# ---------------------------
# 7. Hyperparameter Tuning + Training + Report
# ---------------------------

def run(train_file, test_file):
    train_instances, train_labels = load_raw_data(train_file)
    test_instances, test_labels = load_raw_data(test_file)

    # -------------------------------
    # Automatic Feature Selection
    # -------------------------------
    selected_features = automatic_feature_selection(train_instances, threshold=5.0)

    # -------------------------------
    # Hyperparameter tuning: smoothing
    # -------------------------------
    smoothing_values = [1e-9, 1e-6, 1e-3]
    acc_list, prec_list, rec_list = [], [], []

    for sm in smoothing_values:
        params = training(train_instances, train_labels, selected_features, sm)

        predictions = [predict(inst, params) for inst in test_instances]

        acc, prec, rec = compute_metrics(predictions, test_labels)
        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)

        logging.info(f"[Smoothing={sm}] Accuracy={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}")

    # -------------------------------
    # Save tuning graph
    # -------------------------------
    plt.figure(figsize=(8,5))
    plt.plot(smoothing_values, acc_list, marker='o', label='Accuracy')
    plt.plot(smoothing_values, prec_list, marker='o', label='Precision')
    plt.plot(smoothing_values, rec_list, marker='o', label='Recall')
    plt.xscale('log')
    plt.xlabel("Variance Smoothing")
    plt.ylabel("Score")
    plt.title("Hyperparameter Tuning Results")
    plt.legend()
    plt.grid(True)
    plt.savefig("tuning_results.png")
    logging.info("Saved tuning graph: tuning_results.png")


# ---------------------------
# 8. Main CLI
# ---------------------------

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
