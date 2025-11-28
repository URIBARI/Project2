import os
import sys
import argparse
import logging
import math

# --------------------------------------------------------
# 설정: 어떤 feature를 어떤 분포로 볼 것인지
#   index 기준: 
#   0: date (문자열)
#   1: avg temperature       -> Gaussian
#   2: max temperature       -> Gaussian
#   7: power                 -> Lognormal
# --------------------------------------------------------
FEATURE_INDICES = [1, 2, 7]
FEATURE_DISTS   = ["gaussian", "gaussian", "lognormal"]


# --------------------------------------------------------
# 분포별 PDF 정의
# --------------------------------------------------------
def gaussian_pdf(x, mean, var):
    """Gaussian(정규분포) 확률밀도함수"""
    var = max(var, 1e-9)
    exponent = math.exp(-((x - mean) ** 2) / (2 * var))
    return (1.0 / math.sqrt(2 * math.pi * var)) * exponent


def lognormal_pdf(x, mean_log, var_log):
    """Lognormal 분포 확률밀도함수 (mean/var는 log(x) 기준)"""
    if x <= 0:
        # 로그정규분포는 0 이하에서 정의 안 되므로 아주 작은 확률 부여
        return 1e-12
    var_log = max(var_log, 1e-9)
    sigma = math.sqrt(var_log)
    return (1.0 / (x * sigma * math.sqrt(2 * math.pi))) * \
           math.exp(-((math.log(x) - mean_log) ** 2) / (2 * var_log))


# --------------------------------------------------------
# 1) TRAINING: Naive Bayes 학습
# --------------------------------------------------------
def training(instances, labels, eps: float = 1e-6):
    """
    Gaussian + Lognormal Naive Bayes 학습 함수.

    반환되는 parameters 구조:
    {
        "feature_indices": [...],
        "feature_dists": [...],
        "classes": [0,1],
        "class_params": {
            c: {
                "prior": float,
                "means": [f1_mean, f2_mean, f3_mean],
                "vars":  [f1_var,  f2_var,  f3_var ],
            },
            ...
        }
    }
    """
    classes = sorted(set(labels))
    total = len(labels)

    params = {
        "feature_indices": FEATURE_INDICES,
        "feature_dists": FEATURE_DISTS,
        "classes": classes,
        "class_params": {}
    }

    for c in classes:
        idxs_c = [i for i, lab in enumerate(labels) if lab == c]
        class_size = len(idxs_c)
        prior = class_size / total

        feature_means = []
        feature_vars = []

        for fi, dist in zip(FEATURE_INDICES, FEATURE_DISTS):
            # 해당 클래스에 속한 값들만 모으기
            vals = [float(instances[i][fi]) for i in idxs_c]

            # Lognormal 분포일 경우 log 변환 후 평균/분산 계산
            if dist == "lognormal":
                vals = [math.log(v) for v in vals if v > 0]

            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            if var <= 0:
                var = eps

            feature_means.append(mean)
            feature_vars.append(var)

        params["class_params"][c] = {
            "prior": prior,
            "means": feature_means,
            "vars": feature_vars
        }

    logging.info("Training finished. Classes: %s", classes)
    return params


# --------------------------------------------------------
# 2) PREDICT: 한 개 instance에 대해 클래스 예측
# --------------------------------------------------------
def predict(instance, parameters):
    """
    한 개 데이터(instance)에 대해 posterior를 계산하여
    0 또는 1 중 더 높은 쪽 클래스를 반환.
    """
    best_class = None
    best_logp = None

    F_IDX = parameters["feature_indices"]
    DISTS = parameters["feature_dists"]

    for c in parameters["classes"]:
        cp = parameters["class_params"][c]
        logp = math.log(cp["prior"])

        for j, (idx, dist) in enumerate(zip(F_IDX, DISTS)):
            x = float(instance[idx])
            mean = cp["means"][j]
            var = cp["vars"][j]

            if dist == "gaussian":
                pdf = gaussian_pdf(x, mean, var)
            else:  # "lognormal"
                pdf = lognormal_pdf(x, mean, var)

            if pdf <= 0:
                pdf = 1e-12
            logp += math.log(pdf)

        if (best_logp is None) or (logp > best_logp):
            best_logp = logp
            best_class = c

    return best_class


# --------------------------------------------------------
# 3) REPORT / DATA LOADING / MAIN (기존 코드)
# --------------------------------------------------------
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
    precision = round(tp / (tp + fp), 2) * 100 if (tp + fp) > 0 else 0.0

    # recall
    tp = 0
    fn = 0
    for idx in range(len(answers)):
        if answers[idx] == 1:
            if predictions[idx] == 1:
                tp += 1
            else:
                fn += 1
    recall = round(tp / (tp + fn), 2) * 100 if (tp + fn) > 0 else 0.0

    logging.info("accuracy: {}%".format(accuracy))
    logging.info("precision: {}%".format(precision))
    logging.info("recall: {}%".format(recall))


def load_raw_data(fname):
    instances = []
    labels = []
    with open(fname, "r") as f:
        f.readline()  # header skip
        for line in f:
            tmp = line.strip().split(", ")
            tmp[1] = float(tmp[1])  # avg temp
            tmp[2] = float(tmp[2])  # max temp
            tmp[3] = float(tmp[3])  # min temp
            tmp[4] = float(tmp[4])  # avg humidity
            tmp[5] = int(tmp[5])    # max humidity
            tmp[6] = int(tmp[6])    # min humidity
            tmp[7] = float(tmp[7])  # power
            tmp[8] = int(tmp[8])    # label
            instances.append(tmp[:-1])  # 마지막(label) 제외
            labels.append(tmp[-1])      # label만 저장
    return instances, labels


def run(train_file, test_file):
    # training phase
    instances, labels = load_raw_data(train_file)
    logging.debug("instances: {}".format(instances[:3]))
    logging.debug("labels: {}".format(labels[:10]))
    parameters = training(instances, labels)

    # testing phase
    instances, labels = load_raw_data(test_file)
    predictions = []
    for instance in instances:
        result = predict(instance, parameters)

        if result not in [0, 1]:
            logging.error("The result must be either 0 or 1")
            sys.exit(1)

        predictions.append(result)

    # report
    report(predictions, labels)


def command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--training",
        required=True,
        metavar="<file path to the training dataset>",
        help="File path of the training dataset",
        default="training.csv",
    )
    parser.add_argument(
        "-u", "--testing",
        required=True,
        metavar="<file path to the testing dataset>",
        help="File path of the testing dataset",
        default="testing.csv",
    )
    parser.add_argument(
        "-l", "--log",
        help="Log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)",
        type=str,
        default="INFO",
    )

    args = parser.parse_args()
    return args


def main():
    args = command_line_args()
    logging.basicConfig(level=args.log)

    if not os.path.exists(args.training):
        logging.error("The training dataset does not exist: {}".format(args.training))
        sys.exit(1)

    if not os.path.exists(args.testing):
        logging.error("The testing dataset does not exist: {}".format(args.testing))
        sys.exit(1)

    run(args.training, args.testing)


if __name__ == "__main__":
    main()