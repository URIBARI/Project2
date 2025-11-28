import os
import sys
import argparse
import logging
import math
import numpy as np
import scipy.stats as st


# ============================================================
# 0. Candidate Distributions for Automatic Model Selection
# ============================================================

DISTRIBUTIONS = {
    "gaussian": st.norm,
    "lognorm": st.lognorm,
    "gamma": st.gamma,
    "weibull": st.weibull_min,
}


def fit_distribution(data):
    """AIC 기반 분포 자동 선택"""
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


def pdf_eval(dist_name, params, x, smoothing):
    """선택된 분포로 PDF 계산"""
    try:
        dist = DISTRIBUTIONS[dist_name]
        p = dist.pdf(x, *params)
        return max(p, smoothing)
    except Exception:
        return smoothing


# ============================================================
# 1. Training
# ============================================================

def training(instances, labels, smoothing, selected_features, auto_model=False):
    """
    instances: raw rows (date + 7 features)
    selected_features: feature index list (1~7 중 일부)
    auto_model=True: 각 feature에 대해 AIC로 best distribution 선택
    auto_model=False: 모든 feature Gaussian 가정
    """

    X = []
    y = []

    for inst, lab in zip(instances, labels):
        row = [float(inst[f]) for f in selected_features]
        X.append(row)
        y.append(int(lab))

    X = np.array(X)
    y = np.array(y)

    params = {}
    params["prior0"] = np.mean(y == 0)
    params["prior1"] = np.mean(y == 1)
    params["smoothing"] = smoothing

    dist_info_0 = []
    dist_info_1 = []

    for f_idx in range(X.shape[1]):
        vals_all = X[:, f_idx]

        for c in [0, 1]:
            vals = X[y == c, f_idx]

            if auto_model:
                name, dp = fit_distribution(vals)
            else:
                # 기본 Gaussian
                name = "gaussian"
                mu = np.mean(vals)
                sigma = np.std(vals)
                if sigma <= 0:
                    sigma = 1e-6
                dp = (mu, sigma)

            if c == 0:
                dist_info_0.append((name, dp))
            else:
                dist_info_1.append((name, dp))

        logging.info(
            f"Feature {selected_features[f_idx]} → Class0={dist_info_0[-1][0]}, Class1={dist_info_1[-1][0]}"
        )

    params["dist0"] = dist_info_0
    params["dist1"] = dist_info_1

    return params


# ============================================================
# 2. Prediction
# ============================================================

def predict(instance, parameters, selected_features):
    smoothing = parameters["smoothing"]
    x = np.array([float(instance[f]) for f in selected_features])

    L0 = parameters["prior0"]
    L1 = parameters["prior1"]

    for i, val in enumerate(x):
        name0, p0_params = parameters["dist0"][i]
        name1, p1_params = parameters["dist1"][i]

        L0 *= pdf_eval(name0, p0_params, val, smoothing)
        L1 *= pdf_eval(name1, p1_params, val, smoothing)

    posterior = L1 / (L0 + L1 + 1e-9)
    return 1 if posterior > 0.20 else 0


# ============================================================
# 3. Evaluation Metrics
# ============================================================

def evaluate(pred, ans):
    acc = sum(int(p == a) for p, a in zip(pred, ans)) / len(ans)
    tp = sum((p == 1 and a == 1) for p, a in zip(pred, ans))
    fp = sum((p == 1 and a == 0) for p, a in zip(pred, ans))
    fn = sum((p == 0 and a == 1) for p, a in zip(pred, ans))

    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)

    return acc, prec, rec


# ============================================================
# 4. Load Data
# ============================================================

def load_raw(fname):
    inst, lab = [], []
    with open(fname) as f:
        f.readline()
        for line in f:
            tmp = line.strip().split(", ")

            # date 제외 1~7 feature 변환
            for i in range(1, 8):
                tmp[i] = float(tmp[i])
            tmp[8] = int(tmp[8])

            inst.append(tmp[:-1])
            lab.append(tmp[-1])
    return inst, lab


# ============================================================
# 5. Automatic Feature Importance
# ============================================================

ALL_FEATURES = [1, 2, 3, 4, 5, 6, 7]   # 전체 feature


def evaluate_feature_importance(train_file, test_file, auto_model):
    logging.info("=== FEATURE IMPORTANCE TEST ===")

    trainX, trainY = load_raw(train_file)
    testX, testY = load_raw(test_file)

    # Base model with all features
    base_params = training(
        trainX, trainY,
        smoothing=1e-6,
        selected_features=ALL_FEATURES,
        auto_model=auto_model
    )
    base_pred = [predict(inst, base_params, ALL_FEATURES) for inst in testX]
    base_acc, base_prec, base_rec = evaluate(base_pred, testY)

    logging.info(f"Base ACC={base_acc:.3f} PRE={base_prec:.3f} REC={base_rec:.3f}")

    results = []

    for f in ALL_FEATURES:
        reduced = [x for x in ALL_FEATURES if x != f]

        logging.info(f"\n--- Test WITHOUT feature {f} ---")

        params = training(
            trainX, trainY,
            smoothing=1e-6,
            selected_features=reduced,
            auto_model=auto_model
        )
        pred = [predict(inst, params, reduced) for inst in testX]
        acc, prec, rec = evaluate(pred, testY)

        logging.info(f"WITHOUT {f} → ACC={acc:.3f} PRE={prec:.3f} REC={rec:.3f}")
        results.append((f, acc, prec, rec, reduced))

    selected = []
    removed = []

    # 개선 기준: precision 또는 recall이 5% 이상 향상될 때만 제거
    for f, acc, prec, rec, sub in results:
        improve_prec = prec > base_prec * 1.05
        improve_rec = rec > base_rec * 1.05

        if improve_prec or improve_rec:
            removed.append(f)
        else:
            selected.append(f)

    # 만약 하나도 안 남으면, 가장 중요한 feature 1개라도 강제 선택
    if len(selected) == 0:
        logging.info("No feature remained by strict rule → selecting most important one.")
        diffs = []
        for f, acc, prec, rec, sub in results:
            # base에서 얼마나 성능이 떨어지는지 기준
            delta = (base_prec - prec) + (base_rec - rec)
            diffs.append((delta, f))
        diffs.sort(reverse=True)
        best_feature = diffs[0][1]
        selected = [best_feature]

    logging.info(f"\nSelected Features = {selected}")
    logging.info(f"Removed = {removed}")

    return selected


# ============================================================
# 6. Full Run
# ============================================================

def run(train_file, test_file, auto=False, auto_model=False):
    trainX, trainY = load_raw(train_file)
    testX, testY = load_raw(test_file)

    # Step 1: feature selection
    if auto:
        selected = evaluate_feature_importance(train_file, test_file, auto_model)
    else:
        selected = ALL_FEATURES

    # Step 2: hyperparameter tuning
    smoothing_vals = [1e-12, 1e-9, 1e-6, 1e-4, 1e-2]
    results = []

    for sm in smoothing_vals:
        params = training(trainX, trainY, sm, selected, auto_model)
        pred = [predict(inst, params, selected) for inst in testX]
        acc, prec, rec = evaluate(pred, testY)
        results.append((acc + rec, sm, acc, prec, rec))
        logging.info(
            f"Smoothing={sm:.0e} → ACC={acc:.3f}, PRE={prec:.3f}, REC={rec:.3f}"
        )

    best = max(results, key=lambda x: x[0])
    _, best_sm, acc, prec, rec = best

    logging.info(f"\nBEST smoothing = {best_sm}")
    logging.info(f"FINAL Accuracy={acc:.3f} Precision={prec:.3f} Recall={rec:.3f}")


# ============================================================
# 7. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training", required=True)
    parser.add_argument("-u", "--testing", required=True)
    parser.add_argument("--auto", action="store_true",
                        help="Automatic feature selection")
    parser.add_argument("--automodel", action="store_true",
                        help="Automatic distribution selection (AIC)")
    parser.add_argument("-l", "--log", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log)

    if not os.path.exists(args.training):
        logging.error(f"Training file not found: {args.training}")
        sys.exit(1)
    if not os.path.exists(args.testing):
        logging.error(f"Testing file not found: {args.testing}")
        sys.exit(1)

    run(args.training, args.testing,
        auto=args.auto,
        auto_model=args.automodel)


if __name__ == "__main__":
    main()
