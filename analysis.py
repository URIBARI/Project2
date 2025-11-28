import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


def load_dataset():
    df = pd.read_csv("training.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df


DISTRIBUTIONS = {
    "gaussian": st.norm,
    "gamma": st.gamma,
    "lognorm": st.lognorm,
    "weibull": st.weibull_min,
}


def fit_and_compute_aic(data, dist):
    """
    Fits a distribution and computes AIC.
    """
    params = dist.fit(data)
    loglik = np.sum(dist.logpdf(data, *params))
    k = len(params)
    aic = 2 * k - 2 * loglik
    return aic, params


def analyze_feature(name, data):
    print(f"\n======= FEATURE: {name} =======")

    results = {}

    for dist_name, dist in DISTRIBUTIONS.items():
        try:
            aic, params = fit_and_compute_aic(data, dist)
            results[dist_name] = (aic, params)
        except:
            results[dist_name] = (np.inf, None)

    best_dist = min(results, key=lambda k: results[k][0])
    print(f"Best distribution for {name}: {best_dist}")
    print("AIC values:", {k: round(v[0], 2) for k, v in results.items()})

    plt.figure(figsize=(7, 5))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='gray')

    dist = DISTRIBUTIONS[best_dist]
    best_params = results[best_dist][1]
    xmin, xmax = np.min(data), np.max(data)
    x = np.linspace(xmin, xmax, 300)
    y = dist.pdf(x, *best_params)

    plt.plot(x, y, 'r-', label=f"{best_dist} fit")
    plt.title(f"Distribution Fit: {name}")
    plt.legend()
    plt.grid()
    plt.savefig(f"dist_{name}.png")
    plt.close()

    return best_dist, results


def main():
    df = load_dataset()

    df.columns = df.columns.str.strip().str.lower()

    print("Normalized columns:", df.columns)

    feature_cols = [
        "avg (temperature)",
        "max (temperature)",
        "min (temperature)",
        "avg (humidity)",
        "max (humidity)",
        "min (humidity)",
        "power",
    ]

    summary = {}

    for col in feature_cols:
        if col not in df.columns:
            print(f"Column not found after normalization: {col}")
            continue

        data = df[col].values
        best_dist, full_results = analyze_feature(col, data)
        summary[col] = best_dist

    print("\n\n====== FINAL SUMMARY ======")
    for k, v in summary.items():
        print(f"{k:25s} → best fit: {v}")

    print("\nHistogram + fitted curve saved as PNG files.")



if __name__ == "__main__":
    main()
