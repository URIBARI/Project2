import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, expon, weibull_min
import os


def safe_fit_lognorm(data):
    # lognormal requires strictly positive data
    if np.any(data <= 0):
        return None  # impossible
    try:
        s, loc, scale = lognorm.fit(data, floc=0)
        loglik = np.sum(lognorm.logpdf(data, s, loc, scale))
        aic = 2*3 - 2*loglik
        return ("Lognormal", loglik, aic, (s, loc, scale))
    except Exception:
        return None


def safe_fit_exponential(data):
    # exponential requires non-negative data
    if np.any(data < 0):
        return None
    try:
        loc, scale = expon.fit(data, floc=0)
        loglik = np.sum(expon.logpdf(data, loc, scale))
        aic = 2*2 - 2*loglik
        return ("Exponential", loglik, aic, (loc, scale))
    except Exception:
        return None


def fit_and_score(data, feature_name, save_dir="dist_plots"):
    os.makedirs(save_dir, exist_ok=True)

    data = np.array(data)
    data = data[~np.isnan(data)]

    results = {}

    # Gaussian
    mu, sigma = norm.fit(data)
    loglik_gauss = np.sum(norm.logpdf(data, mu, sigma))
    aic_gauss = 2*2 - 2*loglik_gauss
    results["Gaussian"] = (loglik_gauss, aic_gauss, (mu, sigma))

    # Weibull
    c, loc, scale = weibull_min.fit(data, floc=0)
    loglik_weib = np.sum(weibull_min.logpdf(data, c, loc, scale))
    aic_weib = 2*3 - 2*loglik_weib
    results["Weibull"] = (loglik_weib, aic_weib, (c, loc, scale))

    # Lognormal (if possible)
    ln_result = safe_fit_lognorm(data)
    if ln_result is not None:
        _, ll, aic, params = ln_result
        results["Lognormal"] = (ll, aic, params)

    # Exponential (if possible)
    exp_result = safe_fit_exponential(data)
    if exp_result is not None:
        _, ll, aic, params = exp_result
        results["Exponential"] = (ll, aic, params)

    # Select best AIC
    best = sorted(results.items(), key=lambda x: x[1][1])[0]

    print(f"\n=== Feature: {feature_name} ===")
    for dist, (ll, aic, params) in results.items():
        print(f"{dist:<12}  loglik={ll:.2f}  AIC={aic:.2f}  params={params}")

    print(f"\n👉 Best distribution: **{best[0]}**")

    # Plot
    x = np.linspace(min(data), max(data), 300)
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=25, density=True, alpha=0.5, label="Data")

    plt.plot(x, norm.pdf(x, *results["Gaussian"][2]), label="Gaussian")
    plt.plot(x, weibull_min.pdf(x, *results["Weibull"][2]), label="Weibull")

    if "Lognormal" in results:
        plt.plot(x, lognorm.pdf(x, *results["Lognormal"][2]), label="Lognormal")

    if "Exponential" in results:
        plt.plot(x, expon.pdf(x, *results["Exponential"][2]), label="Exponential")

    plt.title(f"Distribution Fit: {feature_name}\nBest: {best[0]}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/{feature_name.replace(' ', '_')}.png")
    plt.close()

    return best[0]


def main():
    df = pd.read_csv("training.csv")
    df.columns = [c.strip() for c in df.columns]

    features = [
        "avg (temperature)", "max (temperature)", "min (temperature)",
        "avg (humidity)", "max (humidity)", "min (humidity)",
        "power"
    ]

    best_map = {}

    for col in features:
        best_map[col] = fit_and_score(df[col].values, col)

    print("\n=== FINAL BEST DISTRIBUTIONS ===")
    for k, v in best_map.items():
        print(f"{k:<20}: {v}")


if __name__ == "__main__":
    main()
