import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, expon, weibull_min
import os


def fit_and_score(data, feature_name, save_dir="dist_plots"):
    os.makedirs(save_dir, exist_ok=True)

    data = np.array(data)
    data = data[~np.isnan(data)]

    results = {}

    # ================ Gaussian Fit =================
    mu, sigma = norm.fit(data)
    loglik_gauss = np.sum(norm.logpdf(data, mu, sigma))
    aic_gauss = 2*2 - 2*loglik_gauss
    results["Gaussian"] = (loglik_gauss, aic_gauss, (mu, sigma))

    # ================ Lognormal Fit =================
    s, loc, scale = lognorm.fit(data, floc=0)
    loglik_lognorm = np.sum(lognorm.logpdf(data, s, loc, scale))
    aic_lognorm = 2*3 - 2*loglik_lognorm
    results["Lognormal"] = (loglik_lognorm, aic_lognorm, (s, loc, scale))

    # ================ Exponential Fit =================
    loc_e, scale_e = expon.fit(data, floc=0)
    loglik_exp = np.sum(expon.logpdf(data, loc_e, scale_e))
    aic_exp = 2*2 - 2*loglik_exp
    results["Exponential"] = (loglik_exp, aic_exp, (loc_e, scale_e))

    # ================ Weibull Fit =================
    c, loc_w, scale_w = weibull_min.fit(data, floc=0)
    loglik_weibull = np.sum(weibull_min.logpdf(data, c, loc_w, scale_w))
    aic_weibull = 2*3 - 2*loglik_weibull
    results["Weibull"] = (loglik_weibull, aic_weibull, (c, loc_w, scale_w))

    # ================ Pick best distribution by AIC ================
    best = sorted(results.items(), key=lambda x: x[1][1])[0]

    print(f"\n=== Feature: {feature_name} ===")
    for dist, (ll, aic, params) in results.items():
        print(f"{dist:<12}  log-likelihood={ll:.2f}   AIC={aic:.2f}   params={params}")

    print(f"\n👉 BEST distribution for {feature_name}: **{best[0]}**\n")

    # ================ Plot histogram + PDF =================
    x = np.linspace(min(data), max(data), 300)

    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=25, density=True, alpha=0.5, label="Data")

    # Plot PDFs
    plt.plot(x, norm.pdf(x, *results["Gaussian"][2]), label="Gaussian")
    plt.plot(x, lognorm.pdf(x, *results["Lognormal"][2]), label="Lognormal")
    plt.plot(x, expon.pdf(x, *results["Exponential"][2]), label="Exponential")
    plt.plot(x, weibull_min.pdf(x, *results["Weibull"][2]), label="Weibull")
    
    plt.title(f"Distribution Fit: {feature_name}\nBest: {best[0]}")
    plt.legend()
    plt.grid(True)

    plt.savefig(f"{save_dir}/{feature_name.replace(' ', '_')}.png")
    plt.close()

    return best[0]


def main():
    df = pd.read_csv("training.csv")

    # Clean column names
    df.columns = [c.strip() for c in df.columns]

    # Numerical columns only
    features = [
        "avg (temperature)",
        "max (temperature)",
        "min (temperature)",
        "avg (humidity)",
        "max (humidity)",
        "min (humidity)",
        "power"
    ]

    best_map = {}

    for col in features:
        data = df[col].values
        best = fit_and_score(data, col)
        best_map[col] = best

    print("\n====== FINAL BEST DISTRIBUTIONS ======")
    for k, v in best_map.items():
        print(f"{k:<20}: {v}")


if __name__ == "__main__":
    main()
