import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ANOVA Test

# 1) Load data
df = pd.read_csv("training.csv")

# Clean column names (strip spaces)
df.columns = [c.strip() for c in df.columns]

feature_cols = [
    'avg (temperature)', 'max (temperature)', 'min (temperature)',
    'avg (humidity)', 'max (humidity)', 'min (humidity)',
    'power'
]
X = df[feature_cols]
y = df['label']

# 2) Mutual Information
mi_scores = mutual_info_classif(X, y, random_state=42)
mi = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)
print("\n=== Mutual Information ===")
print(mi)

# 3) ANOVA (F-test)
f_scores, p_values = f_classif(X, y)
anova = pd.Series(f_scores, index=feature_cols).sort_values(ascending=False)
print("\n=== ANOVA F-scores ===")
print(anova)

# 4) Logistic Regression Coeff Importance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

log_clf = LogisticRegression(max_iter=500)
log_clf.fit(X_scaled, y)

log_coef = pd.Series(np.abs(log_clf.coef_[0]), index=feature_cols).sort_values(ascending=False)
print("\n=== Logistic Regression Coeff Importance ===")
print(log_coef)

# 5) Permutation Importance (model-agnostic)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

rf = RandomForestClassifier(n_estimators=500)
rf.fit(X_train, y_train)

result = permutation_importance(rf, X_test, y_test, n_repeats=20, random_state=42)
perm = pd.Series(result.importances_mean, index=feature_cols).sort_values(ascending=False)
print("\n=== Permutation Importance (RF) ===")
print(perm)


# 6) Summarize & pick top 3
summary = pd.DataFrame({
    "MI": mi,
    "ANOVA": anova,
    "LOG_COEF": log_coef,
    "PERM": perm
})

summary["RANK_SUM"] = summary.rank(ascending=False).sum(axis=1)
summary_sorted = summary.sort_values("RANK_SUM")

print("\n\n=== Final Summary & Ranking ===")
print(summary_sorted)

print("\n\n=== TOP 3 SELECTED FEATURES ===")
print(summary_sorted.index[:3])