"""
Machine Learning-Based Organic Farming Suitability Prediction
BCA 5th Semester - PBL Project

Usage:
    python organic_farming_ml.py path/to/dataset.csv      (or .xlsx)

Outputs go to ./outputs (figures + CSV tables + best model).

NOTE: The dataset is SOURCE-CALIBRATED SYNTHETIC data (see README).
      Labels come from an instructional rubric, not field certification.
"""
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve, ConfusionMatrixDisplay)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")

# ----------------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------------
path = sys.argv[1] if len(sys.argv) > 1 else "organic_farming_dataset.csv"
df = pd.read_excel(path) if path.lower().endswith(("xlsx", "xls")) else pd.read_csv(path)
df.columns = [c.strip() for c in df.columns]
print("Shape:", df.shape)
print(df.head())
print("\nDtypes:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# ----------------------------------------------------------------------------
# 2. TARGET + LEAKAGE CONTROL
# ----------------------------------------------------------------------------
target_col = next(c for c in df.columns
                  if "suitab" in c.lower() and "score" not in c.lower())
score_cols = [c for c in df.columns if "score" in c.lower()]


def to_binary(v):
    s = str(v).strip().lower()
    if s in {"1", "1.0", "yes", "true", "suitable"}:
        return 1
    if s in {"0", "0.0", "no", "false", "not suitable", "not_suitable", "unsuitable"}:
        return 0
    return 0 if "not" in s or "un" in s else 1


y = df[target_col].apply(to_binary)
# Suitability_Score is the numeric rubric the label was derived from ->
# using it would be DATA LEAKAGE, so it is removed from the features.
X = df.drop(columns=[target_col] + score_cols)
id_like = [c for c in X.columns if "id" == c.lower() or c.lower().endswith("_id")]
X = X.drop(columns=id_like)
print(f"\nTarget: {target_col} | dropped (leakage/ID): {score_cols + id_like}")
print(y.value_counts().rename({1: "Suitable", 0: "Not Suitable"}))

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

# ----------------------------------------------------------------------------
# 3. EDA
# ----------------------------------------------------------------------------
df.describe(include="all").T.to_csv(OUT / "summary_statistics.csv")

fig, ax = plt.subplots(figsize=(4.5, 4))
y.map({1: "Suitable", 0: "Not Suitable"}).value_counts().plot.bar(
    ax=ax, color=["#2a9d8f", "#e76f51"])
ax.set_title("Target class distribution")
ax.set_ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig(OUT / "01_class_distribution.png", dpi=150); plt.close()

n = len(num_cols)
cols = 4
rows = int(np.ceil(n / cols))
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
for ax, c in zip(axes.ravel(), num_cols):
    sns.histplot(X[c], kde=True, ax=ax, color="#264653")
    ax.set_title(c)
for ax in axes.ravel()[n:]:
    ax.axis("off")
plt.tight_layout(); plt.savefig(OUT / "02_numeric_distributions.png", dpi=130); plt.close()

fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
tmp = X[num_cols].copy(); tmp["target"] = y.values
for ax, c in zip(axes.ravel(), num_cols):
    sns.boxplot(data=tmp, x="target", y=c, ax=ax, palette=["#e76f51", "#2a9d8f"])
    ax.set_xticklabels(["Not Suit.", "Suitable"]); ax.set_xlabel("")
for ax in axes.ravel()[n:]:
    ax.axis("off")
plt.tight_layout(); plt.savefig(OUT / "03_features_by_class.png", dpi=130); plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(X[num_cols].assign(target=y.values).corr(), cmap="coolwarm",
            center=0, annot=len(num_cols) <= 14, fmt=".2f", linewidths=.3)
plt.title("Correlation heatmap")
plt.tight_layout(); plt.savefig(OUT / "04_correlation_heatmap.png", dpi=150); plt.close()

for c in cat_cols[:3]:
    ct = pd.crosstab(X[c], y, normalize="index")[1].sort_values()
    plt.figure(figsize=(7, 3.5))
    ct.plot.barh(color="#2a9d8f")
    plt.xlabel("Share Suitable"); plt.title(f"Suitability rate by {c}")
    plt.tight_layout(); plt.savefig(OUT / f"05_rate_by_{c}.png", dpi=150); plt.close()

# ----------------------------------------------------------------------------
# 4. PREPROCESSING (inside a Pipeline -> no leakage from test set)
# ----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
print("\nTrain:", X_train.shape, "Test:", X_test.shape)


def preprocessor(scale):
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])


models = {
    "Logistic Regression": Pipeline([("pre", preprocessor(True)),
                                     ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))]),
    "Decision Tree": Pipeline([("pre", preprocessor(False)),
                               ("clf", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE))]),
    "KNN": Pipeline([("pre", preprocessor(True)),
                     ("clf", KNeighborsClassifier(n_neighbors=7))]),
    "Random Forest": Pipeline([("pre", preprocessor(False)),
                               ("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))]),
}

# ----------------------------------------------------------------------------
# 5. TRAIN + EVALUATE
# ----------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rows_, roc_data, fitted = [], {}, {}
fig_cm, axes_cm = plt.subplots(1, 4, figsize=(18, 4))
for ax, (name, pipe) in zip(axes_cm, models.items()):
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    rows_.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
        "CV ROC-AUC (mean)": cv_auc.mean(),
        "CV ROC-AUC (std)": cv_auc.std(),
    })
    roc_data[name] = roc_curve(y_test, proba)
    fitted[name] = pipe
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred),
                           display_labels=["Not Suit.", "Suitable"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(name)
    print(f"\n{name}: confusion matrix\n", confusion_matrix(y_test, pred))
plt.tight_layout(); fig_cm.savefig(OUT / "06_confusion_matrices.png", dpi=150); plt.close(fig_cm)

results = pd.DataFrame(rows_).sort_values("ROC-AUC", ascending=False).round(4)
results.to_csv(OUT / "model_comparison.csv", index=False)
print("\nMODEL COMPARISON\n", results.to_string(index=False))

plt.figure(figsize=(6, 5))
for name, (fpr, tpr, _) in roc_data.items():
    auc = results.loc[results.Model == name, "ROC-AUC"].iloc[0]
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC curves"); plt.legend(loc="lower right")
plt.tight_layout(); plt.savefig(OUT / "07_roc_curves.png", dpi=150); plt.close()

melt = results.melt(id_vars="Model",
                    value_vars=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
plt.figure(figsize=(9, 4.5))
sns.barplot(data=melt, x="variable", y="value", hue="Model")
plt.ylim(max(0, melt.value.min() - 0.05), 1.01)
plt.xlabel(""); plt.title("Model comparison")
plt.tight_layout(); plt.savefig(OUT / "08_model_comparison.png", dpi=150); plt.close()

# ----------------------------------------------------------------------------
# 6. TUNE BEST TREE-BASED MODEL + SELECT FINAL MODEL
# ----------------------------------------------------------------------------
grid = GridSearchCV(
    models["Random Forest"],
    {"clf__n_estimators": [100, 200, 300],
     "clf__max_depth": [None, 6, 10],
     "clf__min_samples_leaf": [1, 3]},
    scoring="roc_auc", cv=cv, n_jobs=-1)
grid.fit(X_train, y_train)
tuned = grid.best_estimator_
t_proba = tuned.predict_proba(X_test)[:, 1]
t_pred = tuned.predict(X_test)
print("\nTuned RF best params:", grid.best_params_)
print("Tuned RF test AUC:", round(roc_auc_score(y_test, t_proba), 4),
      "| F1:", round(f1_score(y_test, t_pred), 4))

best_name = results.iloc[0]["Model"]
best_model = fitted[best_name]
if roc_auc_score(y_test, t_proba) > results.iloc[0]["ROC-AUC"]:
    best_name, best_model = "Random Forest (tuned)", tuned
print("Selected model:", best_name)
joblib.dump(best_model, OUT / "best_model.pkl")

# ----------------------------------------------------------------------------
# 7. FEATURE IMPORTANCE (permutation, on original columns -> easy to explain)
# ----------------------------------------------------------------------------
perm = permutation_importance(best_model, X_test, y_test, scoring="roc_auc",
                              n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1)
imp = (pd.DataFrame({"Feature": X.columns,
                     "Importance": perm.importances_mean,
                     "Std": perm.importances_std})
       .sort_values("Importance", ascending=False))
imp.round(4).to_csv(OUT / "feature_importance.csv", index=False)
plt.figure(figsize=(7, 6))
top = imp.head(15).iloc[::-1]
plt.barh(top.Feature, top.Importance, xerr=top.Std, color="#2a9d8f")
plt.xlabel("Drop in ROC-AUC when shuffled"); plt.title(f"Permutation importance - {best_name}")
plt.tight_layout(); plt.savefig(OUT / "09_feature_importance.png", dpi=150); plt.close()
print("\nTop features:\n", imp.head(10).round(4).to_string(index=False))

# Decision-tree rules are the most readable explanation
dt = fitted["Decision Tree"]
from sklearn.tree import plot_tree
names = dt.named_steps["pre"].get_feature_names_out()
plt.figure(figsize=(20, 9))
plot_tree(dt.named_steps["clf"], feature_names=names, class_names=["Not Suit.", "Suitable"],
          filled=True, max_depth=3, fontsize=8)
plt.title("Decision tree (top 3 levels)")
plt.tight_layout(); plt.savefig(OUT / "10_decision_tree.png", dpi=130); plt.close()

# ----------------------------------------------------------------------------
# 8. PREDICT FOR A NEW FIELD (example: first test row)
# ----------------------------------------------------------------------------
sample = X_test.iloc[[0]]
p = best_model.predict_proba(sample)[0, 1]
print("\nSample prediction ->", "Suitable" if p >= .5 else "Not Suitable",
      f"(P(Suitable)={p:.2f})")
print("\nDone. See the 'outputs' folder.")
print("REMINDER: dataset is synthetic - state this limitation in the report.")
