import itertools
import requests
import time
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from itertools import islice
import os
import pickle
from utils.pipeline import *

from sklearn.linear_model import LinearRegression
import pandas as pd

# Load data -- todo: optimize search so it does not take too long or crash
# find a way to only ask for specific authors / institutions instead of taking all
fnAll = "dfAll.p"
if os.path.exists(fnAll):
    dfAll = pickle.load(open(fnAll, "rb"))
else:
    dfAll={}

# Get all institutions authors (test)
allInstitutions = [
    "i4210087039", # 1 Instituto Investig. Biomedicas
    "i4210130807", # 2 Instituto Astrofisico de Canarias
    "i4210107147", # 3 Instituto Catalan de Oncologia
    "i4210120109", # 4 Museo Nac. Ciencias Naturales
    "i4210151127", # 5 Inst. Geociencias
    "i4210126640", # 6 Inst. Filosofia
    "i4210165411", # 7 CIEMAT
    "i4210113665", # 8 IIB Granada
    "i4210086614", # 9 Inst. Invest. 12 de octubre
    "i4210105802", # 10 Inst. Historia
    "i4210105141", # 11 Bioingenieria de Zaragoza
    "i4210118429", # 12 Ciencia de materiales de Madrid
    "i4210146061", # 13 CBM
    "i2799803557", # 14 BSC
    "i4210102407", # 15 Inst. Invest. Vall d'Hebron
    "i4210147680", # 16 CIB
    "i4210151560", # 17 Ciencias del Mar
    "i4210159146", # 18 Inst. Astrofisica de Andalucia
    "i4210148332", # 19 Barcelona Global Health
    "i4210129656", # 20 Ecologia
]

year=2020 # todo use this in the ddbb search!?
for inst_id in allInstitutions:
    if inst_id not in dfAll:
       aids = authors_working_at_institution_in_year(inst_id, year)
       print(inst_id,len(aids))
       df = build_author_df_and_unique_work_distributions(aids, Y=year, mailto=MAILTO, sleep_s=0.25)
       df = df[(df["count1"] > 0)].reset_index(drop=True)
       dfAll[inst_id]=df
       pickle.dump(dfAll, open(fnAll, "wb"))


# Get citations
cols = ["count", "citationAvg", "maxCitation"]
dfClean = {}
parts = []

for inst_id, df in dfAll.items():
    d = df.loc[df["count1"] > 1].copy()

    d["citationAvg1"] = d["citations1"] / d["count1"]
    d["citationAvg2"] = d["citations2"] / d["count2"]

    for suffix in ["1", "2"]:
        for c in cols:
            col = f"{c}{suffix}"
            d[f"{col}Perc"] = d[col].rank(pct=True)

    # collect only percentile columns
    perc_cols = [f"{c}{suffix}Perc" for suffix in ["1", "2"] for c in cols]
    out = d.loc[:, perc_cols].copy()

    parts.append(out)
    dfClean[inst_id] = d

dfPercAll = pd.concat(parts, axis=0, ignore_index=True)


# Pearson correlation
dfPercAll['avgPerc1'] = (dfPercAll['count1Perc'] + dfPercAll['citationAvg1Perc'] + dfPercAll['maxCitation1Perc'])/3
dfPercAll['avgPerc2'] = (dfPercAll['count1Perc'] + dfPercAll['citationAvg2Perc'] + dfPercAll['maxCitation2Perc'])/3
set1 = [
    "count1Perc",
    "citationAvg1Perc",
    "maxCitation1Perc",
    "avgPerc1"
]

set2 = [
    "count2Perc",
    "citationAvg2Perc",
    "maxCitation2Perc",
    "avgPerc2"
]
corr12 = dfPercAll[set1 + set2].corr(method="pearson").loc[set1, set2]
print(corr12)


# Compute VIF (variance_inflation_factor)
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

def compute_vif(df, features):
    X = df[features].dropna()
    X = sm.add_constant(X)  # required for statsmodels

    vif = pd.DataFrame({
        "variable": X.columns,
        "VIF": [variance_inflation_factor(X.values, i)
                for i in range(X.shape[1])]
    })

    return vif
set1 = ["count1Perc", "citationAvg1Perc"]
set1 = ["count1Perc", "maxCitation1Perc"]
vif_set1 = compute_vif(dfPercAll, set1)
print(vif_set1)


# PCA analysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

set1 = [
    "count1Perc",
    "citationAvg1Perc",
    "maxCitation1Perc",
]

df_X = dfPercAll[set1].dropna()
X = df_X.values

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_std)

explained_var = pca.explained_variance_ratio_

dfExplained = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(len(set1))],
    "ExplainedVariance": explained_var,
    "Cumulative": np.cumsum(explained_var)
})

print(dfExplained)


# Loadings
dfLoadings = pd.DataFrame(
    pca.components_.T,
    index=set1,
    columns=[f"PC{i+1}" for i in range(len(set1))]
)

print(dfLoadings)


# Get best models
set1 = ["count1Perc", "maxCitation1Perc"]
def max_vif(X):
    Xc = sm.add_constant(X)
    vifs = [
        variance_inflation_factor(Xc.values, i)
        for i in range(1, Xc.shape[1])  # skip intercept
    ]
    return max(vifs)

vif_max = 5
results = []

for yvar in set2:
    df_base = dfPercAll[set1 + [yvar]].dropna()

    for k in range(1, len(set1) + 1):
        for predictors in itertools.combinations(set1, k):
            X = df_base[list(predictors)]
            y = df_base[yvar]

            # VIF check
            if max_vif(X) > vif_max:
                continue

            model = LinearRegression().fit(X, y)
            r2 = model.score(X, y)

            results.append({
                "target": yvar,
                "predictors": predictors,
                "k": k,
                "R2": r2,
                "maxVIF": max_vif(X),
                "n": len(df_base)
            })

dfModels = pd.DataFrame(results)
best_models = (
  dfModels
  .sort_values(["target", "R2"], ascending=[True, False])
  .groupby("target")
  .head(1)
)

print(best_models)


# Model statistics
df_xy = dfPercAll[["avgPerc1", "avgPerc2"]].dropna()

X = df_xy[["avgPerc1"]]
y = df_xy["avgPerc2"]

model = LinearRegression()
model.fit(X, y)

r2 = model.score(X, y)
coef = model.coef_[0]
intercept = model.intercept_

print(f"avgPerc2 = {intercept:.3f} + {coef:.3f} · avgPerc1")
print(f"R² = {r2:.3f}")



# Scatter plot
import matplotlib.pyplot as plt

x_vals = np.linspace(X.min().values[0], X.max().values[0], 200).reshape(-1, 1)
y_pred = model.predict(x_vals)

plt.scatter(X, y, alpha=0.2)
plt.plot(x_vals, x_vals, linewidth=2, label="Identity", color="green")
plt.plot(x_vals, y_pred, linewidth=2, label="Predicted (no intercept)", color="red")

plt.xlabel("avgPerc1")
plt.ylabel("avgPerc2")
plt.show()


# Info about best models
rows = []

for _, row in best_models.iterrows():
    target = row["target"]
    predictors = list(row["predictors"])

    df_xy = dfPercAll[predictors + [target]].dropna()
    X = df_xy[predictors]
    y = df_xy[target]

    model = LinearRegression()
    model.fit(X, y)

    r2 = model.score(X, y)

    for p, c in zip(predictors, model.coef_):
        rows.append({
            "target": target,
            "predictor": p,
            "coef": c,
            "R2_model": r2,
            "n": len(df_xy)
        })

dfCoefs = pd.DataFrame(rows)
print(dfCoefs)


# Coefficients
from sklearn.linear_model import LinearRegression
import pandas as pd

rows = []

for _, row in best_models.iterrows():
    target = row["target"]
    predictors = ['count1Perc']

    df_xy = dfPercAll[predictors + [target]].dropna()
    X = df_xy[predictors]
    y = df_xy[target]

    model = LinearRegression()
    model.fit(X, y)

    r2 = model.score(X, y)

    for p, c in zip(predictors, model.coef_):
        rows.append({
            "target": target,
            "predictor": p,
            "coef": c,
            "R2_model": r2,
            "n": len(df_xy)
        })

dfCoefs = pd.DataFrame(rows)
print(dfCoefs)


# Coefficients 2
from sklearn.linear_model import LinearRegression
import pandas as pd

rows = []

for _, row in best_models.iterrows():
    target = row["target"]
    predictors = ['maxCitation1Perc']

    df_xy = dfPercAll[predictors + [target]].dropna()
    X = df_xy[predictors]
    y = df_xy[target]

    model = LinearRegression()
    model.fit(X, y)

    r2 = model.score(X, y)

    for p, c in zip(predictors, model.coef_):
        rows.append({
            "target": target,
            "predictor": p,
            "coef": c,
            "R2_model": r2,
            "n": len(df_xy)
        })

dfCoefs = pd.DataFrame(rows)
print(dfCoefs)


# More statistics
rows = []

for _, row in best_models.iterrows():
    target = row["target"]
    predictors = list(row["predictors"])

    df_xy = dfPercAll[predictors + [target]].dropna()
    y = df_xy[target]

    # full model
    X_full = df_xy[predictors]
    r2_full = LinearRegression().fit(X_full, y).score(X_full, y)

    for p in predictors:
        reduced = [x for x in predictors if x != p]
        X_red = df_xy[reduced]

        r2_red = LinearRegression().fit(X_red, y).score(X_red, y)

        rows.append({
            "target": target,
            "predictor": p,
            "partial_R2": r2_full - r2_red,
            "R2_full": r2_full
        })

dfPartial = pd.DataFrame(rows)
print(dfPartial)


# Scatter plots
xvar = "count1Perc"
yvars = [
    "count2Perc",
    "citationAvg2Perc",
    "maxCitation2Perc",
    "avgPerc2",
]

fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=False)
axes = axes.ravel()

for ax, yvar in zip(axes, yvars):
    df_xy = dfPercAll[[xvar, yvar]].dropna()

    ax.scatter(df_xy[xvar], df_xy[yvar], alpha=0.2)
    ax.set_title(f"{yvar} vs {xvar}")
    ax.set_xlabel(xvar)
    ax.set_ylabel(yvar)
    ax.grid(True)

# optional: make x scale identical and meaningful
for ax in axes:
    ax.set_xlim(0, 1)

plt.tight_layout()
plt.show()


# Scatter plots
set1 = [
    "count1Perc",
    "citationAvg1Perc",
    "maxCitation1Perc",
    "avgPerc1",
]

set2 = [
    "count2Perc",
    "citationAvg2Perc",
    "maxCitation2Perc",
    "avgPerc2",
]

fig, axes = plt.subplots(
    nrows=len(set2),
    ncols=len(set1),
    figsize=(8,8),
    sharex=True,
    sharey=False
)

for i, yvar in enumerate(set2):
    for j, xvar in enumerate(set1):
        ax = axes[i, j]
        df_xy = dfPercAll[[xvar, yvar]].dropna()

        ax.scatter(df_xy[xvar], df_xy[yvar], alpha=0.35)

        if i == len(set2) - 1:
            ax.set_xlabel(xvar)
        if j == 0:
            ax.set_ylabel(yvar)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True)

plt.tight_layout()
plt.show()


# Heatmap
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None

set1 = ["count1Perc", "citationAvg1Perc", "maxCitation1Perc", "avgPerc1"]
set2 = ["count2Perc", "citationAvg2Perc", "maxCitation2Perc", "avgPerc2"]

bins = 60
sigma = 3  # smoothing strength

fig, axes = plt.subplots(
    nrows=len(set2),
    ncols=len(set1),
    figsize=(8,8),
    sharex=True,
    sharey=True
)

for i, yvar in enumerate(set2):
    for j, xvar in enumerate(set1):
        ax = axes[i, j]
        df_xy = dfPercAll[[xvar, yvar]].dropna()

        H, xedges, yedges = np.histogram2d(
            df_xy[xvar].to_numpy(),
            df_xy[yvar].to_numpy(),
            bins=bins,
            range=[[0, 1], [0, 1]],
            density=False
        )

        if gaussian_filter is not None:
            H = gaussian_filter(H, sigma=sigma)

        ax.imshow(
            H.T,
            origin="lower",
            extent=(0, 1, 0, 1),
            aspect="auto"
        )

        if i == len(set2) - 1:
            ax.set_xlabel(xvar)
        if j == 0:
            ax.set_ylabel(yvar)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

plt.tight_layout()
plt.show()


# Heatmaps 2
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None

set1 = ["count1Perc", "citationAvg1Perc", "maxCitation1Perc"]
set2 = ["count2Perc", "citationAvg2Perc", "maxCitation2Perc"]

bins = 60
sigma = 3  # smoothing strength

fig, axes = plt.subplots(
    nrows=len(set2),
    ncols=len(set1),
    figsize=(8,8),
    sharex=True,
    sharey=True
)

for i, yvar in enumerate(set2):
    for j, xvar in enumerate(set1):
        ax = axes[i, j]
        df_xy = dfPercAll[[xvar, yvar]].dropna()

        H, xedges, yedges = np.histogram2d(
            df_xy[xvar].to_numpy(),
            df_xy[yvar].to_numpy(),
            bins=bins,
            range=[[0, 1], [0, 1]],
            density=False
        )

        if gaussian_filter is not None:
            H = gaussian_filter(H, sigma=sigma)

        ax.imshow(
            H.T,
            origin="lower",
            extent=(0, 1, 0, 1),
            aspect="auto"
        )

        if i == len(set2) - 1:
            ax.set_xlabel(xvar)
        if j == 0:
            ax.set_ylabel(yvar)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

plt.tight_layout()
plt.show()


# Heatmaps 3
set1 = "avgPerc1"
set2 = ["count2Perc", "citationAvg2Perc", "maxCitation2Perc", "avgPerc2"]

bins = 60
sigma = 3  # smoothing strength

fig, axes = plt.subplots(
    nrows=1,
    ncols=len(set2),
    figsize=(4 * len(set2), 4),
    sharex=True,
    sharey=True
)

for i, yvar in enumerate(set2):
    ax = axes[i]
    df_xy = dfPercAll[[set1, yvar]].dropna()

    H, xedges, yedges = np.histogram2d(
        df_xy[set1].to_numpy(),
        df_xy[yvar].to_numpy(),
        bins=bins,
        range=[[0, 1], [0, 1]],
        density=False
    )

    if gaussian_filter is not None:
        H = gaussian_filter(H, sigma=sigma)

    ax.imshow(
        H.T,
        origin="lower",
        extent=(0, 1, 0, 1),
        aspect="auto"
    )

    ax.set_title(yvar)
    ax.set_xlabel(set1)
    ax.set_ylabel(yvar)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.show()


# Citations Avg
cols = ["count", "citationAvg", "maxCitation"]
dfClean = {}
parts = []

for inst_id, df in dfAll.items():
    d = df.loc[df["count1"] > 1].copy()

    d["citationAvg1"] = d["citations1"] / d["count1"]
    d["citationAvg2"] = d["citations2"] / d["count2"]

    collect_cols = [f"{c}{suffix}" for suffix in ["1", "2"] for c in cols]
    out = d.loc[:, collect_cols].copy()

    parts.append(out)
    dfClean[inst_id] = d

dfAll = pd.concat(parts, axis=0, ignore_index=True)


# CDFs
# 1) compute the best-decile threshold for maxCitation2 (raw values)
x_mc2 = dfAll["maxCitation2"].dropna().to_numpy()
thr = np.quantile(x_mc2, 0.9)

# 2) subset: maxCitation2 in the best decile
df_top = dfAll[dfAll["maxCitation2"] > thr].copy()

# 3) empirical CDF helper
def ecdf(x):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    x = np.sort(x)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

# 4) compute CDFs
x_count2, y_count2 = ecdf(df_top["count2"].to_numpy())
x_cavg2,  y_cavg2  = ecdf(df_top["citationAvg2"].to_numpy())

# 5) medians on the subset
med_count2 = float(np.median(x_count2)) if len(x_count2) else np.nan
med_cavg2  = float(np.median(x_cavg2))  if len(x_cavg2)  else np.nan

print(f"maxCitation2 90th percentile threshold: {thr:.3f}")
print(f"Subset size (top decile by maxCitation2): {len(df_top)}")
print(f"Median count2 (subset): {med_count2}")
print(f"Median citationAvg2 (subset): {med_cavg2}")

# 6) plot CDFs side-by-side
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

axes[0].plot(x_count2, y_count2)
axes[0].axvline(med_count2, linestyle="--")
axes[0].set_title("CDF of count2 (top decile maxCitation2)")
axes[0].set_xlabel("count2")
axes[0].set_ylabel("CDF")

axes[1].plot(x_cavg2, y_cavg2)
axes[1].axvline(med_cavg2, linestyle="--")
axes[1].set_title("CDF of citationAvg2 (top decile maxCitation2)")
axes[1].set_xlabel("citationAvg2")

plt.tight_layout()
plt.show()


# CDFs 2
# 1) compute the best-decile threshold for maxCitation2 (raw values)
x_mc2 = dfAll["maxCitation2"].dropna().to_numpy()
thr = np.quantile(x_mc2, 0.5)

# 2) subset: maxCitation2 in the best decile
df_top = dfAll[dfAll["maxCitation2"] > thr].copy()

# 3) empirical CDF helper
def ecdf(x):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    x = np.sort(x)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

# 4) compute CDFs
x_count2, y_count2 = ecdf(df_top["count2"].to_numpy())
x_cavg2,  y_cavg2  = ecdf(df_top["citationAvg2"].to_numpy())

# 5) medians on the subset
med_count2 = float(np.median(x_count2)) if len(x_count2) else np.nan
med_cavg2  = float(np.median(x_cavg2))  if len(x_cavg2)  else np.nan

print(f"maxCitation2 50th percentile threshold: {thr:.3f}")
print(f"Subset size (top half by maxCitation2): {len(df_top)}")
print(f"Median count2 (subset): {med_count2}")
print(f"Median citationAvg2 (subset): {med_cavg2}")

# 6) plot CDFs side-by-side
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

axes[0].plot(x_count2, y_count2)
axes[0].axvline(med_count2, linestyle="--")
axes[0].set_xlabel("No. Works 2")
axes[0].set_ylabel("CDF")

axes[1].plot(x_cavg2, y_cavg2)
axes[1].axvline(med_cavg2, linestyle="--")
axes[1].set_xlabel("Avg. Citations 2")

plt.tight_layout()
plt.show()


# plot_ccdfs
def plot_ccdf(x, ax, label=None):
    x = np.asarray(x)
    x = x[x > 0]          # power-law diagnostics require x > 0
    x = np.sort(x)
    y = 1.0 - np.arange(1, len(x) + 1) / len(x)

    ax.loglog(x, y, marker=".", linestyle="none")
    ax.set_xlabel(label if label else "x")
    ax.set_ylabel("P(X ≥ x)")

# Columns to plot
cols = ["count2", "citationAvg2", "maxCitation2"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for ax, c in zip(axes, cols):
    plot_ccdf(dfAll[c], ax=ax, label=c)
    ax.set_title(c)

axes[0].set_ylabel("P(X ≥ x)")

plt.tight_layout()
plt.show()


# Box plots
import pandas as pd
import matplotlib.pyplot as plt

# Work on a copy
df = dfPercAll.copy()

# --------------------------------------------------
# 1. Define decile bins for count2Perc
# --------------------------------------------------
bins = [i / 10 for i in range(11)]
labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(10)]

df["count1Perc_bin"] = pd.cut(
    df["count1Perc"],
    bins=bins,
    labels=labels,
    include_lowest=True,
    right=False
)

# --------------------------------------------------
# 2. Collect maxCitation2Perc per bin
# --------------------------------------------------
box_data = [
    df.loc[df["count1Perc_bin"] == label, "maxCitation2Perc"].dropna()
    for label in labels
]

# --------------------------------------------------
# 3. Plot boxplot
# --------------------------------------------------
plt.figure(figsize=(12, 5))

plt.boxplot(
    box_data,
    showfliers=False,   # suppress extreme outliers (heavy tails)
    widths=0.6
)

plt.xticks(
    ticks=range(1, len(labels) + 1),
    labels=labels,
    rotation=45
)

plt.xlabel("Productivity deciles (count1Perc)")
plt.ylabel("maxCitation2 percentile")

plt.tight_layout()
plt.show()


# Box plots 2
import pandas as pd
import matplotlib.pyplot as plt

# Work on a copy
df = dfPercAll.copy()

# --------------------------------------------------
# 1. Define decile bins for count2Perc
# --------------------------------------------------
bins = [i / 10 for i in range(11)]
labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(10)]

df["avgPerc1_bin"] = pd.cut(
    df["avgPerc1"],
    bins=bins,
    labels=labels,
    include_lowest=True,
    right=False
)

# --------------------------------------------------
# 2. Collect maxCitation2Perc per bin
# --------------------------------------------------
box_data = [
    df.loc[df["avgPerc1_bin"] == label, "maxCitation2Perc"].dropna()
    for label in labels
]

# --------------------------------------------------
# 3. Plot boxplot
# --------------------------------------------------
plt.figure(figsize=(12, 5))

plt.boxplot(
    box_data,
    showfliers=False,   # suppress extreme outliers (heavy tails)
    widths=0.6
)

plt.xticks(
    ticks=range(1, len(labels) + 1),
    labels=labels,
    rotation=45
)

plt.xlabel("Average 1 deciles")
plt.ylabel("maxCitation2 percentile")

plt.tight_layout()
plt.show()


# Boxplots 3
# --------------------------------------------------
# 2. Collect count2Perc per bin
# --------------------------------------------------
box_data = [
    df.loc[df["avgPerc1_bin"] == label, "count2Perc"].dropna()
    for label in labels
]

# --------------------------------------------------
# 3. Plot boxplot
# --------------------------------------------------
plt.figure(figsize=(12, 5))

plt.boxplot(
    box_data,
    showfliers=False,   # suppress extreme outliers (heavy tails)
    widths=0.6
)

plt.xticks(
    ticks=range(1, len(labels) + 1),
    labels=labels,
    rotation=45
)

plt.xlabel("Average 1 deciles")
plt.ylabel("count2 percentile")

plt.tight_layout()
plt.show()


# Boxplots 4
# --------------------------------------------------
# 2. Collect citationAvg2 per bin
# --------------------------------------------------
box_data = [
    df.loc[df["avgPerc1_bin"] == label, "citationAvg2Perc"].dropna()
    for label in labels
]

# --------------------------------------------------
# 3. Plot boxplot
# --------------------------------------------------
plt.figure(figsize=(12, 5))

plt.boxplot(
    box_data,
    showfliers=False,   # suppress extreme outliers (heavy tails)
    widths=0.6
)

plt.xticks(
    ticks=range(1, len(labels) + 1),
    labels=labels,
    rotation=45
)

plt.xlabel("Average 1 deciles")
plt.ylabel("citationAvg2 percentile")

plt.tight_layout()
plt.show()


# todo this is what i need
fnAll = "dfAll.p"
if os.path.exists(fnAll):
    dfAll = pickle.load(open(fnAll, "rb"))
else:
    dfAll={}

# budget allocation
cols = ["count", "citationAvg", "maxCitation"]
dfClean = {}
parts = []

for inst_id, df in dfAll.items():
    d = df.loc[df["count1"] > 1].copy()

    d["citationAvg1"] = d["citations1"] / d["count1"]
    d["citationAvg2"] = d["citations2"] / d["count2"]

    for suffix in ["1","2"]:
      for c in cols:
          col = f"{c}{suffix}"
          d[f"{col}Perc"] = d[col].rank(pct=True)
    d["avgPerc1"]=(d["citationAvg1Perc"]+d["count1Perc"]+d["maxCitation1Perc"])/3
    d["avgPerc2"]=(d["citationAvg2Perc"]+d["count2Perc"]+d["maxCitation2Perc"])/3

    # collect only percentile columns
    perc_cols = [f"{c}{suffix}Perc" for c in cols for suffix in ["1", "2"]]
    out = d.loc[:, ["authorID","avgPerc1","avgPerc2"]+perc_cols].copy()

    parts.append(out)
    dfClean[inst_id] = d

df = pd.concat(parts, axis=0, ignore_index=True)
df = (
    df
    .sort_values(by="avgPerc1", ascending=False)
    .reset_index(drop=True)
)


# Parameters
def apply_floor_cap_proportionally(b, B, b_min=0.0, b_max=np.inf, max_iter=200, tol=1e-9):
    """
    Enforce per-researcher minimum/maximum funding while keeping sum(b)=B.
    Simple iterative waterfilling-style adjustment.

    Parameters
    ----------
    b : array-like
        Initial allocations (nonnegative).
    B : float
        Total budget.
    b_min : float
        Minimum allocation per researcher (optional).
    b_max : float
        Maximum allocation per researcher (optional).
    max_iter : int
        Max number of adjustment iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    np.ndarray
        Adjusted allocations summing to B (up to numerical tolerance).
    """
    b = np.asarray(b, dtype=float).copy()
    n = len(b)
    if n == 0:
        return b

    # Apply minimum
    if b_min > 0:
        b = np.maximum(b, b_min)

    # If minimums already exceed budget, scale down proportionally
    s = b.sum()
    if s > B and s > 0:
        return b * (B / s)

    # Iteratively enforce maximum and redistribute residual
    for _ in range(max_iter):
        b_prev = b.copy()

        # Cap
        over = b > b_max
        if np.any(over):
            b[over] = b_max

        total = b.sum()
        remaining = B - total

        if abs(remaining) < tol:
            break

        if remaining > 0:
            # redistribute extra to those not at max
            eligible = b < b_max - 1e-15
            if not np.any(eligible):
                break
            weights = b[eligible]
            if weights.sum() <= 1e-15:
                b[eligible] += remaining / eligible.sum()
            else:
                b[eligible] += remaining * (weights / weights.sum())
        else:
            # remove budget from those above min
            eligible = b > b_min + 1e-15
            if not np.any(eligible):
                break
            weights = b[eligible] - b_min
            if weights.sum() <= 1e-15:
                b[eligible] -= (-remaining) / eligible.sum()
            else:
                b[eligible] -= (-remaining) * (weights / weights.sum())

        if np.max(np.abs(b - b_prev)) < tol:
            break

    # Final normalization for small numerical drift
    s = b.sum()
    if s > 0:
        b *= (B / s)
    return b

def allocate_budget(
    df: pd.DataFrame,
    B: float,
    score_col: str,
    alpha: float = 0.3,          # exploration budget share (α)
    lambda_uniform: float = 0.8, # exploration mix (λ): λ*uniform + (1-λ)*score-proportional
    gamma: float = 1.5,          # exploitation concentration (γ)
    b_min: float = 0.0,          # optional floor (b_min)
    b_max: float = np.inf,       # optional cap (b_max)
    id_col: str = "authorID",
    add_columns: bool = True
) -> pd.DataFrame:
    """
    Deterministic hybrid allocation (paper notation):
      B_explore = α B
      b_i^explore = B_explore * [ λ(1/N) + (1-λ) s_i / Σ s ]
      B_exploit = (1-α) B
      b_i^exploit = B_exploit * [ s_i^γ / Σ s^γ ]
      b_i = b_i^explore + b_i^exploit
    with optional bounds b_min <= b_i <= b_max enforced while preserving Σ b_i = B.

    Parameters
    ----------
    df : DataFrame
        Must contain `score_col` and `id_col`.
    B : float
        Total budget.
    score_col : str
        Column name containing the score s_i (higher is better; not necessarily percentile).
    alpha, lambda_uniform, gamma : floats
        Policy hyperparameters (α, λ, γ).
    b_min, b_max : floats
        Optional floor/cap per researcher.
    id_col : str
        Identifier column (default "authorID").
    add_columns : bool
        If False, returns only [id_col, b_total].

    Returns
    -------
    DataFrame
        With columns: score_col, b_explore, b_exploit, b_total (and id_col).
    """
    if df is None or len(df) == 0:
        raise ValueError("df is empty.")
    if score_col not in df.columns:
        raise ValueError(f"score_col='{score_col}' not found in df.")
    if id_col not in df.columns:
        raise ValueError(f"id_col='{id_col}' not found in df.")
    if B <= 0:
        raise ValueError("B must be > 0.")
    if not (0 <= alpha <= 1):
        raise ValueError("alpha must be in [0,1].")
    if not (0 <= lambda_uniform <= 1):
        raise ValueError("lambda_uniform must be in [0,1].")
    if gamma <= 0:
        raise ValueError("gamma must be > 0.")
    if b_min < 0:
        raise ValueError("b_min must be >= 0.")
    if not (b_max > 0):
        raise ValueError("b_max must be > 0 (or np.inf).")
    if b_max < b_min:
        raise ValueError("b_max must be >= b_min.")

    out = df.copy()

    # Scores s_i (nonnegative)
    s = out[score_col].to_numpy(dtype=float)
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    s = np.clip(s, 0.0, None)

    n = len(s)
    if n == 0:
        raise ValueError("df is empty after processing.")

    # If all scores are zero, fall back to uniform (still well-defined)
    if s.sum() <= 1e-15:
        s_norm = np.ones(n) / n
        s_gamma_norm = np.ones(n) / n
    else:
        s_norm = s / s.sum()
        s_gamma = np.power(s, gamma)
        if s_gamma.sum() <= 1e-15:
            s_gamma_norm = np.ones(n) / n
        else:
            s_gamma_norm = s_gamma / s_gamma.sum()

    # Exploration component
    B_explore = alpha * B
    uniform = np.ones(n) / n
    p_explore = lambda_uniform * uniform + (1.0 - lambda_uniform) * s_norm
    b_explore = B_explore * p_explore

    # Exploitation component
    B_exploit = (1.0 - alpha) * B
    b_exploit = B_exploit * s_gamma_norm

    # Total before bounds
    b_total = b_explore + b_exploit

    # Optional floor/cap
    if b_min > 0 or np.isfinite(b_max):
        b_total_adj = apply_floor_cap_proportionally(b_total, B, b_min=b_min, b_max=b_max)

        # After enforcing bounds, keep a best-effort decomposition by scaling components
        scale = b_total_adj / (b_total + 1e-18)
        b_explore = b_explore * scale
        b_exploit = b_exploit * scale
        b_total = b_total_adj

    out["b_explore"] = b_explore
    out["b_exploit"] = b_exploit
    out["b_total"] = b_total

    if not add_columns:
        return out[[id_col, "b_total"]].copy()

    return out


# Utility
def utility_from_params(df, B, score_col, alpha, lambda_uniform, gamma,
                        target_col="avgPerc2", b_min=0.0, b_max=np.inf, id_col="authorID"):
    """
    Compute utility U = b · y where b is the allocated budget vector (b_total)
    and y is the realized/target outcome (avgPerc2 by default).
    """
    alloc = allocate_budget(
        df=df,
        B=B,
        score_col=score_col,
        alpha=alpha,
        lambda_uniform=lambda_uniform,
        gamma=gamma,
        b_min=b_min,
        b_max=b_max,
        id_col=id_col,
        add_columns=True
    )

    # Align and compute dot product
    b = alloc["b_total"].to_numpy(dtype=float)
    y = alloc[target_col].to_numpy(dtype=float)

    mask = np.isfinite(b) & np.isfinite(y)
    if mask.sum() == 0:
        return np.nan

    return float(np.dot(b[mask], y[mask]))


def grid_search_hyperparams(df, B, score_col,
                            alphas=None, lambdas=None, gammas=None,
                            target_col="avgPerc2",
                            b_min=0.0, b_max=np.inf, id_col="authorID"):
    """
    Brute-force grid search over (alpha, lambda_uniform, gamma).
    Returns (best_params, results_df_sorted).
    """
    if alphas is None:
        alphas = np.linspace(0.0, 1.0, 21)          # 0.00, 0.05, ..., 1.00
    if lambdas is None:
        lambdas = np.linspace(0.0, 1.0, 21)         # 0.00, 0.05, ..., 1.00
    if gammas is None:
        gammas = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0])

    rows = []
    for a in alphas:
        for lam in lambdas:
            for g in gammas:
                U = utility_from_params(
                    df=df, B=B, score_col=score_col,
                    alpha=float(a), lambda_uniform=float(lam), gamma=float(g),
                    target_col=target_col, b_min=b_min, b_max=b_max, id_col=id_col
                )
                row={"alpha": float(a), "lambda": float(lam), "gamma": float(g), "utility": U}
                rows.append(row)

    res = pd.DataFrame(rows).sort_values("utility", ascending=False).reset_index(drop=True)
    best = res.iloc[0].to_dict()
    return best, res


# -------------------------
# Example usage
# -------------------------
B = 1                      # total budget
score_col = "avgPerc1"
target_col = "avgPerc2"

best_params, results = grid_search_hyperparams(
    df=df, B=B, score_col=score_col,
    target_col=target_col,
    alphas=np.linspace(0.0, 1.0, 21),
    lambdas=np.linspace(0.0, 1.0, 21),
    gammas=np.linspace(0.1, 3.0, 21),
    # Optional bounds
    b_min=0.0, b_max=np.inf
)

print("Best params:", best_params)
print(results.head(10))


# Utility plots
def plot_utility_scatter(results_df):
    """
    results_df must contain columns:
    ['alpha', 'lambda', 'gamma', 'utility']
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    # Utility vs alpha
    axes[0].scatter(results_df["alpha"], results_df["utility"], s=15, alpha=0.6)
    axes[0].set_xlabel(r"$\alpha$ (exploration weight)")
    axes[0].set_ylabel("Utility")
    axes[0].set_title("Utility vs α")

    # Utility vs lambda
    axes[1].scatter(results_df["lambda"], results_df["utility"], s=15, alpha=0.6)
    axes[1].set_xlabel(r"$\lambda$ (uniform mix)")
    axes[1].set_title("Utility vs λ")

    # Utility vs gamma
    axes[2].scatter(results_df["gamma"], results_df["utility"], s=15, alpha=0.6)
    axes[2].set_xlabel(r"$\gamma$ (concentration)")
    axes[2].set_title("Utility vs γ")

    plt.tight_layout()
    plt.show()


# Call it
plot_utility_scatter(results)



# Allocate budget with best params
alloc = allocate_budget(
        df=df,
        B=1,
        score_col='avgPerc1',
        alpha=best_params['alpha'],
        lambda_uniform=best_params['lambda'],
        gamma=best_params['gamma'],
        id_col='authorID',
        add_columns=True
    )

# PLot
plt.plot(alloc["avgPerc1"],alloc["b_total"])
plt.xlabel("avgPerc1")
plt.ylabel("Fraction of budget")


# Different gammas
gammas = [0.5, 1.0, 5.0]

plt.figure(figsize=(8, 5))

for gamma in gammas:
    alloc = allocate_budget(
        df=df,
        B=1.0,
        score_col="avgPerc1",
        alpha=best_params["alpha"],
        lambda_uniform=best_params["lambda"],
        gamma=gamma,
        id_col="authorID",
        add_columns=True
    )

    # Sort by score for a clean curve
    alloc_sorted = alloc.sort_values("avgPerc1")

    plt.plot(
        alloc_sorted["avgPerc1"].to_numpy(),
        alloc_sorted["b_total"].to_numpy(),
        linewidth=2,
        label=rf"$\gamma={gamma}$"
    )

plt.xlabel(r"Score $s_i$ (avgPerc1)")
plt.ylabel(r"Fraction of allocated budget $b_i$")
plt.title(r"Deterministic allocation $b_i$ as a function of score $s_i$")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



