"""
Air Quality Index (AQI) Prediction using Ensemble Techniques
==============================================================
Capstone Project 2 — Oh Chen Wei (22043574)
Beijing Multi-Site Air Quality Dataset — Gucheng station

Implements the methodology described in Chapter 3 of the final report:
  1. Load & time-order the Gucheng station data
  2. Handle missing values with forward fill (time-aware)
  3. Screen for physically implausible sensor values ("outliers")
  4. Numerically encode wind direction
  5. Construct a continuous AQI target from PM2.5 (EPA breakpoint interpolation)
  6. 70/30 time-aware train/test split (no shuffling)
  7. Train 3 baseline models: Multiple Linear Regression, SVR, Decision Tree
  8. Train 3 ensemble models: Stacking (bagging of the 3 baselines + meta-model),
     Random Forest (bagging), XGBoost (boosting)
  9. Evaluate every model with RMSE, MAE, R2
 10. Feature importance via permutation importance (+ SHAP if installed)

Run:
    pip install pandas numpy scikit-learn xgboost shap matplotlib
    python FYPcode.py
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless / file-output backend
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================
DATA_PATH = r"C:\Users\Chen Wei\Desktop\Uni Degree\CP2\beijing+multi+site+air+quality+data (1)\PRSA_Data_20130301-20170228\PRSA_Data_Gucheng_20130301-20170228.csv"
OUTPUT_DIR = r"C:\Users\Chen Wei\Desktop\Uni Degree\CP2\code output\excludepm25"
RANDOM_STATE = 42
TRAIN_FRACTION = 0.70  # 70/30 split, per report methodology

# Report's "Data Feature Selection" table (Figure 3.1) lists PM2.5 itself as
# one of the selected predictors, alongside the meteorological variables.
# Keep this True to reproduce the report exactly. NOTE: because the AQI
# target is derived directly from PM2.5 (see AQI construction below), doing
# this makes AQI an almost-deterministic function of one of its own inputs,
# which trivially inflates model performance and drowns out the
# meteorological features in importance analysis (this is discussed in the
# chat response and is worth a paragraph in your Chapter 4 discussion).
# Set to False to instead test how well AQI can be predicted from
# meteorology ALONE, which is the more scientifically interesting version
# of the "does weather add predictive value" question the report poses.
INCLUDE_PM25_AS_FEATURE = False

RAW_FEATURE_COLUMNS = ["PM2.5", "TEMP", "PRES", "DEWP", "RAIN", "wd", "WSPM"]

# 16-point compass -> bearing in degrees (numeric encoding of wind direction)
WD_TO_DEGREES = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5,
    "SE": 135, "SSE": 157.5, "S": 180, "SSW": 202.5, "SW": 225,
    "WSW": 247.5, "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}

# EPA PM2.5 -> AQI breakpoints, current (May 2024) revision, in µg/m3.
# Source: US EPA "Final Updates to the Air Quality Index (AQI) for
# Particulate Matter" fact sheet, Feb 2024 / effective May 6, 2024.
# Each tuple: (conc_low, conc_high, index_low, index_high)
PM25_BREAKPOINTS = [
    (0.0,   9.0,   0,   50),   # Good
    (9.1,   35.4,  51,  100),  # Moderate
    (35.5,  55.4,  101, 150),  # Unhealthy for Sensitive Groups
    (55.5,  125.4, 151, 200),  # Unhealthy
    (125.5, 225.4, 201, 300),  # Very Unhealthy
    (225.5, 325.4, 301, 500),  # Hazardous
]


# ============================================================================
# 1. AQI TARGET CONSTRUCTION
# ============================================================================
def pm25_to_aqi(pm):
    """Convert a PM2.5 concentration (µg/m3) to a continuous AQI value via
    linear breakpoint interpolation, following EPA's official formula:
        AQI = (I_hi - I_lo) / (C_hi - C_lo) * (C - C_lo) + I_lo
    Values above the top official breakpoint (325.4) are extrapolated using
    the same slope as the top segment, since we need a continuous regression
    target rather than a capped category (Gucheng PM2.5 can exceed 325.4
    during severe pollution episodes)."""
    if pd.isna(pm):
        return np.nan
    pm = max(pm, 0.0)
    for c_lo, c_hi, i_lo, i_hi in PM25_BREAKPOINTS:
        if c_lo <= pm <= c_hi:
            return round((i_hi - i_lo) / (c_hi - c_lo) * (pm - c_lo) + i_lo)
    c_lo, c_hi, i_lo, i_hi = PM25_BREAKPOINTS[-1]
    return round((i_hi - i_lo) / (c_hi - c_lo) * (pm - c_lo) + i_lo)


# ============================================================================
# 2. LOAD, CLEAN & PREPROCESS
# ============================================================================
def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Build a proper datetime index and sort chronologically (dataset is
    # already sorted, but we don't rely on that assumption).
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df[["datetime"] + RAW_FEATURE_COLUMNS].copy()

    # --- Missing values: forward fill (time-aware), per methodology ---
    n_missing_before = df[RAW_FEATURE_COLUMNS].isna().sum().sum()
    df[RAW_FEATURE_COLUMNS] = df[RAW_FEATURE_COLUMNS].ffill()
    # Backfill guards against NaNs in the very first row(s), which ffill
    # cannot resolve (no prior value to carry forward).
    df[RAW_FEATURE_COLUMNS] = df[RAW_FEATURE_COLUMNS].bfill()
    n_missing_after = df[RAW_FEATURE_COLUMNS].isna().sum().sum()
    print(f"[preprocess] missing values: {n_missing_before} -> {n_missing_after} "
          f"(forward-fill + backfill)")

    # --- Outlier screening ---
    # The report specifies outliers are "examined case-by-case ... using
    # context of other feature values" rather than blanket removal. We
    # implement this as a first-pass automated screen against physically
    # plausible bounds for Beijing's climate; anything flagged is printed
    # for manual (case-by-case) review rather than silently dropped, since
    # that judgement call belongs to the researcher.
    flagged = screen_outliers(df)
    if flagged.sum() > 0:
        print(f"[preprocess] {flagged.sum()} row(s) flagged for manual outlier review:")
        print(df.loc[flagged])
    else:
        print("[preprocess] no physically-implausible sensor values found "
              "within domain bounds; no rows removed.")

    # --- Wind direction: numeric encoding (compass -> degrees) ---
    df["wd_deg"] = df["wd"].map(WD_TO_DEGREES)
    df = df.drop(columns=["wd"])

    # --- AQI target construction from PM2.5 ---
    df["AQI"] = df["PM2.5"].apply(pm25_to_aqi)

    return df


def screen_outliers(df):
    """Flag rows with values outside physically plausible ranges for
    Beijing's climate. Returns a boolean mask; does not modify df."""
    domain_bounds = {
        "PM2.5": (0, 1000),   # µg/m3, extreme pollution episodes documented
        "TEMP":  (-40, 45),   # °C
        "PRES":  (960, 1050), # hPa, sea-level-adjusted station pressure
        "DEWP":  (-40, 35),   # °C
        "RAIN":  (0, 100),    # mm, hourly precipitation
        "WSPM":  (0, 30),     # m/s
    }
    mask = pd.Series(False, index=df.index)
    for col, (lo, hi) in domain_bounds.items():
        mask |= (df[col] < lo) | (df[col] > hi)
    return mask


# ============================================================================
# 3. TIME-AWARE 70/30 SPLIT
# ============================================================================
def time_aware_split(df, feature_names, target_col="AQI", train_fraction=TRAIN_FRACTION):
    split_idx = int(len(df) * train_fraction)
    X = df[feature_names].values
    y = df[target_col].values

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"[split] train: {X_train.shape[0]} rows "
          f"({df['datetime'].iloc[0].date()} to {df['datetime'].iloc[split_idx - 1].date()})")
    print(f"[split] test:  {X_test.shape[0]} rows "
          f"({df['datetime'].iloc[split_idx].date()} to {df['datetime'].iloc[-1].date()})")

    return X_train, X_test, y_train, y_test


# ============================================================================
# 4. MODELS
# ============================================================================
def build_baseline_models():
    """The 3 baseline regressors, also used as base learners for stacking."""
    return {
        "MLR": LinearRegression(),
        "SVR": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10, epsilon=0.5)),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE),
    }


def build_stacking_model():
    base_learners = [
        ("mlr", LinearRegression()),
        ("svr", make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10, epsilon=0.5))),
        ("dt", DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE)),
    ]
    # cv=5 generates out-of-fold base-model predictions to train the
    # meta-model on, avoiding the meta-model learning from base models'
    # in-sample (overfit) predictions.
    return StackingRegressor(
        estimators=base_learners,
        final_estimator=LinearRegression(),
        cv=5,
        n_jobs=-1,
    )


def build_random_forest():
    return RandomForestRegressor(
        n_estimators=200, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1
    )


def build_xgboost():
    try:
        from xgboost import XGBRegressor
    except ImportError as e:
        raise ImportError(
            "xgboost is not installed. Run: pip install xgboost"
        ) from e
    return XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ============================================================================
# 5. TRAIN, PREDICT & EVALUATE
# ============================================================================
def evaluate(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def train_and_evaluate_all(X_train, X_test, y_train, y_test):
    results = []
    fitted_models = {}
    predictions = {}

    model_groups = [("Baseline", build_baseline_models())]

    model_groups.append(("Ensemble", {"Stacking": build_stacking_model(),
                                       "Random Forest": build_random_forest()}))
    try:
        model_groups[-1][1]["XGBoost"] = build_xgboost()
    except ImportError as e:
        print(f"[warning] skipping XGBoost: {e}")

    for group_name, models in model_groups:
        for name, model in models.items():
            t0 = time.time()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            elapsed = time.time() - t0

            metrics = evaluate(y_test, y_pred)
            metrics["Type"] = group_name
            metrics["Model"] = name
            metrics["Train_time_s"] = round(elapsed, 1)
            results.append(metrics)

            fitted_models[name] = model
            predictions[name] = y_pred

            print(f"[{group_name}] {name:15s} RMSE={metrics['RMSE']:.3f}  "
                  f"MAE={metrics['MAE']:.3f}  R2={metrics['R2']:.4f}  "
                  f"({elapsed:.1f}s)")

    results_df = pd.DataFrame(results)[["Type", "Model", "RMSE", "MAE", "R2", "Train_time_s"]]
    return results_df, fitted_models, predictions


# ============================================================================
# 6. FEATURE IMPORTANCE
# ============================================================================
def permutation_importance_report(fitted_models, X_test, y_test, feature_names,
                                   sample_size=3000, n_repeats=10):
    """Model-agnostic importance for the 3 ensemble models, per methodology
    ('permutation importance or SHAP values'). Uses a test-set subsample to
    keep runtime reasonable for the slower models."""
    rng = np.random.RandomState(RANDOM_STATE)
    n = min(sample_size, len(X_test))
    idx = rng.choice(len(X_test), size=n, replace=False)
    X_sample, y_sample = X_test[idx], y_test[idx]

    importance_tables = {}
    for name in ["Random Forest", "XGBoost", "Stacking"]:
        if name not in fitted_models:
            continue
        model = fitted_models[name]
        pi = permutation_importance(
            model, X_sample, y_sample, n_repeats=n_repeats,
            random_state=RANDOM_STATE, n_jobs=-1
        )
        table = pd.DataFrame({
            "feature": feature_names,
            "importance_mean": pi.importances_mean,
            "importance_std": pi.importances_std,
        }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
        importance_tables[name] = table
        print(f"\n[permutation importance] {name}")
        print(table.to_string(index=False))

    return importance_tables


def shap_report(fitted_models, X_test, feature_names, outdir, sample_size=1000):
    """Optional richer feature-analysis via SHAP (skipped gracefully if the
    shap package isn't installed — permutation importance above already
    satisfies the methodology's 'permutation importance OR SHAP' requirement)."""
    try:
        import shap
    except ImportError:
        print("\n[info] shap not installed — skipping SHAP analysis "
              "(permutation importance above already covers feature analysis). "
              "Run `pip install shap` to enable it.")
        return

    rng = np.random.RandomState(RANDOM_STATE)
    n = min(sample_size, len(X_test))
    idx = rng.choice(len(X_test), size=n, replace=False)
    X_sample = X_test[idx]

    for name in ["Random Forest", "XGBoost"]:
        if name not in fitted_models:
            continue
        model = fitted_models[name]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        plt.figure()
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"shap_summary_{name.replace(' ', '_')}.png"), dpi=150)
        plt.close()
        print(f"[shap] saved summary plot for {name}")


# ============================================================================
# 7. PLOTS
# ============================================================================
def plot_eda(df, outdir):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df["datetime"], df["AQI"], linewidth=0.4)
    ax.set_title("Gucheng station: AQI over time (derived from PM2.5)")
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "eda_aqi_timeseries.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["AQI"], bins=60)
    ax.set_title("Distribution of AQI values")
    ax.set_xlabel("AQI")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "eda_aqi_distribution.png"), dpi=150)
    plt.close(fig)

    corr_cols = ["PM2.5", "TEMP", "PRES", "DEWP", "RAIN", "WSPM", "wd_deg", "AQI"]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_cols)))
    ax.set_yticklabels(corr_cols)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Feature correlation matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "eda_correlation_matrix.png"), dpi=150)
    plt.close(fig)


def plot_metric_comparison(results_df, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, metric in zip(axes, ["RMSE", "MAE", "R2"]):
        colors = ["#4C72B0" if t == "Baseline" else "#DD8452" for t in results_df["Type"]]
        ax.bar(results_df["Model"], results_df[metric], color=colors)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=40)
    fig.suptitle("Model comparison — blue = baseline, orange = ensemble")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "model_comparison.png"), dpi=150)
    plt.close(fig)


def plot_actual_vs_predicted(df, split_idx, predictions, outdir, models=("Random Forest", "XGBoost", "Stacking")):
    test_dates = df["datetime"].iloc[split_idx:].values
    y_test = df["AQI"].iloc[split_idx:].values

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(test_dates, y_test, label="Actual", linewidth=0.6, color="black")
    for name in models:
        if name in predictions:
            ax.plot(test_dates, predictions[name], label=name, linewidth=0.6, alpha=0.8)
    ax.set_title("Actual vs. predicted AQI on the held-out test period")
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "actual_vs_predicted.png"), dpi=150)
    plt.close(fig)


def plot_feature_importance(importance_tables, outdir):
    n = len(importance_tables)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, (name, table) in zip(axes, importance_tables.items()):
        table = table.sort_values("importance_mean")
        ax.barh(table["feature"], table["importance_mean"], xerr=table["importance_std"])
        ax.set_title(f"Permutation importance — {name}")
        ax.set_xlabel("Mean decrease in R2 when shuffled")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "feature_importance.png"), dpi=150)
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("1. Loading & preprocessing data")
    print("=" * 70)
    df = load_and_preprocess(DATA_PATH)

    feature_names = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM", "wd_deg"]
    if INCLUDE_PM25_AS_FEATURE:
        feature_names = ["PM2.5"] + feature_names
    print(f"[features] using: {feature_names}")

    print("\n" + "=" * 70)
    print("2. Exploratory data analysis")
    print("=" * 70)
    plot_eda(df, OUTPUT_DIR)
    print(f"[eda] plots saved to {OUTPUT_DIR}/eda_*.png")

    print("\n" + "=" * 70)
    print("3. Time-aware 70/30 train/test split")
    print("=" * 70)
    X_train, X_test, y_train, y_test = time_aware_split(df, feature_names)
    split_idx = int(len(df) * TRAIN_FRACTION)

    print("\n" + "=" * 70)
    print("4. Training & evaluating models")
    print("=" * 70)
    results_df, fitted_models, predictions = train_and_evaluate_all(
        X_train, X_test, y_train, y_test
    )
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
    print(f"\n[results] saved to {OUTPUT_DIR}/model_comparison.csv")
    print(results_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("5. Feature importance analysis")
    print("=" * 70)
    importance_tables = permutation_importance_report(
        fitted_models, X_test, y_test, feature_names
    )
    shap_report(fitted_models, X_test, feature_names, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("6. Plots")
    print("=" * 70)
    plot_metric_comparison(results_df, OUTPUT_DIR)
    plot_actual_vs_predicted(df, split_idx, predictions, OUTPUT_DIR)
    plot_feature_importance(importance_tables, OUTPUT_DIR)
    print(f"[plots] all figures saved to {OUTPUT_DIR}/")

    print("\nDone.")


if __name__ == "__main__":
    main()
