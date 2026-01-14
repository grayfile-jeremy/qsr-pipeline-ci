
import os
import glob
import numpy as np
import pandas as pd

import warnings
# Suppress noisy ParserWarning spam from pandas when skipping malformed lines
warnings.filterwarnings('ignore', category=pd.errors.ParserWarning)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMRegressor

###############################################################################
# 1. Data loading
###############################################################################

def load_all_store_files(data_dir="."):
    pattern = os.path.join(data_dir, "per_check_store_*_small.csv")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")

    dfs = []
    for path in files:
        name = os.path.basename(path)
        store_id = (
            name.replace("per_check_store_", "")
                .replace("_small", "")
                .replace(".csv", "")
        )

        try:
            # Try fast C engine first
            df = pd.read_csv(path, engine='python', on_bad_lines='skip')
        except pd.errors.ParserError as e:
            print(f"\n[READ ERROR] ParserError in file: {path}")
            print("Falling back to python engine and warning on bad lines...")

            # This will continue and print warnings for malformed rows
            df = pd.read_csv(path, engine='python', on_bad_lines='skip')

        df["StoreID"] = store_id
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True, sort=False)
    return full_df



###############################################################################
# 2. Feature engineering
###############################################################################

STATION_COLS = [
    "order_time_sec",
    "grill_time_sec",
    "make_time_sec",
    "expo_time_sec",
    "dt_time_dt_expo_sec",
    "custard_time_sec",
    "window_time_sec",
    "kitchen_duration_sec",
]

ID_LIKE_COLS = {
    "StoreID",
    "StoreKey",
    "DateKey",
    "CheckKey",
    "storenum",
    "dateofbusiness",
    "CheckId",
    "OrderType",
    "NCROrderType",
    "OrderType.1",
    "NCROrderType.1",
    "Day Part",
}

TARGET_COLS = {"total_check_time_sec"} | set(STATION_COLS)

STAFFING_FEATURE_COLS = [
    "labor_total",
    "labor_FOH",
    "labor_BOH",
    "CPLH_total",
    "CPLH_BOH",
    "CPLH_FOH",
    "LLI_BOH",
    "LLI_total",
    "labor_per_check_total",
    "labor_per_check_BOH",
    "labor_per_check_FOH",
]


def map_order_channel(row):
    otype = str(row.get("OrderType", "")).strip()
    ncr = str(row.get("NCROrderType", "")).strip()

    otype_norm = otype.replace("-", " ").lower()
    ncr_norm = ncr.replace("-", " ").lower()

    if "drive" in otype_norm or "drive" in ncr_norm:
        return "drive_thru"
    if "mobile" in ncr_norm:
        return "mobile_carryout"
    if "dine" in otype_norm or "dine" in ncr_norm:
        return "dine_in"
    if "carryout" in otype_norm or "carryout" in ncr_norm or "out f" in ncr_norm:
        return "carryout"
    if any(x in ncr_norm for x in ["ubereats", "grubhub", "doordash", "dispatch", "delivery"]):
        return "delivery"
    if "delivery" in otype_norm:
        return "delivery"
    return "carryout"


def map_daypart(value):
    s = str(value).strip().lower()
    if "lunch" in s:
        return "lunch"
    if "dinner" in s:
        return "dinner"
    return "off_peak"


def to_binary(x):
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return 1
    if s in {"0", "false", "f", "no", "n", ""}:
        return 0
    return 1


def engineer_features(df, enc=None, inplace=False):
    required_cols = [
        "OrderType",
        "NCROrderType",
        "Hotline Red Ticket",
        "DT Trouble Ticket",
        "kitchen_overlap_count",
        "labor_total",
        "labor_FOH",
        "labor_BOH",
        "Day Part",
        "total_check_time_sec",
        "wk",
        "pd",
        "fy",
        "NetSales",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df if inplace else df.copy()
    # Memory safety: prefer float32 for large numeric columns
    for _c in list(df.columns):
        if isinstance(_c, str) and (_c.endswith('_sec') or _c.endswith('_seconds')):
            df[_c] = pd.to_numeric(df[_c], errors='coerce').astype('float32')
    df["OrderChannel"] = df.apply(map_order_channel, axis=1)
    df["DayPartNorm"] = df["Day Part"].apply(map_daypart)

    df["hotline_red"] = df["Hotline Red Ticket"].apply(to_binary)
    df["dt_trouble"] = df["DT Trouble Ticket"].apply(to_binary)

    eps = 1e-6
    df["CPLH_total"] = df["kitchen_overlap_count"] / (df["labor_total"] + eps)
    df["CPLH_BOH"] = df["kitchen_overlap_count"] / (df["labor_BOH"] + eps)
    df["CPLH_FOH"] = df["kitchen_overlap_count"] / (df["labor_FOH"] + eps)

    df["LLI_BOH"] = df["kitchen_overlap_count"] / (df["labor_BOH"] + eps)
    df["LLI_total"] = df["kitchen_overlap_count"] / (df["labor_total"] + eps)

    df["labor_per_check_total"] = df["labor_total"] / (df["kitchen_overlap_count"] + 1)
    df["labor_per_check_BOH"] = df["labor_BOH"] / (df["kitchen_overlap_count"] + 1)
    df["labor_per_check_FOH"] = df["labor_FOH"] / (df["kitchen_overlap_count"] + 1)

    cat_cols = ["OrderChannel", "DayPartNorm"]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df[[c + "_enc" for c in cat_cols]] = enc.fit_transform(df[cat_cols])

    return df, enc


###############################################################################
# 3. Cleaning and feature matrices
###############################################################################

def clean_station_outliers(df, quantile_low=0.005, quantile_high=0.995):
    df = df.copy()
    cols_to_clip = ["total_check_time_sec"] + [c for c in STATION_COLS if c in df.columns]
    for col in cols_to_clip:
        q_low = df[col].quantile(quantile_low)
        q_hi = df[col].quantile(quantile_high)
        df[col] = df[col].clip(lower=q_low, upper=q_hi)
    return df


def build_baseline_feature_matrix(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = set()
    exclude.update(ID_LIKE_COLS)
    exclude.update(TARGET_COLS)
    exclude.update(STAFFING_FEATURE_COLS)
    return [c for c in num_cols if c not in exclude]


def build_staffing_feature_matrix(df):
    cols = [c for c in STAFFING_FEATURE_COLS if c in df.columns]
    for ctx_col in ["OrderChannel_enc", "DayPartNorm_enc"]:
        if ctx_col in df.columns:
            cols.append(ctx_col)
    seen, deduped = set(), []
    for c in cols:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


def compute_safe_pct_delta(delta, expected, min_expected, clip_val=3.0):
    safe_expected = np.maximum(expected, min_expected)
    pct = delta / safe_expected
    pct = np.clip(pct, -clip_val, clip_val)
    return pct


###############################################################################
# 4. Multi-target execution + staffing models (total + stations)
###############################################################################

def fit_execution_and_staffing_multi(
    df,
    target_cols,
    sample_frac=0.5,
    random_state=42,
):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    # Avoid duplicating the full dataframe (very large). We'll only create derived frames as needed.
    baseline_features = build_baseline_feature_matrix(df)
    staffing_features = build_staffing_feature_matrix(df)

    if not baseline_features:
        raise ValueError("No baseline features found.")
    if not staffing_features:
        raise ValueError("No staffing features found.")

    print(f"Execution baseline: using {len(baseline_features)} features.")
    print(f"Staffing model: using {len(staffing_features)} features.")

    min_expected_total = 60.0
    min_expected_station = {
        "order_time_sec": 10.0,
        "grill_time_sec": 20.0,
        "make_time_sec": 15.0,
        "expo_time_sec": 10.0,
        "dt_time_dt_expo_sec": 15.0,
        "custard_time_sec": 10.0,
        "window_time_sec": 20.0,
        "kitchen_duration_sec": 60.0,
    }

    baseline_models = {}
    staffing_models = {}
    metrics_rows = []
    # Avoid duplicating the full dataframe; we will add new columns into an indexed view.
    df_enriched = df
    df_enriched_idx = df_enriched.set_index(["StoreID", "CheckKey"], drop=False)

    for target_col in target_cols:
        if target_col not in df.columns:
            print(f"[WARN] Target column {target_col} not in DataFrame; skipping.")
            continue

        print("\n==================================================")
        print(f"Modeling target: {target_col}")

        df_model = df[df[target_col].notna()].copy()
        print(f"  Non-null rows for {target_col}: {len(df_model)}")
        if len(df_model) == 0:
            print(f"  No data for {target_col}; skipping.")
            continue

        # Baseline model
        if 0 < sample_frac < 1.0:
            df_train = df_model.sample(frac=sample_frac, random_state=random_state)
            print(f"  Baseline training sample for {target_col}: {len(df_train)} rows")
        else:
            df_train = df_model

        X = df_train[baseline_features].to_numpy(dtype=np.float32, copy=False)
        y = df_train[target_col].to_numpy(dtype=np.float32, copy=False)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        baseline_model = LGBMRegressor(
            objective="quantile",
            alpha=0.5,
            n_estimators=400,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=-1,
        )

        print("  Fitting EXECUTION baseline model...")
        baseline_model.fit(X_train, y_train)

        from sklearn.metrics import mean_absolute_error, mean_squared_error
        y_val_pred = baseline_model.predict(X_val)
        mae = mean_absolute_error(y_val, y_val_pred)
        rmse = mean_squared_error(y_val, y_val_pred, squared=False)
        print(f"  Execution baseline ({target_col}) MAE={mae:.2f}, RMSE={rmse:.2f}")

        metrics_rows.append({
            "target": target_col,
            "model_type": "baseline",
            "mae_sec": mae,
            "rmse_sec": rmse,
            "n_val": int(len(y_val)),
        })

        expected_col = f"expected_exec_{target_col}"
        df_model[expected_col] = baseline_model.predict(df_model[baseline_features])

        delta_col = f"exec_delta_{target_col}"
        pct_col = f"exec_pct_delta_{target_col}"
        df_model[delta_col] = df_model[target_col] - df_model[expected_col]

        if target_col == "total_check_time_sec":
            min_expected = min_expected_total
        else:
            min_expected = min_expected_station.get(target_col, 10.0)

        df_model[pct_col] = compute_safe_pct_delta(
            df_model[delta_col],
            df_model[expected_col],
            min_expected=min_expected,
            clip_val=3.0,
        )

        baseline_models[target_col] = baseline_model

        # Staffing model
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        staff_mask = df_model[staffing_features].notna().all(axis=1)
        staff_n = int(staff_mask.sum())
        print(f"  Staffing model rows (non-null features) for {target_col}: {staff_n}")

        if staff_n == 0:
            print(f"  No staffing data for {target_col}; skipping staffing model.")
            continue

        # Build feature/label matrices directly (avoid copying full df slices)
        X2_all = df_model.loc[staff_mask, staffing_features].to_numpy(dtype=np.float32, copy=False)
        y2_all = df_model.loc[staff_mask, delta_col].to_numpy(dtype=np.float32, copy=False)

        # Subsample for training without creating a large DataFrame copy
        if 0 < sample_frac < 1.0:
            rng = np.random.RandomState(random_state)
            n_train = max(1, int(round(sample_frac * staff_n)))
            train_idx = rng.choice(staff_n, size=n_train, replace=False)
            print(f"  Staffing training sample for {target_col}: {n_train} rows")
        else:
            train_idx = None

        X2 = X2_all if train_idx is None else X2_all[train_idx]
        y2 = y2_all if train_idx is None else y2_all[train_idx]


        X2_train, X2_val, y2_train, y2_val = train_test_split(
            X2, y2, test_size=0.2, random_state=random_state
        )

        staffing_model = LGBMRegressor(
            objective="quantile",
            alpha=0.5,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=-1,
        )

        print("  Fitting STAFFING impact model...")
        staffing_model.fit(X2_train, y2_train)

        y2_val_pred = staffing_model.predict(X2_val)
        mae2 = mean_absolute_error(y2_val, y2_val_pred)
        rmse2 = mean_squared_error(y2_val, y2_val_pred, squared=False)
        print(f"  Staffing model ({delta_col}) MAE={mae2:.2f}, RMSE={rmse2:.2f}")

        metrics_rows.append({
            "target": target_col,
            "model_type": "staffing",
            "mae_sec": mae2,
            "rmse_sec": rmse2,
            "n_val": int(len(y2_val)),
        })

        staff_effect_col = f"staffing_effect_{target_col}"
        staff_resid_col = f"staffing_residual_{target_col}"

        df_model.loc[staff_mask, staff_effect_col] = staffing_model.predict(X2_all)
        df_model[staff_resid_col] = df_model[delta_col] - df_model[staff_effect_col]

        staffing_models[target_col] = staffing_model        # Attach per-check predictions without a full DataFrame merge (prevents multi-GB copies)
        tmp = df_model[["StoreID", "CheckKey", expected_col, delta_col, pct_col, staff_effect_col, staff_resid_col]].copy()
        tmp = tmp.set_index(["StoreID", "CheckKey"], drop=True)
        # Ensure columns exist, then assign only the matching rows
        for _c in [expected_col, delta_col, pct_col, staff_effect_col, staff_resid_col]:
            if _c not in df_enriched_idx.columns:
                df_enriched_idx[_c] = np.nan
            # Assign as float32 to reduce memory
            # NOTE: df_enriched_idx may have a non-unique MultiIndex (StoreID, CheckKey) in large datasets.
            # Using .loc[...] assignment triggers an internal reindex that fails on non-unique indexes.
            # Instead, build a key->value mapping (deduped) and map onto the full index (works with duplicates).
            _s = pd.to_numeric(tmp[_c], errors='coerce')
            if not _s.index.is_unique:
                _s = _s.groupby(level=[0, 1]).first()
            # Reindex onto the (possibly non-unique) df_enriched_idx index; this is safe and avoids huge dicts.
            df_enriched_idx[_c] = pd.to_numeric(_s.reindex(df_enriched_idx.index), errors='coerce').astype('float32')


        # Memory cleanup between targets (prevents pandas/NumPy from holding multi-GB arrays)
        try:
            import gc
            del X, y, X_train, X_val, y_train, y_val
            del X2_all, y2_all, X2, y2
            gc.collect()
        except Exception:
            pass


    df_enriched = df_enriched_idx.reset_index(drop=True)
    return df_enriched, baseline_models, staffing_models, baseline_features, staffing_features, metrics_rows


###############################################################################
# 5. Store summary, correlations, and readable outputs
###############################################################################

def summarize_store_execution_and_staffing_multi(
    df,
    target_cols,
    min_checks_per_group=1,
):
    df = df.copy()
    group_cols = ["StoreID", "DayPartNorm", "OrderChannel", "dateofbusiness"]
    for c in group_cols:
        if c not in df.columns:
            raise ValueError(f"Required group column '{c}' not found in DataFrame.")

    agg_dict = {"CheckKey": "count"}
    if "NetSales" in df.columns:
        agg_dict["NetSales"] = "sum"

    for target_col in target_cols:
        if target_col not in df.columns:
            continue
        expected_col = f"expected_exec_{target_col}"
        delta_col = f"exec_delta_{target_col}"
        pct_col = f"exec_pct_delta_{target_col}"
        staff_effect_col = f"staffing_effect_{target_col}"
        staff_resid_col = f"staffing_residual_{target_col}"
        for col in [target_col, expected_col, delta_col, pct_col, staff_effect_col, staff_resid_col]:
            if col in df.columns:
                agg_dict[col] = "mean"

    if "hotline_red" in df.columns:
        agg_dict["hotline_red"] = "mean"
    if "dt_trouble" in df.columns:
        agg_dict["dt_trouble"] = "mean"

    grouped = df.groupby(group_cols).agg(agg_dict).reset_index()
    grouped = grouped.rename(columns={
        "CheckKey": "n_checks",
        "NetSales": "total_sales",
        "hotline_red": "hotline_sla_rate",
        "dt_trouble": "dt_sla_rate",
    })

    rename_map = {}
    for target_col in target_cols:
        if target_col not in df.columns:
            continue
        expected_col = f"expected_exec_{target_col}"
        delta_col = f"exec_delta_{target_col}"
        pct_col = f"exec_pct_delta_{target_col}"
        staff_effect_col = f"staffing_effect_{target_col}"
        staff_resid_col = f"staffing_residual_{target_col}"

        rename_map[target_col] = f"mean_{target_col}"
        rename_map[expected_col] = f"mean_expected_exec_{target_col}"
        rename_map[delta_col] = f"mean_exec_delta_{target_col}"
        rename_map[pct_col] = f"mean_exec_pct_delta_{target_col}"
        rename_map[staff_effect_col] = f"mean_staffing_effect_{target_col}"
        rename_map[staff_resid_col] = f"mean_staffing_residual_{target_col}"

    grouped = grouped.rename(columns=rename_map)

    mask_enough = grouped["n_checks"] >= min_checks_per_group
    for target_col in target_cols:
        if target_col not in df.columns:
            continue
        for c in [
            f"mean_exec_pct_delta_{target_col}",
            f"mean_exec_delta_{target_col}",
            f"mean_staffing_effect_{target_col}",
            f"mean_staffing_residual_{target_col}",
        ]:
            if c in grouped.columns:
                grouped.loc[~mask_enough, c] = np.nan

    return grouped


def compute_full_feature_correlation(df, output_path=None):
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = [c for c in ID_LIKE_COLS if c in num_cols]
    cols = [c for c in num_cols if c not in drop_cols]
    corr = df[cols].corr(method="pearson")
    if output_path is not None:
        corr.to_csv(output_path, index=False)
        print(f"Full feature correlation matrix written to: {output_path}")
    return corr


def rename_columns_readable_per_check(df, inplace=False):
    df = df.copy()
    rename_map = {}

    if "total_check_time_sec" in df.columns:
        rename_map["total_check_time_sec"] = "total_check_time_actual_seconds"
    if "expected_exec_total_check_time_sec" in df.columns:
        rename_map["expected_exec_total_check_time_sec"] = "total_check_time_expected_seconds"
    if "exec_delta_total_check_time_sec" in df.columns:
        rename_map["exec_delta_total_check_time_sec"] = "total_check_time_difference_seconds"
    if "exec_pct_delta_total_check_time_sec" in df.columns:
        rename_map["exec_pct_delta_total_check_time_sec"] = "total_check_time_difference_percent"
    if "staffing_effect_total_check_time_sec" in df.columns:
        rename_map["staffing_effect_total_check_time_sec"] = "total_check_time_staffing_effect_seconds"
    if "staffing_residual_total_check_time_sec" in df.columns:
        rename_map["staffing_residual_total_check_time_sec"] = "total_check_time_execution_residual_seconds"

    for col in STATION_COLS:
        if col in df.columns:
            rename_map[col] = f"{col.replace('_sec','')}_actual_seconds"
        exp = f"expected_exec_{col}"
        if exp in df.columns:
            rename_map[exp] = f"{col.replace('_sec','')}_expected_seconds"
        d = f"exec_delta_{col}"
        if d in df.columns:
            rename_map[d] = f"{col.replace('_sec','')}_difference_seconds"
        p = f"exec_pct_delta_{col}"
        if p in df.columns:
            rename_map[p] = f"{col.replace('_sec','')}_difference_percent"
        se = f"staffing_effect_{col}"
        if se in df.columns:
            rename_map[se] = f"{col.replace('_sec','')}_staffing_effect_seconds"
        sr = f"staffing_residual_{col}"
        if sr in df.columns:
            rename_map[sr] = f"{col.replace('_sec','')}_execution_residual_seconds"

    return df.rename(columns=rename_map)


def rename_columns_readable_store_summary(df, target_cols, inplace=False):
    df = df if inplace else df.copy()
    # Memory safety: prefer float32 for large numeric columns
    for _c in list(df.columns):
        if isinstance(_c, str) and (_c.endswith('_sec') or _c.endswith('_seconds')):
            df[_c] = pd.to_numeric(df[_c], errors='coerce').astype('float32')
    rename_map = {
        "n_checks": "number_of_checks",
        "total_sales": "total_sales_dollars",
        "hotline_sla_rate": "hotline_sla_rate_fraction",
        "dt_sla_rate": "drive_thru_sla_rate_fraction",
    }
    for target_col in target_cols:
        base = target_col.replace("_sec", "")
        mean_actual = f"mean_{target_col}"
        mean_expected = f"mean_expected_exec_{target_col}"
        mean_delta = f"mean_exec_delta_{target_col}"
        mean_pct = f"mean_exec_pct_delta_{target_col}"
        mean_staff = f"mean_staffing_effect_{target_col}"
        mean_resid = f"mean_staffing_residual_{target_col}"
        if mean_actual in df.columns:
            rename_map[mean_actual] = f"average_{base}_actual_seconds"
        if mean_expected in df.columns:
            rename_map[mean_expected] = f"average_{base}_expected_seconds"
        if mean_delta in df.columns:
            rename_map[mean_delta] = f"average_{base}_difference_seconds"
        if mean_pct in df.columns:
            rename_map[mean_pct] = f"average_{base}_difference_percent"
        if mean_staff in df.columns:
            rename_map[mean_staff] = f"average_{base}_staffing_effect_seconds"
        if mean_resid in df.columns:
            rename_map[mean_resid] = f"average_{base}_execution_residual_seconds"
    return df.rename(columns=rename_map)



def reorder_store_summary_columns(store_df, target_cols):
    """Reorder columns in the store summary for readability.

    For each metric (e.g., total_check_time, grill_time), group columns in this order:
      - average_{base}_actual_seconds
      - average_{base}_expected_seconds
      - average_{base}_difference_seconds
      - average_{base}_difference_percent
      - average_{base}_staffing_effect_seconds
      - {base}_staffing_portion_of_difference
      - average_{base}_execution_residual_seconds
      - {base}_execution_portion_of_difference
    Core ID / volume columns (StoreID, DayPartNorm, OrderChannel, number_of_checks, total_sales_dollars,
    percent_of_store_checks, percent_of_store_sales) are kept at the front.
    Any remaining columns are appended at the end in their existing order.
    """
    df = store_df.copy()
    cols = list(df.columns)

    # Start with identity / volume columns if present
    ordered = []
    for c in [
        "StoreID",
        "DayPartNorm",
        "OrderChannel",
        "number_of_checks",
        "total_sales_dollars",
        "percent_of_store_checks",
        "percent_of_store_sales",
    ]:
        if c in cols and c not in ordered:
            ordered.append(c)

    # Group metric columns per target
    for target_col in target_cols:
        base = target_col.replace("_sec", "")
        group = [
            f"average_{base}_actual_seconds",
            f"average_{base}_expected_seconds",
            f"average_{base}_difference_seconds",
            f"average_{base}_difference_percent",
            f"average_{base}_staffing_effect_seconds",
            f"{base}_staffing_portion_of_difference",
            f"average_{base}_execution_residual_seconds",
            f"{base}_execution_portion_of_difference",
        ]
        for c in group:
            if c in cols and c not in ordered:
                ordered.append(c)

    # Append any leftover columns
    for c in cols:
        if c not in ordered:
            ordered.append(c)

    return df[ordered]
def build_column_glossary(per_check_df, store_df, output_path):
    all_cols = sorted(set(per_check_df.columns).union(set(store_df.columns)))
    rows = []
    for col in all_cols:
        desc = None
        if col.endswith("_actual_seconds"):
            base = col.replace("_actual_seconds", "").replace("_", " ")
            desc = f"Actual {base} in seconds."
        elif col.endswith("_expected_seconds"):
            base = col.replace("_expected_seconds", "").replace("_", " ")
            desc = f"Model-predicted expected {base} in seconds."
        elif col.endswith("_difference_seconds"):
            base = col.replace("_difference_seconds", "").replace("_", " ")
            desc = f"Actual minus expected {base} in seconds."
        elif col.endswith("_difference_percent"):
            base = col.replace("_difference_percent", "").replace("_", " ")
            desc = f"Difference between actual and expected {base}, as a percentage of expected."
        elif col.endswith("_staffing_effect_seconds"):
            base = col.replace("_staffing_effect_seconds", "").replace("_", " ")
            desc = f"Portion of the {base} difference explained by staffing levels, in seconds."
        elif col.endswith("_execution_residual_seconds"):
            base = col.replace("_execution_residual_seconds", "").replace("_", " ")
            desc = f"Remaining difference in {base} after removing staffing effects (pure execution)."
        elif col == "number_of_checks":
            desc = "Number of checks in this store / daypart / order channel group."
        elif col == "total_sales_dollars":
            desc = "Total sales dollars in this store / daypart / order channel group."
        elif col == "hotline_sla_rate_fraction":
            desc = "Fraction of checks in this group that met the hotline SLA."
        elif col == "drive_thru_sla_rate_fraction":
            desc = "Fraction of checks in this group that met the drive-thru SLA."
        elif col == "labor_total":
            desc = "Total labor hours on the clock during the check."
        elif col == "labor_FOH":
            desc = "Front-of-house labor hours on the clock during the check."
        elif col == "labor_BOH":
            desc = "Back-of-house labor hours on the clock during the check."
        elif col == "CPLH_total":
            desc = "Total covers (or checks) per labor hour for the whole restaurant."
        elif col == "CPLH_BOH":
            desc = "Covers per labor hour for back-of-house."
        elif col == "CPLH_FOH":
            desc = "Covers per labor hour for front-of-house."
        elif col == "LLI_total":
            desc = "Total labor load index (how stretched total labor is)."
        elif col == "LLI_BOH":
            desc = "Back-of-house labor load index."
        elif col == "labor_per_check_total":
            desc = "Total labor hours divided by number of checks (labor per check)."
        elif col == "labor_per_check_BOH":
            desc = "Back-of-house labor hours per check."
        elif col == "labor_per_check_FOH":
            desc = "Front-of-house labor hours per check."
        elif col == "OrderChannel_enc":
            desc = "Numeric code representing the order channel (dine-in, drive-thru, delivery, etc.)."
        elif col == "DayPartNorm_enc":
            desc = "Numeric code representing normalized daypart (lunch, dinner, off-peak)."
        else:
            desc = f"Engineered feature or metric '{col}'."
        rows.append((col, desc))
    glossary_df = pd.DataFrame(rows, columns=["column_name", "description"])
    glossary_df.to_csv(output_path, index=False)
    print(f"Full column glossary written to: {output_path}")


###############################################################################
# 6. SINGLE main: only V4-style outputs
###############################################################################

def main(data_dir=".", sample_frac=0.5):
    """Clean pipeline that ONLY writes 4 outputs:

      1) per_check_execution_staffing_enriched_v4_clean.csv
      2) store_execution_staffing_summary_v4_clean.csv
      3) full_feature_correlation_matrix_v4_clean.csv
      4) column_name_glossary_v4_clean.csv
    """
    print(f"[CLEAN V4 PIPELINE] Loading data from: {data_dir}")
    df_raw = load_all_store_files(data_dir=data_dir)
    print(f"Loaded {len(df_raw)} rows from all stores.")

    print("Engineering features (OrderChannel, DayPartNorm, labor metrics)...")
    df_fe, enc = engineer_features(df_raw)

    print("Cleaning station & total times (winsorizing outliers)...")
    df_clean = clean_station_outliers(df_fe, quantile_low=0.005, quantile_high=0.995)

    # Targets: total + any station columns present
    target_cols = ["total_check_time_sec"]
    for col in STATION_COLS:
        if col in df_clean.columns:
            target_cols.append(col)

    print("Fitting execution + staffing models for targets:", target_cols)
    (
        df_enriched,
        baseline_models,
        staffing_models,
        baseline_features,
        staffing_features,
        metrics_rows,
    ) = fit_execution_and_staffing_multi(
        df_clean,
        target_cols=target_cols,
        sample_frac=sample_frac,
        random_state=42,
    )

    # Full correlation matrix
    corr_path = os.path.join(data_dir, "full_feature_correlation_matrix_v4_clean.csv")
    compute_full_feature_correlation(df_clean, output_path=corr_path)

    # Store summary (not writing *_raw)
    store_summary = summarize_store_execution_and_staffing_multi(
        df_enriched,
        target_cols=target_cols,
        min_checks_per_group=1,
    )

    # Human-readable
    df_enriched_readable = rename_columns_readable_per_check(df_enriched, inplace=True)
    # df_enriched_readable is df_enriched (renamed in-place)
    store_summary_readable = rename_columns_readable_store_summary(
        store_summary, target_cols=target_cols, inplace=True
    )

    # --- New percentage metrics ---
    # 1) Share of store orders and sales for this row
    if "StoreID" in store_summary_readable.columns and        "number_of_checks" in store_summary_readable.columns and        "total_sales_dollars" in store_summary_readable.columns:
        store_totals = (
            store_summary_readable
            .groupby(["StoreID", "dateofbusiness"])[["number_of_checks", "total_sales_dollars"]]
            .sum()
            .rename(columns={
                "number_of_checks": "store_total_checks",
                "total_sales_dollars": "store_total_sales_dollars",
            })
            .reset_index()
        )
        store_summary_readable = store_summary_readable.merge(store_totals, on=["StoreID", "dateofbusiness"], how="left")
        store_summary_readable["percent_of_store_checks"] = (
            store_summary_readable["number_of_checks"] /
            store_summary_readable["store_total_checks"].replace({0: float("nan")})
        )
        store_summary_readable["percent_of_store_sales"] = (
            store_summary_readable["total_sales_dollars"] /
            store_summary_readable["store_total_sales_dollars"].replace({0: float("nan")})
        )
        store_summary_readable = store_summary_readable.drop(
            columns=["store_total_checks", "store_total_sales_dollars"]
        )

    # 2) Share of time difference attributable to staffing vs execution
    # Do this for total check time and for every station target that has a breakdown.
    for target_col in target_cols:
        base = target_col.replace("_sec", "")
        diff_col = f"average_{base}_difference_seconds"
        staff_col = f"average_{base}_staffing_effect_seconds"
        resid_col = f"average_{base}_execution_residual_seconds"
        if all(c in store_summary_readable.columns for c in [diff_col, staff_col, resid_col]):
            denom = store_summary_readable[diff_col].replace({0: float("nan")})
            staff_share_col = f"{base}_staffing_portion_of_difference"
            exec_share_col = f"{base}_execution_portion_of_difference"
            store_summary_readable[staff_share_col] = store_summary_readable[staff_col] / denom
            store_summary_readable[exec_share_col] = store_summary_readable[resid_col] / denom


    per_check_path = os.path.join(
        data_dir, "per_check_execution_staffing_enriched_v4_clean.csv"
    )
    df_enriched_readable.to_csv(per_check_path, index=False)
    print(f"Per-check execution/staffing enriched data written to: {per_check_path}")

    # Reorder columns for readability (group metrics) BEFORE writing summary CSVs
    store_summary_readable = reorder_store_summary_columns(store_summary_readable, target_cols)

    sdc_path = os.path.join(
        data_dir, "store_execution_staffing_summary_v4_clean.csv"
    )
    store_summary_readable.to_csv(sdc_path, index=False)
    print(f"Store execution vs staffing summary written to: {sdc_path}")

    glossary_path = os.path.join(
        data_dir, "column_name_glossary_v4_clean.csv"
    )
    # Percentages-only summary CSV
    base_cols = ["StoreID", "DayPartNorm", "OrderChannel", "dateofbusiness", "number_of_checks", "total_sales_dollars"]
    pct_cols = base_cols.copy()
    # Include all columns that look like percentage metrics:
    # - percent_of_store_checks / percent_of_store_sales
    # - any *_difference_percent
    # - any *_portion_of_difference
    for col in store_summary_readable.columns:
        if (
            "percent" in col
            or col.endswith("_portion_of_difference")
        ):
            if col not in pct_cols:
                pct_cols.append(col)
    pct_df = store_summary_readable[pct_cols].copy()
    pct_path = os.path.join(
        data_dir, "store_execution_staffing_percentages_v4_clean.csv"
    )
    pct_df.to_csv(pct_path, index=False)
    print(f"Store percentages summary written to: {pct_path}")

    build_column_glossary(df_enriched_readable, store_summary_readable, glossary_path)

print("[CLEAN V4 PIPELINE] Done.")


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    sample_frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    main(data_dir=data_dir, sample_frac=sample_frac)