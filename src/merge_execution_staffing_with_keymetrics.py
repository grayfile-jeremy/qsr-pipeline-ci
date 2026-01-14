#!/usr/bin/env python3
"""
merge_execution_staffing_with_keymetrics.py

Pull daily keymetrics from Domo and merge them onto the
execution/staffing daily summary (by store / date / daypart / channel)
so you get one wide dataframe ready for ML.

Inputs:
  - daily execution/staffing CSV (output from restaurant_performance_execution_staffing_v4_pct5_daily_v2.py)
  - Domo dataset ID for keymetrics

The script:
  1) Infers store IDs and date range from the daily CSV.
  2) Uses domo_query_std_filters.py helpers to pull the matching
     keymetrics rows in one bulk query.
  3) Normalizes store/date keys on both sides and merges keymetrics
     onto every (store, date, DayPartNorm, OrderChannel) row.

Auth:
  - Expects DOMO_CLIENT_ID and DOMO_CLIENT_SECRET to be set in env,
    same as domo_query_std_filters.py.
"""

import argparse
import os
import pandas as pd
from datetime import datetime

# We reuse helpers from your existing Domo utility
# domo_query_std_filters.py
from domo_query_std_filters import build_sql, query_to_csv  # type: ignore


# --------------------------
# Local helpers (adapted from merge_keymetrics_daily)
# --------------------------
def pick_column(df, candidates, label, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise SystemExit(f"Could not find any of {candidates} for {label}. Columns: {list(df.columns)}")
    return None


def normalize_store(series):
    return series.astype(str).str.strip()


def normalize_date(series):
    return pd.to_datetime(series).dt.date


def domo_date_for_between(date_str: str) -> str:
    """
    Turn a YYYY-MM-DD or m/d/YYYY string into m/d/YYYY for Domo filters.
    """
    dt = pd.to_datetime(date_str).date()
    return f"{dt.month}/{dt.day}/{dt.year}"


def pull_keymetrics_for_daily(
    daily_df: pd.DataFrame,
    keymetrics_dataset_id: str,
    out_csv: str = "keymetrics_for_daily.csv",
) -> pd.DataFrame:
    """
    Infer store list and date range from the daily execution/staffing
    dataframe and pull the matching keymetrics via the Domo Query API.
    """
    if daily_df.empty:
        raise SystemExit("Daily execution/staffing dataframe is empty; nothing to pull.")

    # Identify store/date columns in the daily file
    store_col = pick_column(
        daily_df,
        ["StoreID", "store#", "Store#", "Store", "store", "StoreNum", "storenum"],
        "daily store",
    )
    date_col = pick_column(
        daily_df,
        ["dateofbusiness", "DateOfBusiness", "date", "Date", "BusinessDate"],
        "daily date",
    )

    # Infer store list + date range
    stores = sorted(normalize_store(daily_df[store_col]).dropna().unique().tolist())
    if not stores:
        raise SystemExit("Could not infer any store IDs from the daily CSV.")

    dates = normalize_date(daily_df[date_col])
    date_min = dates.min()
    date_max = dates.max()
    if pd.isna(date_min) or pd.isna(date_max):
        raise SystemExit("Could not infer a valid date range from the daily CSV.")

    start_domodate = domo_date_for_between(str(date_min))
    end_domodate = domo_date_for_between(str(date_max))

    store_list = ",".join(stores)

    # Build filters consistent with your existing keymetrics pulls
    filters = [
        f"Store# is {store_list}",
        f"date between {start_domodate}..{end_domodate}",
    ]

    # Build SQL and run query
    sql = build_sql(select_cols=None, filters=filters, order_by=None, limit=None, offset=None)

    bytes_written = query_to_csv(
        dataset_id=keymetrics_dataset_id,
        sql=sql,
        out_path=out_csv,
        accept="application/json",
    )
    if bytes_written <= 0 or not os.path.exists(out_csv):
        print("[WARN] No keymetrics bytes written; returning empty DataFrame.")
        return pd.DataFrame()

    km_df = pd.read_csv(out_csv)
    return km_df


def merge_daily_with_keymetrics(daily_df: pd.DataFrame, km_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge execution/staffing daily summary with keymetrics on
    store + date. Keymetrics are at a store-day grain, so they
    will be duplicated across DayPartNorm/OrderChannel rows for
    that store-day.
    """
    if daily_df.empty:
        print("[WARN] Daily execution/staffing DF is empty; nothing to merge.")
        return daily_df

    if km_df.empty:
        print("[WARN] Keymetrics DF is empty; returning daily_df unchanged.")
        return daily_df

    daily_df = daily_df.copy()
    km = km_df.copy()

    # Detect store/date columns on both sides
    daily_store_col = pick_column(
        daily_df, ["StoreID", "store#", "Store#", "Store", "store", "StoreNum", "storenum"], "daily store"
    )
    daily_date_col = pick_column(
        daily_df,
        ["dateofbusiness", "DateOfBusiness", "date", "Date", "BusinessDate"],
        "daily date",
    )

    km_store_col = pick_column(
        km, ["Store#", "store#", "Store", "store", "StoreNum", "storenum"], "keymetrics store"
    )
    km_date_col = pick_column(
        km,
        ["date", "Date", "DateOfBusiness", "dateofbusiness", "BusinessDate"],
        "keymetrics date",
    )

    # Normalize join keys
    daily_df["_StoreKey"] = normalize_store(daily_df[daily_store_col])
    daily_df["_DateKey"] = normalize_date(daily_df[daily_date_col])

    km["_StoreKey"] = normalize_store(km[km_store_col])
    km["_DateKey"] = normalize_date(km[km_date_col])

    # Deduplicate keymetrics to one row per store/date
    km_dedup = km.drop_duplicates(subset=["_StoreKey", "_DateKey"])

    # Avoid column collisions (besides keys)
    km_cols = [c for c in km_dedup.columns if c not in [km_store_col, km_date_col, "_StoreKey", "_DateKey"]]
    rename_map = {}
    for c in km_cols:
        if c in daily_df.columns:
            rename_map[c] = c + "_km"
    if rename_map:
        km_dedup = km_dedup.rename(columns=rename_map)
        km_cols = [rename_map.get(c, c) for c in km_cols]

    merged = daily_df.merge(
        km_dedup[["_StoreKey", "_DateKey"] + km_cols],
        on=["_StoreKey", "_DateKey"],
        how="left",
    )

    return merged


def main():
    ap = argparse.ArgumentParser(
        description="Merge execution/staffing daily summary with Domo daily keymetrics for ML."
    )
    ap.add_argument(
        "--daily-input",
        required=True,
        help="CSV from restaurant_performance_execution_staffing_v4_pct5_daily_v2.py (per store/day/daypart/channel).",
    )
    ap.add_argument(
        "--keymetrics-dataset-id",
        required=True,
        help="Domo dataset GUID for the daily keymetrics dataset.",
    )
    ap.add_argument(
        "--out",
        default="daily_execution_with_keymetrics.csv",
        help="Output CSV path.",
    )
    ap.add_argument(
        "--keymetrics-csv-cache",
        default="keymetrics_for_daily.csv",
        help="Intermediate CSV path for the raw keymetrics pull.",
    )

    args = ap.parse_args()

    if not os.path.exists(args.daily_input):
        raise SystemExit(f"Daily input file not found: {args.daily_input}")

    daily_df = pd.read_csv(args.daily_input)

    print(f"[INFO] Loaded daily execution/staffing summary: {len(daily_df)} rows")

    km_df = pull_keymetrics_for_daily(
        daily_df,
        keymetrics_dataset_id=args.keymetrics_dataset_id,
        out_csv=args.keymetrics_csv_cache,
    )
    print(f"[INFO] Loaded keymetrics rows from Domo: {len(km_df)} rows")

    merged = merge_daily_with_keymetrics(daily_df, km_df)
    print(f"[INFO] Final merged dataset has {len(merged)} rows and {merged.shape[1]} columns")

    merged.to_csv(args.out, index=False)
    print(f"[DONE] Wrote merged daily execution + keymetrics file to {args.out}")


if __name__ == "__main__":
    main()
