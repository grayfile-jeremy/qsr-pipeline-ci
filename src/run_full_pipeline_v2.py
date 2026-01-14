#!/usr/bin/env python3
"""run_full_pipeline_v2.py

High-level orchestrator to run the full feature engineering pipeline for:
- multiple stores
- multiple dates OR a continuous date range (e.g. 18 months)
- optional contiguous ranges of stores (e.g. 07-0001 through 07-0120)

Steps per run:
1. For each (store, date) combination:
   - Call enrich_sos_daily_with_labor_globaljobs_stations.py
   - Call domo_query_std_filters.py for keymetrics
2. Concatenate all per-check outputs:
   - Write one per-check CSV per store: per_check_store_<store>.csv
   - Write one combined per-check CSV for all stores/dates (feeds aggregator)
3. Call aggregate_store_day_ml.py (or v2) on the combined per-check data
4. Merge keymetrics into aggregated store-day file

This script does NOT change your feature logic; it just wires the existing
scripts together and lets you scale across many stores/dates.

Usage examples
--------------

Single store, single day:

  python run_full_pipeline_v2.py \
    --stores 07-0002 \
    --dates 3/2/2022 \
    --sos-dataset-id d4b5fc98-182d-412c-8e39-e727b9ede64e \
    --checks-dataset-id bbb4d0e1-7b08-4fc2-a86e-d8d61cdd7a41 \
    --labor-dataset-id 7c9bfd01-ba4a-4aec-ba2a-6233bcd7e8f8 \
    --jobcodes-file jobcodes_master.csv \
    --keymetrics-dataset-id 2c8c9f0f-5eb8-4737-9b1f-0c67acb845dd

Single store, many days (explicit list):

  python run_full_pipeline_v2.py \
    --stores 07-0002 \
    --dates 3/1/2022,3/2/2022,3/3/2022 \
    ...

Single store, continuous date range (e.g., 18 months):

  python run_full_pipeline_v2.py \
    --stores 07-0002 \
    --date-range-start 2022-01-01 \
    --date-range-end   2023-06-30 \
    ...

Multiple stores, continuous date range:

  python run_full_pipeline_v2.py \
    --stores-range-start 07-0001 \
    --stores-range-end   07-0120 \
    --date-range-start 2022-01-01 \
    --date-range-end   2023-06-30 \
    ...

You can also mix explicit stores + a range; the script will deduplicate.
"""

import argparse
import os
import sys
import subprocess
from datetime import timedelta

import pandas as pd


# ----------------------------------------------------------------------
# Helpers to match your existing merge_keymetrics_daily behavior
# ----------------------------------------------------------------------
def pick_column(df, candidates, label, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise SystemExit(
            f"Could not find any of {candidates} for {label}. Columns: {list(df.columns)}"
        )
    return None


def normalize_store(series):
    return series.astype(str).str.strip()


def normalize_date(series):
    return pd.to_datetime(series).dt.date


def safe_date_for_filename(dob: str) -> str:
    try:
        dt = pd.to_datetime(dob)
        return dt.date().isoformat()
    except Exception:
        return dob.replace("/", "-").replace(" ", "_")  # best-effort


def expand_date_range(start_str: str, end_str: str):
    """Return list of date strings between start and end inclusive.

    Accepts flexible input formats (YYYY-MM-DD, m/d/YYYY, etc.).
    Returns dates formatted as m/d/YYYY to match your existing Domo filter usage.
    """
    start = pd.to_datetime(start_str).date()
    end = pd.to_datetime(end_str).date()
    if end < start:
        raise SystemExit(f"date-range-end {end_str} is before date-range-start {start_str}")
    dates = []
    cur = start
    while cur <= end:
        dates.append(f"{cur.month}/{cur.day}/{cur.year}")  # m/d/YYYY
        cur += timedelta(days=1)
    return dates


def expand_store_range(start_store: str, end_store: str):
    """Expand a contiguous range of store IDs like 07-0001..07-0120.

    Assumes:
      - Both stores contain a '-' separator.
      - Prefix (before last '-') is identical.
      - Suffix (after last '-') is zero-padded numeric.

    Example:
      start_store = '07-0001'
      end_store   = '07-0003'
      -> ['07-0001','07-0002','07-0003']
    """
    try:
        start_prefix, start_suffix = start_store.rsplit("-", 1)
        end_prefix, end_suffix = end_store.rsplit("-", 1)
    except ValueError:
        raise SystemExit(
            f"Store range must look like '07-0001'..'07-0120' (got '{start_store}' and '{end_store}')"
        )

    if start_prefix != end_prefix:
        raise SystemExit(
            f"Store range prefixes differ: '{start_prefix}' vs '{end_prefix}'. "
            f"Range expansion is only supported when the prefix is the same."
        )

    try:
        start_num = int(start_suffix)
        end_num = int(end_suffix)
    except ValueError:
        raise SystemExit(
            f"Store suffix must be numeric for range expansion (got '{start_suffix}' and '{end_suffix}')."
        )

    if end_num < start_num:
        raise SystemExit(
            f"stores-range-end {end_store} is before stores-range-start {start_store}"
        )

    width = max(len(start_suffix), len(end_suffix))
    stores = [
        f"{start_prefix}-{str(i).zfill(width)}"
        for i in range(start_num, end_num + 1)
    ]
    return stores


# ----------------------------------------------------------------------
# Per-check enrichment for one (store, date)
# ----------------------------------------------------------------------
def run_enrich(store, dob, sos_dataset_id, checks_dataset_id, labor_dataset_id, jobcodes_file):
    """Run enrich_sos_daily_with_labor_globaljobs_stations.py for one store/date.

    Returns (df, output_csv_path).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    enrich_script = os.path.join(script_dir, "enrich_sos_daily_with_labor_globaljobs_stations.py")

    safe_date = safe_date_for_filename(dob)
    out_csv = f"sos_with_checks_labor_stations_{store}_{safe_date}.csv"

    cmd = [
        sys.executable,
        enrich_script,
        "--store", store,
        "--date", dob,
        "--sos-dataset-id", sos_dataset_id,
        "--checks-dataset-id", checks_dataset_id,
        "--labor-dataset-id", labor_dataset_id,
        "--jobcodes-file", jobcodes_file,
        "--out", out_csv,
    ]

    print("\n[ENRICH]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if not os.path.exists(out_csv):
        print(f"[WARN] Expected per-check output not found for {store} {dob}: {out_csv}")
        return pd.DataFrame(), out_csv

    df = pd.read_csv(out_csv)
    if df.empty:
        print(f"[INFO] Empty per-check output for {store} {dob}.")
    else:
        print(f"[INFO] Loaded {len(df)} per-check rows for store={store}, date={dob}")
    return df, out_csv


# ----------------------------------------------------------------------
# Key metrics pull for one (store, date)
# ----------------------------------------------------------------------
def run_keymetrics(store, dob, keymetrics_dataset_id):
    """Run domo_query_std_filters.py to fetch key metrics for one store/date.

    Returns (df, output_csv_path).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    domo_script = os.path.join(script_dir, "domo_query_std_filters.py")

    safe_date = safe_date_for_filename(dob)
    out_csv = f"keymetrics_{store}_{safe_date}.csv"

    cmd = [
        sys.executable,
        domo_script,
        "--dataset-id", keymetrics_dataset_id,
        "--out", out_csv,
        "--filter", f"Store# is {store}",
        "--filter", f"date is {dob}",
    ]

    print("\n[KEYMETRICS]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if not os.path.exists(out_csv):
        print(f"[WARN] Expected keymetrics output not found for {store} {dob}: {out_csv}")
        return pd.DataFrame(), out_csv

    df = pd.read_csv(out_csv)
    if df.empty:
        print(f"[INFO] Empty keymetrics for {store} {dob}.")
    else:
        print(f"[INFO] Loaded {len(df)} keymetrics rows for store={store}, date={dob}")
    return df, out_csv


# ----------------------------------------------------------------------
# Merge aggregated store-day features with key metrics (in-memory)
# ----------------------------------------------------------------------
def merge_keymetrics(agg_df: pd.DataFrame, km_df: pd.DataFrame) -> pd.DataFrame:
    """Merge a per-store-per-day aggregated DF with a keymetrics DF.

    Mirrors the behavior of merge_keymetrics_daily.py:
    - Detects store/date columns on each side
    - Normalizes store as string, date as datetime.date
    - Deduplicates keymetrics rows per (store, date)
    - Renames conflicting keymetrics columns with _km suffix
    - Left-joins onto aggregated data
    """

    if agg_df.empty:
        print("[WARN] Aggregated DF is empty; nothing to merge.")
        return agg_df

    if km_df.empty:
        print("[WARN] Keymetrics DF is empty; returning agg_df unchanged.")
        return agg_df

    agg_df = agg_df.copy()
    km = km_df.copy()

    agg_store_col = pick_column(
        agg_df, ["store#", "Store#", "Store", "store", "StoreNum", "storenum"], "agg store"
    )
    agg_date_col = pick_column(
        agg_df,
        ["date", "Date", "DateOfBusiness", "dateofbusiness", "BusinessDate"],
        "agg date",
    )

    km_store_col = pick_column(
        km, ["Store#", "store#", "Store", "store", "StoreNum", "storenum"], "keymetrics store"
    )
    km_date_col = pick_column(
        km,
        ["date", "Date", "DateOfBusiness", "dateofbusiness", "BusinessDate"],
        "keymetrics date",
    )

    agg_df["_StoreKey"] = normalize_store(agg_df[agg_store_col])
    agg_df["_DateKey"] = normalize_date(agg_df[agg_date_col])

    km["_StoreKey"] = normalize_store(km[km_store_col])
    km["_DateKey"] = normalize_date(km[km_date_col])

    km_dedup = km.drop_duplicates(subset=["_StoreKey", "_DateKey"])

    km_cols = [
        c
        for c in km_dedup.columns
        if c not in [km_store_col, km_date_col, "_StoreKey", "_DateKey"]
    ]
    rename_map = {}
    for c in km_cols:
        if c in agg_df.columns:
            rename_map[c] = c + "_km"
    if rename_map:
        km_dedup = km_dedup.rename(columns=rename_map)
        km_cols = [rename_map.get(c, c) for c in km_cols]

    merged = agg_df.merge(
        km_dedup[["_StoreKey", "_DateKey"] + km_cols],
        on=["_StoreKey", "_DateKey"],
        how="left",
    )

    print(
        f"[INFO] Merged key metrics: {len(merged)} rows, added {len(km_cols)} key-metric columns."
    )
    return merged


# ----------------------------------------------------------------------
# Main CLI
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=(
            "Run per-check enrichment for many stores/dates, write per-store per-check CSVs, "
            "aggregate to store-day using aggregate_store_day_ml, and merge keymetrics."
        )
    )

    # Stores can come from:
    #   --stores (comma-separated list)
    #   and/or
    #   --stores-range-start + --stores-range-end (contiguous range like 07-0001..07-0120)
    ap.add_argument(
        "--stores",
        help="Comma-separated store IDs, e.g. 07-0002,07-0003. Optional if using a store range.",
    )
    ap.add_argument(
        "--stores-range-start",
        help="Start of a contiguous store range, e.g. 07-0001.",
    )
    ap.add_argument(
        "--stores-range-end",
        help="End of a contiguous store range, e.g. 07-0120.",
    )

    # Either --dates OR (--date-range-start and --date-range-end)
    ap.add_argument(
        "--dates",
        help="Comma-separated list of business dates (e.g. 3/1/2022,3/2/2022). "
             "If omitted, you must supply --date-range-start and --date-range-end.",
    )
    ap.add_argument(
        "--date-range-start",
        help="Start of continuous date range (YYYY-MM-DD or m/d/YYYY). Inclusive.",
    )
    ap.add_argument(
        "--date-range-end",
        help="End of continuous date range (YYYY-MM-DD or m/d/YYYY). Inclusive.",
    )

    ap.add_argument("--sos-dataset-id", required=True)
    ap.add_argument("--checks-dataset-id", required=True)
    ap.add_argument("--labor-dataset-id", required=True)
    ap.add_argument("--jobcodes-file", required=True)
    ap.add_argument("--keymetrics-dataset-id", required=True)

    ap.add_argument(
        "--agg-out",
        default="store_day_aggregated_ml.csv",
        help="Output CSV path for aggregated per-store-per-day features.",
    )
    ap.add_argument(
        "--final-out",
        default="store_day_with_keymetrics.csv",
        help="Final output CSV path (aggregated + keymetrics).",
    )
    ap.add_argument(
        "--per-check-all-out",
        default="per_check_all.csv",
        help="Output path for the combined per-check dataset that feeds aggregate_store_day_ml.",
    )
    ap.add_argument(
        "--agg-script-name",
        default="aggregate_store_day_ml.py",
        help="Name of the aggregation script to call (e.g., aggregate_store_day_ml.py or aggregate_store_day_ml_v2.py).",
    )

    args = ap.parse_args()

    # ----- Build store list -----
    store_set = []

    if args.stores:
        store_set.extend([s.strip() for s in args.stores.split(",") if s.strip()])

    if args.stores_range_start and args.stores_range_end:
        expanded = expand_store_range(args.stores_range_start.strip(), args.stores_range_end.strip())
        store_set.extend(expanded)

    # Deduplicate while preserving order
    seen_stores = set()
    stores = []
    for s in store_set:
        if s not in seen_stores:
            seen_stores.add(s)
            stores.append(s)

    if not stores:
        raise SystemExit(
            "No stores provided. Use --stores and/or --stores-range-start/--stores-range-end."
        )

    # ----- Build date list -----
    date_list = []
    if args.dates:
        date_list.extend([d.strip() for d in args.dates.split(",") if d.strip()])

    if args.date_range_start and args.date_range_end:
        range_dates = expand_date_range(args.date_range_start, args.date_range_end)
        date_list.extend(range_dates)

    seen_dates = set()
    dates = []
    for d in date_list:
        if d not in seen_dates:
            seen_dates.add(d)
            dates.append(d)

    if not dates:
        raise SystemExit(
            "No dates provided. Use --dates or --date-range-start + --date-range-end."
        )

    print(f"[INFO] Stores: {stores[:10]}{' ...' if len(stores) > 10 else ''}")
    print(f"[INFO] Total store count: {len(stores)}")
    print(f"[INFO] Dates:  {dates[:5]}{' ...' if len(dates) > 5 else ''}")
    print(f"[INFO] Total date count: {len(dates)}")
    print("[INFO] Will run all combinations (store x date).")

    # ----- Run the pipeline -----
    all_per_check = []
    all_km = []
    per_store_frames = {s: [] for s in stores}

    # 1) Run per-check enrichment + keymetrics pulls for each store/date
    for store in stores:
        for dob in dates:
            df, percheck_csv = run_enrich(
                store=store,
                dob=dob,
                sos_dataset_id=args.sos_dataset_id,
                checks_dataset_id=args.checks_dataset_id,
                labor_dataset_id=args.labor_dataset_id,
                jobcodes_file=args.jobcodes_file,
            )
            if not df.empty:
                all_per_check.append(df)
                per_store_frames[store].append(df)

            km_df, km_csv = run_keymetrics(
                store=store,
                dob=dob,
                keymetrics_dataset_id=args.keymetrics_dataset_id,
            )
            if not km_df.empty:
                all_km.append(km_df)

            # Clean up the per-store-per-date CSVs after loading
            if os.path.exists(percheck_csv):
                os.remove(percheck_csv)
            if os.path.exists(km_csv):
                os.remove(km_csv)

    if not all_per_check:
        raise SystemExit("No per-check data produced for any store/date.")

    # 2) Write per-store per-check CSVs
    for store, frames in per_store_frames.items():
        if not frames:
            continue
        per_store_df = pd.concat(frames, ignore_index=True)
        per_store_path = f"per_check_store_{store}.csv"
        per_store_df.to_csv(per_store_path, index=False)
        print(f"[DONE] Wrote per-check CSV for store {store}: {per_store_path}")

    # 3) Write combined per-check dataset that feeds aggregate_store_day_ml
    per_check_all = pd.concat(all_per_check, ignore_index=True)
    per_check_all.to_csv(args.per_check_all_out, index=False)
    print(f"[DONE] Wrote combined per-check dataset to {args.per_check_all_out}")

    # 4) Call aggregation script on the combined per-check file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    agg_script = os.path.join(script_dir, args.agg_script_name)

    agg_cmd = [
        sys.executable,
        agg_script,
        "--input", args.per_check_all_out,
        "--out", args.agg_out,
    ]
    print("\n[AGGREGATE]", " ".join(agg_cmd))
    subprocess.run(agg_cmd, check=True)

    if not os.path.exists(args.agg_out):
        raise SystemExit(f"Expected aggregated output not found: {args.agg_out}")

    agg_df = pd.read_csv(args.agg_out)
    print(f"[INFO] Loaded aggregated store-day features: {len(agg_df)} rows")

    # 5) Merge keymetrics in-memory
    if all_km:
        km_all = pd.concat(all_km, ignore_index=True)
        final_df = merge_keymetrics(agg_df, km_all)
    else:
        print("[WARN] No keymetrics rows gathered; final file will match agg-out.")
        final_df = agg_df

    final_df.to_csv(args.final_out, index=False)
    print(f"[DONE] Wrote final store-day + keymetrics file to {args.final_out}")


if __name__ == "__main__":
    main()
