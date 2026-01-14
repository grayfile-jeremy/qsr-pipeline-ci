#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import re

def pick_column(df, candidates, label, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise SystemExit(f"Could not find any of {candidates} for {label}. Columns: {list(df.columns)}")
    return None

def safe_suffix(s):
    s = str(s).strip()
    if not s:
        return "Unknown"
    s = re.sub(r"[–-]", "_", s)
    s = re.sub(r"[^0-9A-Za-z_]+", "", s)
    return s or "Unknown"

def classify_channel(val):
    if pd.isna(val):
        return "OTHER"
    s = str(val).upper()
    if "DRIVE" in s or "DT" in s or "DRIVE THRU" in s:
        return "DT"
    if (
        "DINE" in s
        or "CARRY" in s
        or "CARRYOUT" in s
        or "CARRY OUT" in s
        or "TO GO" in s
        or "PICKUP" in s
        or "DELIVERY" in s
    ):
        return "DICO"
    return "OTHER"

DT_METRICS = [
    "registers_in_use_dt",
    "registers_in_use_dt_others",
    "dt_cars_in_lane",
    "dt_cars_in_lane_others",
]

DICO_METRICS = [
    "registers_in_use_dico",
    "registers_in_use_dico_others",
]

def aggregate_daily_features(df, jobcode_bucket_df=None, store_col=None, date_col=None, out_path="store_day_aggregated_ml.csv"):
    # Identify store/date/daypart/channel columns
    if store_col is None:
        store_col = pick_column(df, ["store#","Store#","StoreNum","storenum","Store","store"], "store")
    if date_col is None:
        date_col = pick_column(df, ["DateOfBusiness","dateofbusiness","BusinessDate","date","Date"], "date")

    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    daypart_col = pick_column(df, ["Day Part","DayPart","daypart"], "daypart", required=False)
    channel_col = pick_column(
        df,
        [
            "Channel",
            "Channel Description",
            "OrderChannel",
            "Service Type",
            "ServiceType",
            "OrderType",
            "NCROrderType",
        ],
        "channel",
        required=False,
    )

    if channel_col is not None:
        df["_channel_cat"] = df[channel_col].map(classify_channel)
    else:
        df["_channel_cat"] = "OTHER"

    # Labor columns as created by enrichment (one per JobCodeDesc)
    labor_cols = [c for c in df.columns if c.startswith("labor_")]

    # Build mapping from column name -> FOH/BOH/MGR/OTHER using jobcode_bucket_df
    labor_FOH_cols, labor_BOH_cols, labor_MGR_cols, labor_OTHER_cols = [], [], [], []
    if jobcode_bucket_df is not None and not jobcode_bucket_df.empty:
        bucket_map = dict(zip(jobcode_bucket_df["JobCodeDesc"], jobcode_bucket_df["bucket"]))
        for col in labor_cols:
            # col format: labor_<JobCodeDesc with spaces replaced by _>
            raw = col[len("labor_"):]
            job_desc = raw.replace("_", " ")
            bucket = bucket_map.get(job_desc, "OTHER")
            if bucket == "FOH":
                labor_FOH_cols.append(col)
            elif bucket == "BOH":
                labor_BOH_cols.append(col)
            elif bucket == "MGR":
                labor_MGR_cols.append(col)
            else:
                labor_OTHER_cols.append(col)
    else:
        # Fall back: if no mapping provided, treat all as OTHER to avoid silent misclassification
        labor_OTHER_cols = labor_cols

    if labor_cols:
        df["labor_total"] = df[labor_cols].sum(axis=1)
        df["labor_FOH"] = df[labor_FOH_cols].sum(axis=1) if labor_FOH_cols else 0
        df["labor_BOH"] = df[labor_BOH_cols].sum(axis=1) if labor_BOH_cols else 0
        df["labor_MGR"] = df[labor_MGR_cols].sum(axis=1) if labor_MGR_cols else 0
        df["labor_OTHER"] = df[labor_OTHER_cols].sum(axis=1) if labor_OTHER_cols else 0
    else:
        df["labor_total"] = 0
        df["labor_FOH"] = 0
        df["labor_BOH"] = 0
        df["labor_MGR"] = 0
        df["labor_OTHER"] = 0

    # For simplicity, reuse the existing numeric/binary/stress logic from v2
    # Identify numeric and binary columns
    def is_binary_series(s):
        vals = set(str(v).strip().upper() for v in s.dropna().unique())
        if not vals:
            return False
        allowed = {"0","1","Y","N","YES","NO","TRUE","FALSE"}
        return vals.issubset(allowed) and len(vals) > 1

    def to_binary_numeric(s):
        m = {
            "0": 0, "1": 1,
            "Y": 1, "N": 0,
            "YES": 1, "NO": 0,
            "TRUE": 1, "FALSE": 0
        }
        return s.astype(str).str.strip().str.upper().map(m).astype(float)

    binary_cols = []
    for c in df.columns:
        if c in [store_col, date_col]:
            continue
        if c.startswith("labor_"):
            continue
        if is_binary_series(df[c]):
            binary_cols.append(c)

    for c in binary_cols:
        df[c + "__num"] = to_binary_numeric(df[c])

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    bin_num_cols = [c + "__num" for c in binary_cols]
    id_like = {
        "wk","pd","fy","Weeks Open (Shifted)","Drive Thru Window(s)",
        "StoreKey","DateKey","CheckKey","period end date","week start","week ending"
    }
    numeric_kpi_cols = [
        c for c in numeric_cols
        if not c.startswith("labor_")
        and c not in bin_num_cols
        and c not in binary_cols
        and c not in id_like
    ]

    total_time_col = pick_column(df, ["Total Time","total time","TotalTime"], "Total Time", required=False)

    group_keys = [store_col, date_col]
    grouped = df.groupby(group_keys)

    def is_ticket_like(name: str) -> bool:
        lname = name.lower()
        return ("ticket" in lname) or ("trouble" in lname) or ("red" in lname) or ("expo early bump" in lname)

    dp_binary_cols = [c for c in binary_cols if is_ticket_like(c)]

    rows = []
    for (store, day), g in grouped:
        rec = {"store": store, "date": day}
        rec["n_checks"] = len(g)

        eps = 1e-6

        # Daily labor summaries
        for base in ["labor_total","labor_FOH","labor_BOH","labor_MGR","labor_OTHER"]:
            rec[f"avg_{base}"] = g[base].mean()
            rec[f"max_{base}"] = g[base].max()

        rec["ratio_FOH_BOH"] = rec["avg_labor_FOH"] / (rec["avg_labor_BOH"] + eps)
        rec["ratio_mgr_total"] = rec["avg_labor_MGR"] / (rec["avg_labor_total"] + eps)

        # Total Time stability metrics (whole day)
        if total_time_col is not None and total_time_col in g.columns:
            tt_all = g[total_time_col].dropna()
            if not tt_all.empty:
                rec["median_Total_Time"] = tt_all.median()
                p90_tt = float(np.percentile(tt_all, 90))
                rec["p90_Total_Time"] = p90_tt
                rec["tail_gap_Total_Time"] = p90_tt - rec["median_Total_Time"]

        # Binary KPIs (overall)
        for c in binary_cols:
            cnum = c + "__num"
            if cnum not in g.columns:
                continue
            rec[f"pct_{c}"] = g[cnum].mean()
            rec[f"count_{c}"] = g[cnum].sum()

        # Numeric KPIs (overall)
        for c in numeric_kpi_cols:
            series = g[c].dropna()
            if series.empty:
                continue
            rec[f"mean_{c}"] = series.mean()
            rec[f"p90_{c}"] = float(np.percentile(series, 90))

        # Tail-gap for dt_cars_in_lane whole-day
        if "dt_cars_in_lane" in g.columns:
            s_dt = g["dt_cars_in_lane"].dropna()
            if not s_dt.empty:
                m_dt = s_dt.mean()
                p90_dt = float(np.percentile(s_dt, 90))
                rec["mean_dt_cars_in_lane"] = m_dt
                rec["p90_dt_cars_in_lane"] = p90_dt
                rec["tail_gap_dt_cars_in_lane"] = p90_dt - m_dt

        mask_dt = g["_channel_cat"] == "DT"
        mask_dico = g["_channel_cat"] == "DICO"

        # DT / DICO-specific metrics (whole day)
        for col in DT_METRICS:
            if col not in g.columns:
                continue
            s = g.loc[mask_dt, col].dropna()
            if s.empty:
                continue
            rec[f"mean_{col}"] = s.mean()
            rec[f"p90_{col}"] = float(np.percentile(s, 90))

        for col in DICO_METRICS:
            if col not in g.columns:
                continue
            s = g.loc[mask_dico, col].dropna()
            if s.empty:
                continue
            rec[f"mean_{col}"] = s.mean()
            rec[f"p90_{col}"] = float(np.percentile(s, 90))

        # Daypart-specific metrics + DT/DICO + stress
        if daypart_col is not None:
            for dp in g[daypart_col].dropna().unique():
                sub = g[g[daypart_col] == dp]
                suf = safe_suffix(dp)
                n_dp = len(sub)
                rec[f"n_checks_dp_{suf}"] = n_dp

                # Channel mix by daypart
                if n_dp > 0:
                    dt_count = (sub["_channel_cat"] == "DT").sum()
                    dico_count = (sub["_channel_cat"] == "DICO").sum()
                    other_count = (sub["_channel_cat"] == "OTHER").sum()
                    rec[f"pct_DT_dp_{suf}"] = dt_count / n_dp
                    rec[f"pct_DICO_dp_{suf}"] = dico_count / n_dp
                    rec[f"pct_OTHER_dp_{suf}"] = other_count / n_dp
                else:
                    rec[f"pct_DT_dp_{suf}"] = np.nan
                    rec[f"pct_DICO_dp_{suf}"] = np.nan
                    rec[f"pct_OTHER_dp_{suf}"] = np.nan

                # Labor per daypart
                for base in ["labor_total","labor_FOH","labor_BOH","labor_MGR","labor_OTHER"]:
                    rec[f"avg_{base}_dp_{suf}"] = sub[base].mean() if n_dp > 0 else np.nan
                    rec[f"max_{base}_dp_{suf}"] = sub[base].max() if n_dp > 0 else np.nan

                avg_FOH_dp = rec.get(f"avg_labor_FOH_dp_{suf}", np.nan)
                avg_BOH_dp = rec.get(f"avg_labor_BOH_dp_{suf}", np.nan)
                avg_MGR_dp = rec.get(f"avg_labor_MGR_dp_{suf}", np.nan)
                avg_total_dp = rec.get(f"avg_labor_total_dp_{suf}", np.nan)

                if not np.isnan(avg_BOH_dp) and avg_BOH_dp + eps != 0:
                    rec[f"ratio_FOH_BOH_dp_{suf}"] = avg_FOH_dp / (avg_BOH_dp + eps)
                else:
                    rec[f"ratio_FOH_BOH_dp_{suf}"] = np.nan

                if not np.isnan(avg_total_dp) and avg_total_dp + eps != 0:
                    rec[f"ratio_MGR_total_dp_{suf}"] = avg_MGR_dp / (avg_total_dp + eps)
                else:
                    rec[f"ratio_MGR_total_dp_{suf}"] = np.nan

                if not np.isnan(avg_FOH_dp) and avg_FOH_dp + eps != 0:
                    rec[f"ratio_MGR_FOH_dp_{suf}"] = avg_MGR_dp / (avg_FOH_dp + eps)
                else:
                    rec[f"ratio_MGR_FOH_dp_{suf}"] = np.nan

                if not np.isnan(avg_BOH_dp) and avg_BOH_dp + eps != 0:
                    rec[f"ratio_MGR_BOH_dp_{suf}"] = avg_MGR_dp / (avg_BOH_dp + eps)
                else:
                    rec[f"ratio_MGR_BOH_dp_{suf}"] = np.nan

                if n_dp > 0:
                    rec[f"labor_total_per_check_dp_{suf}"] = avg_total_dp / n_dp
                    rec[f"labor_FOH_per_check_dp_{suf}"] = avg_FOH_dp / n_dp
                    rec[f"labor_BOH_per_check_dp_{suf}"] = avg_BOH_dp / n_dp
                    rec[f"labor_MGR_per_check_dp_{suf}"] = avg_MGR_dp / n_dp
                else:
                    rec[f"labor_total_per_check_dp_{suf}"] = np.nan
                    rec[f"labor_FOH_per_check_dp_{suf}"] = np.nan
                    rec[f"labor_BOH_per_check_dp_{suf}"] = np.nan
                    rec[f"labor_MGR_per_check_dp_{suf}"] = np.nan

                # Total Time daypart stats
                if total_time_col is not None and total_time_col in sub.columns:
                    tt = sub[total_time_col].dropna()
                    if not tt.empty:
                        mean_tt_dp = tt.mean()
                        p90_tt_dp = float(np.percentile(tt, 90))
                        median_tt_dp = tt.median()
                        rec[f"mean_{total_time_col}_dp_{suf}"] = mean_tt_dp
                        rec[f"p90_{total_time_col}_dp_{suf}"] = p90_tt_dp
                        rec[f"median_{total_time_col}_dp_{suf}"] = median_tt_dp
                        rec[f"tail_gap_{total_time_col}_dp_{suf}"] = p90_tt_dp - median_tt_dp

                # Ticket/problem flags per daypart
                for c in dp_binary_cols:
                    cnum = c + "__num"
                    if cnum not in sub.columns or n_dp == 0:
                        continue
                    rec[f"pct_{c}_dp_{suf}"] = sub[cnum].mean()
                    rec[f"count_{c}_dp_{suf}"] = sub[cnum].sum()

                # DT/DICO metrics within daypart
                sub_dt = sub[sub["_channel_cat"] == "DT"]
                sub_dico = sub[sub["_channel_cat"] == "DICO"]

                for col in DT_METRICS:
                    if col not in sub.columns:
                        continue
                    sdt = sub_dt[col].dropna()
                    if not sdt.empty:
                        mean_dt_dp = sdt.mean()
                        p90_dt_dp = float(np.percentile(sdt, 90))
                        rec[f"mean_{col}_dp_{suf}"] = mean_dt_dp
                        rec[f"p90_{col}_dp_{suf}"] = p90_dt_dp
                        rec[f"tail_gap_{col}_dp_{suf}"] = p90_dt_dp - mean_dt_dp

                for col in DICO_METRICS:
                    if col not in sub.columns:
                        continue
                    sdico = sub_dico[col].dropna()
                    if not sdico.empty:
                        mean_dico_dp = sdico.mean()
                        p90_dico_dp = float(np.percentile(sdico, 90))
                        rec[f"mean_{col}_dp_{suf}"] = mean_dico_dp
                        rec[f"p90_{col}_dp_{suf}"] = p90_dico_dp

                # Stress ratios per daypart
                dt_mean_cars = rec.get(f"mean_dt_cars_in_lane_dp_{suf}", np.nan)
                if not np.isnan(dt_mean_cars) and not np.isnan(avg_BOH_dp) and avg_BOH_dp + eps != 0:
                    rec[f"cars_per_BOH_dp_{suf}"] = dt_mean_cars / (avg_BOH_dp + eps)

                dico_mean_regs = rec.get(f"mean_registers_in_use_dico_dp_{suf}", np.nan)
                if not np.isnan(dico_mean_regs) and not np.isnan(avg_FOH_dp) and avg_FOH_dp + eps != 0:
                    rec[f"registers_per_FOH_dp_{suf}"] = dico_mean_regs / (avg_FOH_dp + eps)

                if not np.isnan(avg_BOH_dp) and avg_BOH_dp + eps != 0 and n_dp > 0:
                    rec[f"tickets_per_BOH_dp_{suf}"] = n_dp / (avg_BOH_dp + eps)

        rows.append(rec)

    out_df = pd.DataFrame(rows)
    out_df.sort_values(["store","date"], inplace=True)
    out_df.to_csv(out_path, index=False)
    print(f"[DONE] Wrote {len(out_df)} store-day rows to {out_path}")

def main():
    ap = argparse.ArgumentParser(
        description="Aggregate per-check enriched data into per-store-per-day ML features using explicit jobcode buckets."
    )
    ap.add_argument("--input", required=True, help="Input CSV of per-check data (enriched per-check).")
    ap.add_argument("--out", default="store_day_aggregated_ml.csv", help="Output CSV path.")
    ap.add_argument("--store-col", help="Override store column name.")
    ap.add_argument("--date-col", help="Override date column name.")
    ap.add_argument("--jobcode-buckets-file", default="jobcode_buckets.csv",
                    help="CSV mapping of JobCodeDesc to bucket (FOH/BOH/MGR/OTHER).")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)

    bucket_df = None
    if args.jobcode_buckets_file and os.path.exists(args.jobcode_buckets_file):
        bucket_df = pd.read_csv(args.jobcode_buckets_file)
    else:
        print(f"[WARN] jobcode buckets file not found: {args.jobcode_buckets_file}. All labor_* columns will be treated as OTHER.")

    aggregate_daily_features(df, jobcode_bucket_df=bucket_df, store_col=args.store_col, date_col=args.date_col, out_path=args.out)

if __name__ == "__main__":
    main()
