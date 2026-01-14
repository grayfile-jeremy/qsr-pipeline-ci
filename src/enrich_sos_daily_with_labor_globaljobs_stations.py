#!/usr/bin/env python3
import argparse
import os
import subprocess
import re
import pandas as pd
import numpy as np

def run_cmd(cmd_list):
    print("\n[CMD]", " ".join(cmd_list))
    result = subprocess.run(cmd_list)
    if result.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(cmd_list)}")

def pick_column(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(f"Missing column for {label}. Tried: {candidates}. Found: {list(df.columns)}")

def normalize(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.date.astype(str)
    return series.astype(str)

def time_to_minutes(s):
    """Convert a time-of-check column to minutes from midnight."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    s_str = s.astype(str).str.strip()
    has_colon = s_str.str.contains(":")
    dt = pd.to_datetime(s_str.where(has_colon, np.nan), errors="coerce")
    mask_unparsed = dt.isna() & s_str.notna() & (~has_colon)
    if mask_unparsed.any():
        dt2 = pd.to_datetime(s_str[mask_unparsed], errors="coerce")
        dt = dt.combine_first(dt2)
    return (dt.dt.hour * 60 + dt.dt.minute)

def slugify_job(job_desc):
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(job_desc).strip())
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "Unknown"
    return f"labor_{s}"

def _clip_seconds(series, upper=1800):
    if series is None:
        return pd.Series([np.nan])
    s = pd.to_numeric(series, errors="coerce")
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    return s.where((s > 0) & (s <= upper))

def _positive_diff(a, b, upper=1800):
    a = pd.to_numeric(pd.Series(a), errors="coerce")
    b = pd.to_numeric(pd.Series(b), errors="coerce")
    diff = a - b
    return diff.where((~a.isna()) & (~b.isna()) & (diff > 0) & (diff <= upper))

def add_station_time_and_concurrency_features(sos_df: pd.DataFrame) -> pd.DataFrame:
    """Add station-time, kitchen concurrency, register and DT car concurrency features."""
    if sos_df.empty:
        return sos_df

    col = lambda name: sos_df.get(name)

    total_raw = pd.to_numeric(col("Total Time"), errors="coerce")
    total_time = _clip_seconds(col("Total Time"), upper=3600)

    last_sent = _clip_seconds(col("Last Time Sent"))
    grill = _clip_seconds(col("Grill"))
    grill2 = _clip_seconds(col("Grill 2"))
    dt_grill = _clip_seconds(col("DT Grill"))
    dt_grill2 = _clip_seconds(col("DT Grill 2"))
    make = _clip_seconds(col("Make"))
    dt_make = _clip_seconds(col("DT Make"))
    expo = _clip_seconds(col("Expo"))
    dt_expo = _clip_seconds(col("DT Expo"))
    custard = _clip_seconds(col("Custard"))
    start_to_tender = _clip_seconds(col("Start to Tender"), upper=3600)
    close_clip = _clip_seconds(col("Close"), upper=3600)
    close_raw = pd.to_numeric(col("Close"), errors="coerce")
    tender_raw = pd.to_numeric(col("Tender"), errors="coerce")

    order_type_raw = sos_df.get("OrderType", "").astype(str).str.upper()
    is_dt = order_type_raw.str.contains("DRIVE")
    is_delivery = order_type_raw.str.contains("DELIVER")
    is_dico = (
        order_type_raw.str.contains("DINE") |
        order_type_raw.str.contains("CARRY") |
        order_type_raw.str.contains("DELIVERY")
    )

    perfectco_flag = False
    if "Make Live Date" in sos_df.columns:
        perfectco_flag = sos_df["Make Live Date"].notna()
    elif "Has Perfectco Make" in sos_df.columns:
        perfectco_flag = sos_df["Has Perfectco Make"].astype(str).str.upper().eq("Y")

    # ordering_ts is the "time ordering" / fully sent timestamp
    ordering_ts = np.where(
        is_dico & perfectco_flag,
        last_sent,
        np.where(
            is_dico,
            close_clip,
            last_sent
        )
    )
    ordering_ts = pd.to_numeric(pd.Series(ordering_ts), errors="coerce")

    chosen_grill2 = dt_grill2.combine(grill2, lambda a, b: a if not pd.isna(a) else b)
    chosen_grill1 = dt_grill.combine(grill, lambda a, b: a if not pd.isna(a) else b)
    grill_ts = chosen_grill2.where(~chosen_grill2.isna() & (chosen_grill2 <= 1800), chosen_grill1)

    make_ts = dt_make.combine(make, lambda a, b: a if not pd.isna(a) else b)

    expo_ts = expo
    cust_ts = custard
    dt_expo_ts = dt_expo

    d_grill = _positive_diff(grill_ts, ordering_ts)
    d_make = _positive_diff(make_ts, grill_ts)
    d_expo = _positive_diff(expo_ts, make_ts)
    d_cust = cust_ts

    expo_eb = ((expo_ts > 0) & (make_ts > 0) & (expo_ts < make_ts)).astype(int)
    dt_d_expo = _positive_diff(dt_expo_ts, make_ts)
    dt_expo_eb = ((dt_expo_ts > 0) & (make_ts > 0) & (dt_expo_ts < make_ts)).astype(int)

    final_stage = pd.concat(
        [expo_ts.rename("expo"), cust_ts.rename("cust"), dt_expo_ts.rename("dt_expo"), close_clip.rename("close")],
        axis=1
    )
    win_final = final_stage.max(axis=1)
    window_time = _positive_diff(win_final, start_to_tender, upper=3600)

    toc_col_name = "Time of Check" if "Time of Check" in sos_df.columns else None
    kitchen_start_sec = pd.Series(np.nan, index=sos_df.index)
    kitchen_end_sec = pd.Series(np.nan, index=sos_df.index)
    kitchen_at_expo = pd.Series(np.nan, index=sos_df.index)
    kitchen_at_expo_others = pd.Series(np.nan, index=sos_df.index)
    kitchen_overlap = pd.Series(np.nan, index=sos_df.index)
    kitchen_overlap_others = pd.Series(np.nan, index=sos_df.index)

    registers_total = pd.Series(np.nan, index=sos_df.index)
    registers_total_others = pd.Series(np.nan, index=sos_df.index)
    registers_dt = pd.Series(np.nan, index=sos_df.index)
    registers_dt_others = pd.Series(np.nan, index=sos_df.index)
    registers_dico = pd.Series(np.nan, index=sos_df.index)
    registers_dico_others = pd.Series(np.nan, index=sos_df.index)

    dt_cars = pd.Series(np.nan, index=sos_df.index)
    dt_cars_others = pd.Series(np.nan, index=sos_df.index)

    if toc_col_name is not None:
        toc = pd.to_datetime(sos_df[toc_col_name], errors="coerce")
        tsec = toc.dt.hour * 3600 + toc.dt.minute * 60 + toc.dt.second

        ks = (tsec - total_raw).where(~total_raw.isna() & ~tsec.isna())
        ke = tsec

        kitchen_start_sec = ks
        kitchen_end_sec = ke

        starts = ks.to_numpy()
        ends = ke.to_numpy()
        t_arr = tsec.to_numpy()
        n = len(sos_df)

        valid = (~np.isnan(starts)) & (~np.isnan(ends))

        for i in range(n):
            s_i = starts[i]
            e_i = ends[i]
            t_i = t_arr[i]
            if np.isnan(s_i) or np.isnan(e_i) or np.isnan(t_i):
                continue

            mask_instant = valid & (starts <= t_i) & (ends >= t_i)
            cnt_instant = mask_instant.sum()
            kitchen_at_expo.iloc[i] = cnt_instant
            kitchen_at_expo_others.iloc[i] = cnt_instant - 1

            mask_overlap = valid & (starts <= e_i) & (ends >= s_i)
            cnt_overlap = mask_overlap.sum()
            kitchen_overlap.iloc[i] = cnt_overlap
            kitchen_overlap_others.iloc[i] = cnt_overlap - 1

        # Register concurrency: DI/CO (TimeOfCheck..+Tender), DT (TimeOfCheck..+ordering_ts)
        n = len(sos_df)
        rs = np.full(n, np.nan)
        re_ = np.full(n, np.nan)

        is_dt_np = is_dt.to_numpy()
        is_delivery_np = is_delivery.to_numpy()
        ordering_np = ordering_ts.to_numpy()
        tender_np = tender_raw.to_numpy()
        tsec_np = tsec.to_numpy()

        for i in range(n):
            if np.isnan(tsec_np[i]):
                continue
            if is_dt_np[i]:
                dur = ordering_np[i]
                if not np.isnan(dur) and dur > 0:
                    rs[i] = tsec_np[i]
                    re_[i] = tsec_np[i] + dur
            else:
                if is_delivery_np[i]:
                    continue
                dur = tender_np[i]
                if not np.isnan(dur) and dur > 0:
                    rs[i] = tsec_np[i]
                    re_[i] = tsec_np[i] + dur

        valid_reg = (~np.isnan(rs)) & (~np.isnan(re_))

        for i in range(n):
            if not valid_reg[i]:
                continue
            s_i = rs[i]
            e_i = re_[i]

            mask_all = valid_reg & (rs <= e_i) & (re_ >= s_i)
            cnt_all = mask_all.sum()
            registers_total.iloc[i] = cnt_all
            registers_total_others.iloc[i] = cnt_all - 1

            mask_dt = mask_all & is_dt_np
            cnt_dt = mask_dt.sum()
            registers_dt.iloc[i] = cnt_dt
            registers_dt_others.iloc[i] = cnt_dt - (1 if is_dt_np[i] else 0)

            mask_dico = mask_all & (~is_dt_np)
            cnt_dico = mask_dico.sum()
            registers_dico.iloc[i] = cnt_dico
            registers_dico_others.iloc[i] = cnt_dico - (1 if not is_dt_np[i] else 0)

        # DT car concurrency: TimeOfCheck..+Close
        car_start = np.full(n, np.nan)
        car_end = np.full(n, np.nan)
        close_np = close_raw.to_numpy()

        for i in range(n):
            if not is_dt_np[i]:
                continue
            if np.isnan(tsec_np[i]) or np.isnan(close_np[i]) or close_np[i] <= 0:
                continue
            car_start[i] = tsec_np[i]
            car_end[i] = tsec_np[i] + close_np[i]

        valid_car = (~np.isnan(car_start)) & (~np.isnan(car_end))
        for i in range(n):
            if not valid_car[i]:
                continue
            s_i = car_start[i]
            e_i = car_end[i]
            mask_car = valid_car & (car_start <= e_i) & (car_end >= s_i)
            cnt_car = mask_car.sum()
            dt_cars.iloc[i] = cnt_car
            dt_cars_others.iloc[i] = cnt_car - 1

    sos_df = sos_df.copy()

    sos_df["kitchen_start_sec"] = kitchen_start_sec
    sos_df["kitchen_end_sec"] = kitchen_end_sec
    sos_df["kitchen_in_kitchen_at_expo"] = kitchen_at_expo
    sos_df["kitchen_in_kitchen_at_expo_others"] = kitchen_at_expo_others
    sos_df["kitchen_overlap_count"] = kitchen_overlap
    sos_df["kitchen_overlap_others"] = kitchen_overlap_others

    sos_df["registers_in_use_total"] = registers_total
    sos_df["registers_in_use_total_others"] = registers_total_others
    sos_df["registers_in_use_dt"] = registers_dt
    sos_df["registers_in_use_dt_others"] = registers_dt_others
    sos_df["registers_in_use_dico"] = registers_dico
    sos_df["registers_in_use_dico_others"] = registers_dico_others

    sos_df["dt_cars_in_lane"] = dt_cars
    sos_df["dt_cars_in_lane_others"] = dt_cars_others

    sos_df["dico_ticket_time_sec"] = np.where(is_dico, total_time, np.nan)
    sos_df["dico_time_ordering_sec"] = np.where(is_dico, ordering_ts, np.nan)
    sos_df["dico_time_grill_sec"] = np.where(is_dico, d_grill, np.nan)
    sos_df["dico_time_make_sec"] = np.where(is_dico, d_make, np.nan)
    sos_df["dico_time_expo_sec"] = np.where(is_dico, d_expo, np.nan)
    sos_df["dico_time_custard_sec"] = np.where(is_dico, d_cust, np.nan)
    sos_df["dico_expo_eb_flag"] = np.where(is_dico, expo_eb, np.nan)

    sos_df["dt_ticket_time_sec"] = np.where(is_dt, total_time, np.nan)
    sos_df["dt_time_ordering_sec"] = np.where(is_dt, ordering_ts, np.nan)
    sos_df["dt_time_grill_sec"] = np.where(is_dt, d_grill, np.nan)
    sos_df["dt_time_make_sec"] = np.where(is_dt, d_make, np.nan)
    sos_df["dt_time_expo_sec"] = np.where(is_dt, d_expo, np.nan)
    sos_df["dt_time_custard_sec"] = np.where(is_dt, d_cust, np.nan)
    sos_df["dt_time_dt_expo_sec"] = np.where(is_dt, dt_d_expo, np.nan)
    sos_df["dt_time_window_sec"] = np.where(is_dt, window_time, np.nan)
    sos_df["dt_expo_eb_flag"] = np.where(is_dt, dt_expo_eb, np.nan)

    return sos_df

def build_labor_counts_per_check(checks_csv, labor_csv, master_jobcodes=None):
    """Return (labor_df, labor_cols) where labor_df has one row per check and one column per master job code."""
    checks = pd.read_csv(checks_csv)
    labor = pd.read_csv(labor_csv)

    if checks.empty or labor.empty:
        return pd.DataFrame(), []

    store_col = pick_column(checks, ["storenum","store#","Store#","StoreNum"], "checks store")
    date_col  = pick_column(checks, ["dateofbusiness","DateOfBusiness"], "checks date")
    check_col = pick_column(checks, ["CheckId","CheckID","checkid","Check Id"], "checks CheckId")
    time_col  = pick_column(checks, ["timeofcheck","TimeOfCheck","Time of Check"], "checks timeofcheck")

    per_check = checks[[store_col, date_col, check_col, time_col]].drop_duplicates(subset=[store_col, date_col, check_col])
    per_check["check_minute"] = time_to_minutes(per_check[time_col])

    cin_col = pick_column(labor, ["Clock In Minute","ClockInMinute","Clock_In_Minute"], "labor clock in")
    cout_col = pick_column(labor, ["Clock Out Minute","ClockOutMinute","Clock_Out_Minute"], "labor clock out")

    job_col = pick_column(labor, ["JobCodeDesc"], "labor job description")

    if master_jobcodes is not None:
        job_codes = list(master_jobcodes)
    else:
        job_codes = sorted(labor[job_col].dropna().unique().tolist())

    labor_cols = [slugify_job(j) for j in job_codes]

    rows = []
    for _, row in per_check.iterrows():
        t = row["check_minute"]
        active = labor[(labor[cin_col] <= t) & (labor[cout_col] > t)]
        counts = active[job_col].value_counts()
        rec = {
            "storenum": row[store_col],
            "dateofbusiness": row[date_col],
            "CheckId": row[check_col],
        }
        for job, cnt in counts.items():
            rec[slugify_job(job)] = int(cnt)
        rows.append(rec)

    if not rows:
        return pd.DataFrame(), labor_cols
    labor_df = pd.DataFrame(rows)

    # Ensure all master labor columns exist, fill missing with 0
    for col in labor_cols:
        if col not in labor_df.columns:
            labor_df[col] = 0

    # Also ensure storenum/dateofbusiness/CheckId exist
    return labor_df, labor_cols

def main():
    ap = argparse.ArgumentParser(description="Pull SOS + checks + labor per day and merge (global job codes, station times, kitchen + register + DT car concurrency).")
    ap.add_argument("--store", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--sos-dataset-id", required=True)
    ap.add_argument("--checks-dataset-id", required=True)
    ap.add_argument("--labor-dataset-id", required=True)
    ap.add_argument("--out", default="sos_with_checks_and_labor.csv")
    ap.add_argument("--max-checks", type=int, default=None)
    ap.add_argument("--jobcodes-file", help="CSV with master list of JobCodeDesc to ensure consistent columns.")
    ap.add_argument("--python-exe", default="python")
    args = ap.parse_args()

    store = args.store
    dob = args.date

    SOS_CSV = "sos_raw.csv"
    CHECKS_CSV = "checks_raw.csv"
    SUMMARY_CSV = "checks_summary.csv"
    LABOR_CSV = "labor_raw.csv"

    master_jobcodes = None
    if args.jobcodes_file and os.path.exists(args.jobcodes_file):
        jc_df = pd.read_csv(args.jobcodes_file)
        if "JobCodeDesc" not in jc_df.columns:
            raise SystemExit(f"jobcodes-file must have a 'JobCodeDesc' column; found {list(jc_df.columns)}")
        master_jobcodes = sorted(jc_df["JobCodeDesc"].dropna().unique().tolist())
        print(f"[INFO] Loaded {len(master_jobcodes)} master job codes from {args.jobcodes_file}")

    # 1) SOS
    run_cmd([
        args.python_exe, "domo_query_std_filters.py",
        "--dataset-id", args.sos_dataset_id,
        "--out", SOS_CSV,
        "--filter", f"store# is {store}",
        "--filter", f"DateOfBusiness is {dob}"
    ])
    sos_df = pd.read_csv(SOS_CSV)
    if sos_df.empty:
        sos_df.to_csv(args.out, index=False)
        return

    sos_df = add_station_time_and_concurrency_features(sos_df)

    # 2) Checks
    run_cmd([
        args.python_exe, "domo_query_std_filters.py",
        "--dataset-id", args.checks_dataset_id,
        "--out", CHECKS_CSV,
        "--filter", f"storenum is {store}",
        "--filter", f"dateofbusiness is {dob}"
    ])
    checks_df = pd.read_csv(CHECKS_CSV)
    if checks_df.empty:
        sos_df.to_csv(args.out, index=False)
        return

    # 3) Summarize checks
    run_cmd([
        args.python_exe, "summarize_checks.py",
        "--input", CHECKS_CSV,
        "--output", SUMMARY_CSV
    ])
    sum_df = pd.read_csv(SUMMARY_CSV)
    if sum_df.empty:
        sos_df.to_csv(args.out, index=False)
        return

    if args.max_checks:
        checkcol = pick_column(sum_df, ["CheckId","CheckID","checkid"], "summary checkid")
        allowed = set(sum_df[checkcol].dropna().unique().tolist()[:args.max_checks])
        sum_df = sum_df[sum_df[checkcol].isin(allowed)]
        checks_df = checks_df[checks_df[checkcol].isin(allowed)]

    # 4) Labor
    run_cmd([
        args.python_exe, "domo_query_std_filters.py",
        "--dataset-id", args.labor_dataset_id,
        "--out", LABOR_CSV,
        "--filter", f"Store# is {store}",
        "--filter", f"date is {dob}"
    ])
    labor_df_raw = pd.read_csv(LABOR_CSV)

    if labor_df_raw.empty:
        labor_counts = pd.DataFrame()
        labor_cols = []
    else:
        labor_counts, labor_cols = build_labor_counts_per_check(CHECKS_CSV, LABOR_CSV, master_jobcodes)

    # 5) Merge labor -> summary
    if not labor_counts.empty:
        sum_store = pick_column(sum_df, ["storenum","store#","Store#","StoreNum"], "summary store")
        sum_date = pick_column(sum_df, ["dateofbusiness","DateOfBusiness"], "summary date")
        sum_check = pick_column(sum_df, ["CheckId","CheckID","checkid"], "summary checkid")

        labor_counts_ren = labor_counts.rename(columns={
            "storenum": sum_store,
            "dateofbusiness": sum_date,
            "CheckId": sum_check
        })
        sum_df = sum_df.merge(labor_counts_ren, on=[sum_store, sum_date, sum_check], how="left")
    else:
        print("[INFO] No labor counts computed; skipping labor merge.")

    # 6) Merge summary -> SOS
    sos_store = pick_column(sos_df, ["store#","Store#","storenum"], "sos store")
    sos_date = pick_column(sos_df, ["DateOfBusiness","dateofbusiness"], "sos date")
    sos_check = pick_column(sos_df, ["CheckID","CheckId","checkid"], "sos check")

    sum_store = pick_column(sum_df, ["storenum","store#","Store#","StoreNum"], "summary store")
    sum_date = pick_column(sum_df, ["dateofbusiness","DateOfBusiness"], "summary date")
    sum_check = pick_column(sum_df, ["CheckId","CheckID","checkid"], "summary check")

    sos_df["StoreKey"] = normalize(sos_df[sos_store])
    sos_df["DateKey"] = normalize(sos_df[sos_date])
    sos_df["CheckKey"] = normalize(sos_df[sos_check])

    sum_df["StoreKey"] = normalize(sum_df[sum_store])
    sum_df["DateKey"] = normalize(sum_df[sum_date])
    sum_df["CheckKey"] = normalize(sum_df[sum_check])

    merged = sos_df.merge(sum_df, on=["StoreKey","DateKey","CheckKey"], how="left")
    # Ensure all master labor_* columns from jobcodes-file exist in final output
    if args.jobcodes_file and os.path.exists(args.jobcodes_file):
        try:
            jc_df = pd.read_csv(args.jobcodes_file)
            if 'JobCodeDesc' in jc_df.columns:
                jobcodes = sorted(jc_df['JobCodeDesc'].dropna().unique().tolist())
                for j in jobcodes:
                    colname = slugify_job(j)
                    if colname not in merged.columns:
                        merged[colname] = 0
        except Exception as e:
            print(f'[WARN] Could not enforce master jobcodes on output: {e}')
    merged.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()