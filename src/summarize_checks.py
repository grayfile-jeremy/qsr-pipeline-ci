#!/usr/bin/env python3
"""
summarize_checks.py

Condense the CSV output from domo_query_std_filters.py into one row per check,
with counts by category and useful totals.

Example:
  python summarize_checks.py \
    --input filtered_checkID.csv \
    --output summarized_checks.csv
"""

import argparse
import pandas as pd


def summarize_checks(input_csv: str, output_csv: str) -> None:
    df = pd.read_csv(input_csv)

    # --- Basic cleanup ---
    df["Price"] = pd.to_numeric(df.get("Price", 0), errors="coerce").fillna(0)
    df["LTO"] = df.get("LTO", "").fillna("")
    df["category"] = df.get("category", "").fillna("Blank")
    df["combo"] = df.get("combo", "").fillna("")
    df["InCombo"] = df.get("InCombo", "").fillna("")
    df["name"] = df.get("name", "").fillna("")

    # Normalized helper columns
    df["category_norm"] = df["category"].str.upper()
    df["combo_norm"] = df["combo"].str.upper()
    df["InCombo_norm"] = df["InCombo"].str.upper()
    df["name_norm"] = df["name"].str.upper()

    # --- Define grouping keys ---
    group_keys = ["storenum", "dateofbusiness", "CheckId"]
    group_keys = [c for c in group_keys if c in df.columns]

    if "CheckId" not in group_keys:
        raise ValueError("Expected 'CheckId' column to group by.")

    # Helper functions
    def item_count_fn(s):
        sub = df.loc[s.index]
        return (~sub["category_norm"].isin(["MOD", "BLANK"])).sum()

    def count_combo_fn(s):
        return (df.loc[s.index, "combo_norm"] == "COMBO").sum()

    def combo_item_count_fn(s):
        sub = df.loc[s.index]
        mask_valid = ~sub["category_norm"].isin(["MOD", "BLANK"])
        mask_combo = (sub["combo_norm"] == "COMBO") | (sub["InCombo_norm"] == "Y")
        return (mask_valid & mask_combo).sum()

    def noncombo_item_count_fn(s):
        sub = df.loc[s.index]
        mask_valid = ~sub["category_norm"].isin(["MOD", "BLANK"])
        mask_combo = (sub["combo_norm"] == "COMBO") | (sub["InCombo_norm"] == "Y")
        return (mask_valid & ~mask_combo).sum()

    def fries_count_fn(s):
        sub = df.loc[s.index]
        return ((sub["category_norm"] == "SIDES") & (sub["name_norm"] == "FRIES")).sum()

    def other_sides_count_fn(s):
        sub = df.loc[s.index]
        return ((sub["category_norm"] == "SIDES") & (sub["name_norm"] != "FRIES")).sum()

    # ⭐ NEW: Non-combo sides
    def noncombo_sides_count_fn(s):
        sub = df.loc[s.index]
        return (
            (sub["category_norm"] == "SIDES") &
            (sub["InCombo_norm"] == "N")
        ).sum()

    # --- Aggregations ---
    base_agg = df.groupby(group_keys).agg(
        total_price=("Price", "sum"),
        item_count=("CheckId", item_count_fn),
        LTO_item_count=("LTO", lambda s: (s == "Y").sum()),
        row_count=("CheckId", "size"),
        count_Combo=("CheckId", count_combo_fn),
        combo_item_count=("CheckId", combo_item_count_fn),
        noncombo_item_count=("CheckId", noncombo_item_count_fn),
        fries_count=("CheckId", fries_count_fn),
        other_sides_count=("CheckId", other_sides_count_fn),
        noncombo_sides_count=("CheckId", noncombo_sides_count_fn),   # ⭐ ADDED
    )

    # Category counts
    category_counts = (
        df.groupby(group_keys + ["category"])
          .size()
          .unstack("category", fill_value=0)
          .add_suffix("_count")
    )

    # Combine
    result = (
        base_agg
        .join(category_counts, how="left")
        .reset_index()
    )

    result.to_csv(output_csv, index=False)
    print(f"Wrote summarized checks to: {output_csv}")


def main():
    ap = argparse.ArgumentParser(description="Summarize Domo check detail CSV.")
    ap.add_argument("--input", "-i", required=True)
    ap.add_argument("--output", "-o", required=True)
    args = ap.parse_args()
    summarize_checks(args.input, args.output)


if __name__ == "__main__":
    main()
