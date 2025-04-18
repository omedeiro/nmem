import pandas as pd
import numpy as np
from nmem.analysis.autoprobe_analysis.data import load_autoprobe_data

def summarize_die_yield(df, wafer_rows, min_kohm=1, max_kohm=50000):
    df = df.copy()
    df["Rmean_k"] = df["Rmean"] / 1e3

    summary_records = []

    for row_num in wafer_rows:
        row_df = df[df["die"].str.endswith(row_num)].copy()
        row_df["is_outlier"] = (
            row_df["Rmean_k"].isna() |
            (row_df["Rmean_k"] < min_kohm) |
            (row_df["Rmean_k"] > max_kohm)
        )

        grouped = row_df.groupby("die")
        die_outlier_counts = grouped["is_outlier"].sum().astype(int)
        die_total_counts = grouped["is_outlier"].count()

        for die in die_outlier_counts.index:
            n_bad = die_outlier_counts[die]
            n_total = die_total_counts[die]
            yield_pct = 100 * (1 - n_bad / n_total) if n_total > 0 else np.nan
            summary_records.append({
                "row": row_num,
                "die": die,
                "total_devices": n_total,
                "outliers": n_bad,
                "yield_percent": yield_pct
            })

    summary_df = pd.DataFrame(summary_records)

    # Add row-level statistics
    row_stats = (
        summary_df
        .groupby("row")["yield_percent"]
        .agg(
            row_mean_yield="mean",
            row_std_yield="std",
            row_min_yield="min",
            row_max_yield="max"
        )
        .reset_index()
    )

    return summary_df, row_stats

# Example usage
if __name__ == "__main__":
    df = load_autoprobe_data("autoprobe_parsed.mat")
    wafer_rows = ["7"]
    summary_df, row_stats_df = summarize_die_yield(df, wafer_rows, min_kohm=10, max_kohm=5000)

    print(summary_df.head())      # Die-level stats
    print(row_stats_df.head())    # Row-level summary
