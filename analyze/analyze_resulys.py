import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Analyze message delay CSV file")
    parser.add_argument("input_csv", help="Path to input CSV (source,destination,delay_value)")
    parser.add_argument("--summary", default="message_summary.csv")
    parser.add_argument("--figure", default="message_delays_plot.png")
    parser.add_argument("--broadcast", default=False, action="store_true")
    parser.add_argument("--sensor", default=False, action="store_true")
    parser.add_argument("--DM", default=False, action="store_true")
    args = parser.parse_args()

    title = "Message Delays"
    if args.broadcast:
        title = "Broadcast Messages Delays"
    if args.sensor:
        title = "Sensor Messages Delays"
    if args.DM:
        title = "Direct Messages Delays"

    # ----------------------------
    # Load + normalize types
    # ----------------------------
    df = pd.read_csv(args.input_csv)

    required_cols = {"source", "destination", "delay_value"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Ensure numeric & integral IDs for consistent grouping/labels
    df["source"] = pd.to_numeric(df["source"], errors="coerce")
    df["destination"] = pd.to_numeric(df["destination"], errors="coerce")
    df["delay_value"] = pd.to_numeric(df["delay_value"], errors="coerce")
    df = df.dropna(subset=["source", "destination", "delay_value"])

    # If IDs are integer-like (e.g., 0.0, 1.0), cast to int
    if np.all(np.isclose(df["source"] % 1, 0)) and np.all(np.isclose(df["destination"] % 1, 0)):
        df["source"] = df["source"].astype(int)
        df["destination"] = df["destination"].astype(int)

    # ----------------------------
    # Aggregate stats
    # ----------------------------
    grouped = df.groupby(["source", "destination"])["delay_value"]
    summary = grouped.agg(count="count", mean="mean", min_delay="min", max_delay="max").reset_index()
    summary["range"] = summary["max_delay"] - summary["min_delay"]

    summary.to_csv(args.summary, index=False)
    print(f"Summary written to: {Path(args.summary).resolve()}")

    # ----------------------------
    # Box plot per (src, dst) pair
    # ----------------------------
    pairs = [tuple(x) for x in summary[["source", "destination"]].to_numpy()]
    x_labels = [f"{s}->{d}" for (s, d) in pairs]

    # Collect data for each pair in the same order as summary
    box_data = []
    for pair in pairs:
        src, dst = pair
        delays = df[(df["source"] == src) & (df["destination"] == dst)]["delay_value"].values
        box_data.append(delays)

    fig, ax = plt.subplots(figsize=(14, 7))
    box_colors = plt.cm.Set3.colors  # Use a colormap for variety
    bp = ax.boxplot(
        box_data, patch_artist=True, meanline=True,
        boxprops=dict(facecolor=box_colors[0], color='black'),
        medianprops=dict(color='red'),
        meanprops=dict(color='blue', linewidth=2),
        flierprops=dict(marker='o', markerfacecolor='gray', markersize=5, linestyle='none', alpha=0.5)
    )

    # Set different colors for each box
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)

    # Overlay individual data points (strip plot)
    for i, y in enumerate(box_data, 1):
        ax.scatter(np.full_like(y, i, dtype=float), y, color='black', alpha=0.7, s=12, zorder=3)

    ax.set_xticks(np.arange(1, len(x_labels) + 1))
    ax.set_xticklabels(x_labels, rotation=90)
    ax.set_xlabel("source -> destination")
    ax.set_ylabel("delay_value(ms)")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(args.figure, dpi=200)
    print(f"Figure saved to: {Path(args.figure).resolve()}")
    plt.show()

if __name__ == "__main__":
    main()
