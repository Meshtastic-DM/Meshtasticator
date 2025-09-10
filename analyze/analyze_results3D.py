# analyze_delays_3d.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D projection

def main():
    ap = argparse.ArgumentParser(description="3D visualize delays: x=source, y=destination, z=delay")
    ap.add_argument("input_csv", help="CSV with columns: source,destination,delay_value")
    ap.add_argument("--summary", default="message_summary.csv", help="Output CSV for per-pair stats")
    ap.add_argument("--figure", default="message_delays_3d.png", help="Output image for 3D plot")
    ap.add_argument("--max-points", type=int, default=None,
                    help="Optional cap on #scatter points for speed (randomly sampled)")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="XY jitter size to reduce overplot (e.g., 0.08). 0 = no jitter.")
    ap.add_argument("--logz", action="store_true",
                    help="Plot log10(delay) on the Z axis (helps when delays span decades).")
    args = ap.parse_args()

    # ---- Load & clean ----
    df = pd.read_csv(args.input_csv)
    need = {"source", "destination", "delay_value"}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV must have columns: {need}")

    df["source"] = pd.to_numeric(df["source"], errors="coerce")
    df["destination"] = pd.to_numeric(df["destination"], errors="coerce")
    df["delay_value"] = pd.to_numeric(df["delay_value"], errors="coerce")
    df = df.dropna(subset=["source", "destination", "delay_value"])

    # If IDs are integer-like (e.g., 0.0), cast to int for clean ticks
    if np.all(np.isclose(df["source"] % 1, 0)) and np.all(np.isclose(df["destination"] % 1, 0)):
        df["source"] = df["source"].astype(int)
        df["destination"] = df["destination"].astype(int)

    # Optional downsample for massive datasets (for faster scatter)
    if args.max_points is not None and len(df) > args.max_points:
        df = df.sample(args.max_points, random_state=42).copy()

    # ---- Aggregate stats per (src,dst) ----
    grouped = df.groupby(["source", "destination"])["delay_value"]
    summary = grouped.agg(
        count="count",
        mean="mean",
        min_delay="min",
        max_delay="max"
    ).reset_index()
    summary["range"] = summary["max_delay"] - summary["min_delay"]
    summary.to_csv(args.summary, index=False)
    print(f"Summary written to: {Path(args.summary).resolve()}")

    # ---- Prepare 3D coordinates ----
    # unique sorted axes (for clean tick labeling)
    xs = np.sort(df["source"].unique())
    ys = np.sort(df["destination"].unique())

    # (Optional) log-transform Z if requested
    if args.logz:
        zvals = np.log10(df["delay_value"].to_numpy())
        z_min = float(np.nanmin(zvals))
        z_max = float(np.nanmax(zvals))
        z_label = "log10(delay_value)"
    else:
        zvals = df["delay_value"].to_numpy()
        z_min = float(np.nanmin(zvals))
        z_max = float(np.nanmax(zvals))
        z_label = "delay_value"

    # Build jitter if requested
    rng = np.random.default_rng(42)
    jitter = args.jitter

    # ---- Plot 3D ----
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Scatter all points
    scatter_x = df["source"].to_numpy().astype(float)
    scatter_y = df["destination"].to_numpy().astype(float)
    if jitter > 0:
        scatter_x = scatter_x + (rng.random(len(scatter_x)) - 0.5) * 2 * jitter
        scatter_y = scatter_y + (rng.random(len(scatter_y)) - 0.5) * 2 * jitter
    ax.scatter(scatter_x, scatter_y, zvals, s=8, alpha=0.5)

    # Vertical min-max lines + mean diamonds per pair
    for _, row in summary.iterrows():
        x = float(row["source"])
        y = float(row["destination"])
        z0 = float(row["min_delay"])
        z1 = float(row["max_delay"])
        zm = float(row["mean"])

        if args.logz:
            z0, z1, zm = np.log10([z0, z1, zm])

        # slight XY offset to avoid covering the mean marker by the line (optional)
        xo = x + (rng.random() - 0.5) * 0.0
        yo = y + (rng.random() - 0.5) * 0.0

        # min->max range line
        ax.plot([xo, xo], [yo, yo], [z0, z1], linewidth=2)

        # mean marker (diamond)
        ax.scatter([xo], [yo], [zm], marker="D", s=30)

    # Axes labels and ticks
    ax.set_xlabel("source")
    ax.set_ylabel("destination")
    ax.set_zlabel(z_label)

    # Use integer ticks if sources/destinations are integers
    ax.set_xticks(xs)
    ax.set_yticks(ys)

    # Expand Z limits a bit for headroom
    zpad = 0.02 * (z_max - z_min) if z_max > z_min else 1.0
    ax.set_zlim(z_min - zpad, z_max + zpad)

    ax.set_title("3D delays: x=source, y=destination, z=delay\n(points = messages, line = min–max, diamond = mean)")
    plt.tight_layout()
    plt.savefig(args.figure, dpi=200)
    print(f"3D figure saved to: {Path(args.figure).resolve()}")
    plt.show()

if __name__ == "__main__":
    main()
