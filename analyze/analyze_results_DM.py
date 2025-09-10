import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# --- Assumes you already have df with columns: source, destination, delay_value ---
# If not, uncomment and load:
df = pd.read_csv("dm_packets_R.csv")
# df = pd.read_csv("broadcast_packets.csv")

# Clean types (safe if already clean)
df["source"] = pd.to_numeric(df["source"], errors="coerce")
df["destination"] = pd.to_numeric(df["destination"], errors="coerce")
df["delay_value"] = pd.to_numeric(df["delay_value"], errors="coerce")
df = df.dropna(subset=["source", "destination", "delay_value"])

# Make integer-like IDs into ints for clean labeling
if np.all(np.isclose(df["source"] % 1, 0)):
    df["source"] = df["source"].astype(int)
if np.all(np.isclose(df["destination"] % 1, 0)):
    df["destination"] = df["destination"].astype(int)

# Optional: filter out very small groups (e.g., pairs with < 2 messages)
MIN_COUNT = 1

# Sort sources for deterministic order
sources = sorted(df["source"].unique())

#pdf_path = "delays_by_source_DM_MF.pdf"
pdf_path = "delays_by_source_Broadcast_R.pdf"
with PdfPages(pdf_path) as pdf:
    for src in sources:
        df_src = df[df["source"] == src]
        # Group delays by destination under this source
        dest_groups = (
            df_src.groupby("destination")["delay_value"]
                 .apply(list)
                 .reset_index(name="delays")
                 .sort_values("destination")
        )
        # Filter by count if desired
        dest_groups = dest_groups[dest_groups["delays"].apply(len) >= MIN_COUNT]
        if dest_groups.empty:
            continue

        delays_list = [np.array(v) for v in dest_groups["delays"].tolist()]
        labels = dest_groups["destination"].astype(str).tolist()

        fig, ax = plt.subplots(figsize=(12, 6))
        # Boxplots (spread)
        ax.boxplot(delays_list, labels=labels, showfliers=False)

        # Overlay jittered dots
        for i, vals in enumerate(delays_list, start=1):
            x = np.random.normal(loc=i, scale=0.06, size=len(vals))
            ax.scatter(x, vals, s=8, alpha=0.5)

        ax.set_yscale("log")  # helpful if delays span decades
        ax.set_xlabel("destination")
        ax.set_ylabel("delay_value(ms)")
        ax.set_title(f"Delays from source {src} → each destination for DM messages")

        plt.tight_layout()
        pdf.savefig(fig)  # add page to PDF
        plt.close(fig)

print(f"Saved one page per source to: {pdf_path}")
