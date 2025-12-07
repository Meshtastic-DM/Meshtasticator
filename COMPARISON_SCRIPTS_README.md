# Routing Comparison Scripts

This directory contains two Python scripts for comparing performance metrics between different routing protocols in the Meshtastic simulator.

## Scripts Overview

### 1. `compare_reliability.py`
Compares packet delivery reliability across routing methods.

### 2. `compare_delays.py`
Compares packet transmission delays across routing methods.

---

## Quick Start

```powershell
# Run both comparison scripts
python compare_reliability.py --input-dir output
python compare_delays.py --input-dir output
```

All plots are automatically saved to: **`output/plots/`**

---

## Input Files Required

Both scripts automatically detect CSV files in the input directory with the following naming pattern:

### For Reliability Analysis
```
sensor_reliability_to_dest0_ROUTER_TYPE.<routing_method>.csv
dm_reliability_matrix_ROUTER_TYPE.<routing_method>.csv
broadcast_reliability_ROUTER_TYPE.<routing_method>.csv
```

### For Delay Analysis
```
sensor_packets_ROUTER_TYPE.<routing_method>.csv
dm_packets_ROUTER_TYPE.<routing_method>.csv
broadcast_packets_ROUTER_TYPE.<routing_method>.csv
```

**Example routing methods:**
- `MANAGED_FLOOD`
- `SDN_AODV`
- `AODV`

---

## Output Directory Structure

```
output/
├── plots/                              # All comparison plots saved here
│   ├── sensor_reliability_comparison_all_nodes.png
│   ├── sensor_reliability_average.png
│   ├── sensor_reliability_common_nodes.png
│   ├── dm_reliability_average.png
│   ├── broadcast_reliability_average.png
│   ├── sensor_delay_boxplot.png
│   ├── sensor_delay_violin.png
│   ├── sensor_delay_statistics.png
│   ├── sensor_delay_cdf.png
│   ├── sensor_delay_percentiles.png
│   ├── dm_delay_boxplot.png
│   ├── dm_delay_violin.png
│   ├── dm_delay_statistics.png
│   ├── dm_delay_cdf.png
│   ├── dm_delay_percentiles.png
│   ├── broadcast_delay_boxplot.png
│   ├── broadcast_delay_violin.png
│   ├── broadcast_delay_statistics.png
│   ├── broadcast_delay_cdf.png
│   ├── broadcast_delay_percentiles.png
│   └── overall_delay_comparison.png
│
├── sensor_reliability_to_dest0_ROUTER_TYPE.MANAGED_FLOOD.csv
├── sensor_reliability_to_dest0_ROUTER_TYPE.SDN_AODV.csv
├── dm_reliability_matrix_ROUTER_TYPE.MANAGED_FLOOD.csv
├── dm_reliability_matrix_ROUTER_TYPE.SDN_AODV.csv
└── ... (other CSV files)
```

---

## Detailed Script Documentation

## `compare_reliability.py`

### What It Does
Compares packet delivery reliability (percentage of successfully delivered packets) between different routing protocols.

### Generated Plots (5 total)

#### 1. **sensor_reliability_comparison_all_nodes.png**
- **Type:** Grouped bar chart
- **Shows:** Reliability from each sensor node to destination 0
- **X-axis:** Node IDs (0-24)
- **Y-axis:** Reliability (0-1.0, where 1.0 = 100%)
- **Bars:** Side-by-side comparison for each routing method
- **Features:** Value labels on each bar showing exact reliability

**What to look for:**
- Which routing method has consistently higher bars (better reliability)
- Which nodes have poor reliability (short bars)
- Variance between routing methods

#### 2. **sensor_reliability_average.png**
- **Type:** Bar chart with error bars
- **Shows:** Average reliability across all sensor nodes
- **X-axis:** Routing methods
- **Y-axis:** Average reliability
- **Features:** 
  - Standard deviation shown as error bars
  - Value labels showing both decimal (e.g., 0.8900) and percentage (89.00%)

**What to look for:**
- Overall winner for sensor packet delivery
- Consistency (smaller error bars = more consistent)

#### 3. **sensor_reliability_common_nodes.png**
- **Type:** Grouped bar chart (focused comparison)
- **Shows:** Head-to-head comparison for nodes that have data in both routing methods
- **Similar to plot 1 but filtered to common nodes only**

**What to look for:**
- Direct apples-to-apples comparison
- Per-node performance differences

#### 4. **dm_reliability_average.png**
- **Type:** Bar chart with error bars
- **Shows:** Average reliability for Direct Message (DM) packets across all source-destination pairs
- **X-axis:** Routing methods
- **Y-axis:** Average DM reliability

**What to look for:**
- Which routing method is better for point-to-point communication
- Variability in DM delivery success

#### 5. **broadcast_reliability_average.png**
- **Type:** Bar chart
- **Shows:** Average reliability for broadcast packets from source node 0
- **X-axis:** Routing methods
- **Y-axis:** Average broadcast reliability

**What to look for:**
- Which routing method is better for one-to-many communication
- Trade-offs: some methods excel at unicast but not broadcast

### Console Output Example
```
============================================================
RELIABILITY COMPARISON TOOL
============================================================

Input directory: E:\FYP\Repositories\Meshtasticator\output
Output directory: E:\FYP\Repositories\Meshtasticator\output\plots

Found sensor reliability files:
  - MANAGED_FLOOD: sensor_reliability_to_dest0_ROUTER_TYPE.MANAGED_FLOOD.csv
  - SDN_AODV: sensor_reliability_to_dest0_ROUTER_TYPE.SDN_AODV.csv

============================================================
SENSOR PACKET RELIABILITY COMPARISON
============================================================

Loaded MANAGED_FLOOD: 25 nodes
  Average reliability: 0.5632 (56.32%)
  Min reliability: 0.0000
  Max reliability: 1.0000
  Nodes with data: 12/25

Loaded SDN_AODV: 25 nodes
  Average reliability: 0.8900 (89.00%)
  Min reliability: 0.0000
  Max reliability: 1.0000
  Nodes with data: 12/25

✓ Saved: output\plots\sensor_reliability_comparison_all_nodes.png
✓ Saved: output\plots\sensor_reliability_average.png
✓ Saved: output\plots\sensor_reliability_common_nodes.png

[... similar output for DM and broadcast ...]

============================================================
COMPARISON COMPLETE!
============================================================

All plots saved to: E:\FYP\Repositories\Meshtasticator\output\plots
```

---

## `compare_delays.py`

### What It Does
Compares packet transmission delays (time from send to receive) between different routing protocols.

### Generated Plots (16 total = 5 per packet type × 3 packet types + 1 overall)

For each packet type (sensor, DM, broadcast), the script generates:

#### 1. **<packet_type>_delay_boxplot.png**
- **Type:** Box plot
- **Shows:** Distribution of delays with quartiles
- **Features:**
  - Box shows 25th-75th percentile range
  - Line in box = median
  - Whiskers = 1.5×IQR range
  - Dots = outliers
  
**What to look for:**
- Median delay (line inside box) - lower is better
- Box size (smaller = more consistent)
- Outliers (dots) - indicate occasional high delays

#### 2. **<packet_type>_delay_violin.png**
- **Type:** Violin plot
- **Shows:** Probability density of delay values
- **Features:** Width at any point shows how common that delay value is

**What to look for:**
- Shape of distribution (narrow = consistent, wide = variable)
- Peak locations (most common delay values)
- Multiple peaks = bimodal behavior

#### 3. **<packet_type>_delay_statistics.png**
- **Type:** Grouped bar chart
- **Shows:** Mean, Median, and 95th percentile delays
- **X-axis:** Statistic type (Mean, Median, P95)
- **Y-axis:** Delay in milliseconds

**What to look for:**
- Mean vs Median gap (large gap = many outliers)
- P95 values (95% of packets arrive within this time)

#### 4. **<packet_type>_delay_cdf.png**
- **Type:** Cumulative Distribution Function plot
- **Shows:** Percentage of packets delivered by a given delay
- **X-axis:** Delay (ms)
- **Y-axis:** Cumulative probability (0-1)

**What to look for:**
- Steeper curves = faster delivery
- Higher curves = more packets delivered quickly
- Horizontal shifts = routing method delay differences

#### 5. **<packet_type>_delay_percentiles.png**
- **Type:** Bar chart
- **Shows:** Delay at various percentiles (50th, 75th, 90th, 95th, 99th)
- **X-axis:** Percentile
- **Y-axis:** Delay (ms)

**What to look for:**
- How delays grow at higher percentiles
- Tail behavior (99th percentile = worst-case scenarios)

#### 16. **overall_delay_comparison.png**
- **Type:** Box plot grid (all packet types together)
- **Shows:** Side-by-side delay comparison across all packet types and routing methods
- **Features:** Combined view for easy comparison

**What to look for:**
- Which routing method is consistently faster
- Which packet types have higher delays
- Overall performance trends

### Console Output Example
```
============================================================
DELAY COMPARISON TOOL
============================================================

Input directory: E:\FYP\Repositories\Meshtasticator\output
Output directory: E:\FYP\Repositories\Meshtasticator\output\plots

Found sensor packet files:
  - MANAGED_FLOOD: sensor_packets_ROUTER_TYPE.MANAGED_FLOOD.csv
  - SDN_AODV: sensor_packets_ROUTER_TYPE.SDN_AODV.csv

============================================================
SENSOR PACKET DELAY COMPARISON
============================================================

Loaded MANAGED_FLOOD: 138 packets
  Mean delay: 1163209.88 ms
  Median delay: 705302.07 ms
  Min delay: 4538.48 ms
  Max delay: 5454446.63 ms
  Std dev: 1251620.44 ms

Loaded SDN_AODV: 233 packets
  Mean delay: 61569.14 ms
  Median delay: 12813.28 ms
  Min delay: 1228.48 ms
  Max delay: 484840.83 ms
  Std dev: 107070.02 ms

✓ Saved: output\plots\sensor_delay_boxplot.png
✓ Saved: output\plots\sensor_delay_violin.png
✓ Saved: output\plots\sensor_delay_statistics.png
✓ Saved: output\plots\sensor_delay_cdf.png
✓ Saved: output\plots\sensor_delay_percentiles.png

[... similar output for DM and broadcast ...]

============================================================
SUMMARY STATISTICS TABLE
============================================================
Packet Type & Routing          Mean         Median       P95
------------------------------------------------------------
Sensor MANAGED_FLOOD      1163209.9    705302.1     3866665.1
Sensor SDN_AODV              61569.1     12813.3      343995.5
DM     MANAGED_FLOOD        499345.2     59228.0     2380587.8
DM     SDN_AODV              40248.0      9336.2      221878.7
Broadcast MANAGED_FLOOD   1665358.5   1653287.6    2484708.8
Broadcast SDN_AODV        2965440.6   3114032.8    4218926.2

============================================================
DELAY COMPARISON COMPLETE!
============================================================

All plots saved to: E:\FYP\Repositories\Meshtasticator\output\plots
```

---

## Usage Examples

### Basic Usage (Default Settings)
```powershell
# Compare reliability
python compare_reliability.py --input-dir output

# Compare delays
python compare_delays.py --input-dir output
```

### Custom Output Directory
```powershell
python compare_reliability.py --input-dir output --output-dir results/plots
python compare_delays.py --input-dir output --output-dir results/plots
```

### Run Both and Open Output Folder
```powershell
python compare_reliability.py --input-dir output
python compare_delays.py --input-dir output
explorer.exe output\plots
```

---

## Interpreting Results

### Reliability Results

**High Reliability (0.8 - 1.0 = 80-100%)**
- ✅ Excellent: Protocol successfully delivers most packets
- Best for critical applications

**Medium Reliability (0.5 - 0.8 = 50-80%)**
- ⚠️ Acceptable: Some packet loss occurs
- May need optimization or better conditions

**Low Reliability (0.0 - 0.5 = 0-50%)**
- ❌ Poor: Significant packet loss
- Protocol may not be suitable for this topology

### Delay Results

**Low Delay (< 100,000 ms)**
- ✅ Fast: Packets delivered quickly
- Good for real-time applications

**Medium Delay (100,000 - 1,000,000 ms)**
- ⚠️ Moderate: Some latency present
- Acceptable for non-critical messages

**High Delay (> 1,000,000 ms)**
- ❌ Slow: Significant delays
- May indicate routing issues or network congestion

---

## Common Findings

Based on typical simulation results:

### Sensor & DM Packets (Unicast)
- **SDN_AODV** typically shows:
  - ✅ Higher reliability (85-90%)
  - ✅ Lower delays
  - ✅ More consistent performance
  - Reason: On-demand routing with optimized paths

- **MANAGED_FLOOD** typically shows:
  - ⚠️ Lower reliability (50-60%)
  - ❌ Higher delays
  - ❌ More variable performance
  - Reason: Broadcasting causes congestion

### Broadcast Packets (One-to-Many)
- **MANAGED_FLOOD** typically shows:
  - ✅ Higher reliability (60-70%)
  - Reason: Native flooding mechanism

- **SDN_AODV** typically shows:
  - ⚠️ Lower reliability (30-40%)
  - Reason: Optimized for unicast, not broadcast

---

## Troubleshooting

### "No reliability CSV files found"
- **Cause:** Missing input files or wrong directory
- **Solution:** Run `loraMesh.py` first to generate CSV files
  ```powershell
  python loraMesh.py --from-file config.yaml
  ```

### "Error: Input directory does not exist"
- **Cause:** Specified directory doesn't exist
- **Solution:** Check path or create directory
  ```powershell
  mkdir output
  ```

### Empty or incomplete plots
- **Cause:** Insufficient data in CSV files
- **Solution:** Run longer simulations or check node configurations

### MatplotlibDeprecationWarning
- **Note:** Harmless warning about parameter naming
- **Impact:** None - plots still generate correctly

---

## Dependencies

Both scripts require:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `matplotlib` - Plotting
- `pathlib` - File path handling

Install via:
```powershell
pip install pandas numpy matplotlib
```

Or use the project's virtual environment:
```powershell
.\venv\Scripts\Activate.ps1
```

---

## Output Format

### Plot Specifications
- **Resolution:** 200 DPI (high quality for papers/presentations)
- **Format:** PNG with transparency
- **Size:** Automatically adjusted based on data (typically 800-1400px wide)
- **Colors:** Colorblind-friendly Set3 palette
- **Fonts:** Bold titles, clear axis labels

### Best Practices for Using Plots

**For Papers/Reports:**
- Use PNG files at 200 DPI (print-ready quality)
- Include plots in order: reliability first, then delays
- Reference specific metrics from console output

**For Presentations:**
- Use average comparison plots for overview slides
- Use detailed plots (box plots, CDFs) for technical deep-dives
- Highlight key findings with arrows/annotations in your slides

**For Analysis:**
- Compare corresponding plots side-by-side
- Look for patterns across multiple packet types
- Note trade-offs (e.g., reliability vs. delay)

---

## Summary

| Script | Input Files | Output Plots | Key Metrics |
|--------|-------------|--------------|-------------|
| `compare_reliability.py` | `*reliability*.csv` | 5 plots | Reliability (0-1) |
| `compare_delays.py` | `*packets*.csv` | 16 plots | Delay (ms), statistics |

**Total Output:** 21 high-quality comparison plots saved to `output/plots/`

All plots are automatically generated, labeled, and saved - ready for analysis, presentations, or publications!
