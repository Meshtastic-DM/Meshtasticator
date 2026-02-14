# Running Meshtasticator on Google Colab and Cloud Platforms

This guide explains how to run the Meshtasticator mesh network simulator on Google Colab or other cloud platforms with checkpoint support for long-running simulations.

## 📚 Table of Contents

1. [Quick Start (Google Colab)](#quick-start-google-colab)
2. [Auto-Sync to Google Drive](#auto-sync-to-google-drive)
3. [Checkpoint Support](#checkpoint-support)
4. [Alternative Platforms](#alternative-platforms)
5. [Troubleshooting](#troubleshooting)
6. [Tips for Long Simulations](#tips-for-long-simulations)

---

## 🚀 Quick Start (Google Colab)

### Option 1: Use the Provided Notebook (Recommended)

1. **Upload the notebook** `Meshtasticator_Colab.ipynb` to your Google Drive

2. **Open it with Google Colab**:
   - Right-click the file in Google Drive
   - Select "Open with" → "Google Colaboratory"

3. **Run the cells sequentially**:
   - Click "Runtime" → "Run all" or press `Ctrl+F9`
   - Follow the prompts to mount Google Drive (recommended)

4. **Wait for completion**:
   - Monitor progress in the output cells
   - Results are automatically saved

### Option 2: Manual Setup

If you prefer to set up manually, run these commands in Colab cells:

```python
# 1. Mount Google Drive (optional but recommended)
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone repository
!git clone https://github.com/YOUR_USERNAME/Meshtasticator.git
%cd Meshtasticator

# 3. Install dependencies
!pip install -q -r requirements.txt

# 4. Run simulations with checkpoint AND auto-sync support
!python run_batch_simulations.py --runs 10 --routing AODV MANAGED_FLOOD --checkpoint --auto-sync
```

---

## 🔄 Auto-Sync to Google Drive

**NEW FEATURE**: Results are now automatically backed up to Google Drive **after each completed run**!

### Why Auto-Sync?

**Problem**: Traditional approach waits until all runs complete before manual backup
- ❌ If Colab disconnects at run 15 of 20, you lose all 15 completed runs
- ❌ Must remember to manually copy results
- ❌ Can't resume with partial results if local storage is cleared

**Solution**: Auto-sync backs up after EVERY run
- ✅ Run 1 completes → Immediately synced to Drive
- ✅ Run 2 completes → Immediately synced to Drive
- ✅ If disconnection happens, all completed runs are safe in Drive
- ✅ Can resume from any point

### Enabling Auto-Sync

**Option 1: Auto-detect (Recommended for Colab)**

```bash
python run_batch_simulations.py --runs 20 --checkpoint --auto-sync
```

This automatically:
- Detects if running in Google Colab
- Checks if Drive is mounted
- Uses `/content/drive/MyDrive/Meshtasticator_Results` as backup location

**Option 2: Manual Path**

```bash
python run_batch_simulations.py --runs 20 --checkpoint --sync-to-drive /content/drive/MyDrive/MyResults
```

Specify exact Google Drive path for backups.

### How Auto-Sync Works

```
Run 1 → [Complete] → [Sync to Drive] → [Checkpoint]
Run 2 → [Complete] → [Sync to Drive] → [Checkpoint]
Run 3 → [Complete] → [Sync to Drive] → [Checkpoint]
...
```

Each run:
1. Simulation completes
2. **Results immediately copied to Google Drive**
3. Checkpoint saved
4. Next run begins

### What Gets Synced?

After each run, the entire batch directory is synced:
- ✅ All CSV output files (sensor_packets, dm_packets, broadcast_packets)
- ✅ All PNG plots
- ✅ All PKL data files
- ✅ Configuration files (config.yaml, topology_params)
- ✅ Checkpoint files (.checkpoint.pkl, checkpoint.json)
- ✅ Metadata (run_metadata.json, batch_summary.json)

### Sync Efficiency

- **Incremental**: Only new/changed files are copied
- **Fast**: Typical sync takes 1-5 seconds per run
- **Safe**: Uses Python's `shutil.copy2` to preserve timestamps
- **Automatic**: No manual intervention required

### Checking Sync Status

During simulation, you'll see:
```
[OK] Run 5 completed successfully
[CHECKPOINT] Progress saved (5/20)
[SYNC] Syncing to Google Drive...
[SYNC] ✓ Backed up to Drive: /content/drive/MyDrive/Meshtasticator_Results/batch_results_20260214_120000
```

### Verifying Backups

Check your Google Drive:
```python
from pathlib import Path

drive_path = Path('/content/drive/MyDrive/Meshtasticator_Results')
batches = list(drive_path.glob('batch_results_*'))
print(f"Backed up batches: {len(batches)}")

for batch in sorted(batches)[-3:]:  # Show last 3
    checkpoint = batch / 'checkpoint.json'
    if checkpoint.exists():
        import json
        with open(checkpoint) as f:
            data = json.load(f)
            print(f"{batch.name}: {data['progress_percentage']:.0f}% complete")
```

### Resume with Auto-Sync

When resuming, don't forget to enable auto-sync again:

```bash
python run_batch_simulations.py --resume batch_results_20260214_120000 --auto-sync
```

This ensures continued backup of remaining runs.

### Requirements

**For `--auto-sync` to work**:
1. Running in Google Colab environment
2. Google Drive must be mounted:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Drive mount point accessible at `/content/drive`

**For `--sync-to-drive PATH` to work**:
1. Specified path must be writable
2. Path should exist or script will create it

### Example: Complete Protected Workflow

```bash
# Initial run with full protection
python run_batch_simulations.py \
    --runs 50 \
    --routing AODV MANAGED_FLOOD BL_A_AODV \
    --checkpoint \
    --auto-sync

# [Session disconnects after 20 runs]
# [All 20 runs are safe in Google Drive]

# Resume (in new session, after re-mounting Drive)
python run_batch_simulations.py \
    --resume batch_results_20260214_120000 \
    --auto-sync

# [Continues from run 21, keeps syncing]
```

### Performance Impact

| Operation | Time Added | Impact |
|-----------|------------|---------|
| Sync after each run | 1-5 seconds | Minimal |
| Total for 50 runs | 1-4 minutes | <2% overhead |
| Network usage | ~10-50 MB per run | Low |

**Verdict**: Minimal impact, massive safety improvement! 🎯

---

## 💾 Checkpoint Support

Checkpoints allow you to resume interrupted simulations without losing progress. This is crucial for long-running simulations on platforms with time limits.

### Enabling Checkpoints

Add the `--checkpoint` flag when running simulations:

```bash
python run_batch_simulations.py --runs 20 --checkpoint
```

### How Checkpoints Work

1. **Automatic Saving**: After each completed run, progress is saved to:
   - `.checkpoint.pkl` (binary, for quick loading)
   - `checkpoint.json` (human-readable status)

2. **What's Saved**:
   - Completed run IDs
   - Total runs to complete
   - Routing protocols tested
   - Start time and last update time

3. **Resume Location**: Checkpoints are saved in the batch results directory

### Resuming from Checkpoint

If your session disconnects or you need to stop:

```bash
# List available batch directories
ls -ltr batch_results_*/

# Resume from specific directory
python run_batch_simulations.py --resume batch_results_20260214_120000
```

**The script will**:
- ✅ Detect which runs are already completed
- ✅ Skip completed runs
- ✅ Continue from where it left off
- ✅ Maintain the same routing protocols and parameters

### Example Workflow

```bash
# Initial run (gets interrupted after 5 runs)
python run_batch_simulations.py --runs 20 --checkpoint
# Output: batch_results_20260214_120000

# Resume later
python run_batch_simulations.py --resume batch_results_20260214_120000
# Continues from run 6
```

### Checking Checkpoint Status

View the human-readable checkpoint:

```bash
cat batch_results_TIMESTAMP/checkpoint.json
```

Example output:
```json
{
  "completed_runs": [1, 2, 3, 4, 5],
  "total_runs": 20,
  "routing_types": ["AODV", "MANAGED_FLOOD", "BL_A_AODV"],
  "start_time_iso": "2026-02-14T12:00:00",
  "last_updated_iso": "2026-02-14T12:45:30",
  "progress_percentage": 25.0
}
```

---

## ☁️ Alternative Platforms

### Kaggle Notebooks

Similar to Colab but with different resource limits:

```python
# Setup in Kaggle
!pip install -r requirements.txt
!python run_batch_simulations.py --runs 10 --checkpoint
```

**Kaggle Advantages**:
- Longer session times (9-12 hours)
- More consistent resources
- Built-in dataset management

### Azure Notebooks

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/Meshtasticator.git
cd Meshtasticator
pip install -r requirements.txt

# Run with checkpoints
python run_batch_simulations.py --runs 15 --checkpoint
```

### AWS SageMaker / Cloud9

```bash
# Same commands work on AWS platforms
python run_batch_simulations.py --runs 50 --checkpoint
```

### Local Docker

For consistent environment across platforms:

```dockerfile
FROM python:3.8

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "run_batch_simulations.py", "--runs", "10", "--checkpoint"]
```

Build and run:
```bash
docker build -t meshtasticator .
docker run -v $(pwd)/results:/app/batch_results meshtasticator
```

---

## 🔧 Troubleshooting

### Session Disconnected Mid-Simulation

**Problem**: Colab disconnected after 2 hours

**Solution** (with auto-sync enabled):
```python
# Re-mount Drive and navigate to project
from google.colab import drive
drive.mount('/content/drive')
%cd Meshtasticator

# Resume from last checkpoint with auto-sync
!python run_batch_simulations.py --resume batch_results_TIMESTAMP --auto-sync
```

**Your completed runs are safe in Google Drive!** ✅

### Out of Memory Errors

**Problem**: `MemoryError` during simulation

**Solutions**:

1. **Reduce number of runs**:
   ```bash
   python run_batch_simulations.py --runs 5 --checkpoint  # Instead of 20
   ```

2. **Reduce network size** in `topology_params.yaml`:
   ```yaml
   num_nodes: 20  # Instead of 50
   ```

3. **Test fewer routing protocols**:
   ```bash
   python run_batch_simulations.py --runs 10 --routing AODV --checkpoint
   ```

### Checkpoint Not Found

**Problem**: "No checkpoint found" error

**Solution**: Verify the directory name:
```bash
# List all batch directories
ls -d batch_results_*/

# Use exact name
python run_batch_simulations.py --resume batch_results_20260214_120000
```

### Simulations Running Too Slow

**Problem**: Simulations taking too long

**Solutions**:

1. **Reduce simulation duration** in config files
2. **Use smaller topologies**
3. **Upgrade to Colab Pro** for better resources
4. **Run on more powerful platform** (Kaggle, AWS, etc.)

### Dependencies Not Installing

**Problem**: `pip install` fails

**Solution**:
```bash
# Update pip first
pip install --upgrade pip setuptools

# Install dependencies one by one
pip install simpy numpy matplotlib pandas PyYAML

# Verify installation
python -c "import simpy, numpy, matplotlib; print('OK')"
```

---

## 💡 Tips for Long Simulations

### 1. Always Enable Checkpoints

```bash
# Always use --checkpoint for runs > 5
python run_batch_simulations.py --runs 20 --checkpoint
```

### 2. Always Enable Auto-Sync on Colab

```bash
# BEST PRACTICE: Use checkpoint + auto-sync together
python run_batch_simulations.py --runs 20 --checkpoint --auto-sync
```

**Why?**
- Results backed up after EACH run
- Zero data loss if disconnection happens
- Can resume with all previous work intact
- Only ~1-5 seconds overhead per run

### 3. Save to Google Drive

```python
# In Colab, always mount Drive first
from google.colab import drive
drive.mount('/content/drive')

# Then use --auto-sync flag for automatic backup
# Results continuously synced to /content/drive/MyDrive/Meshtasticator_Results/
```

### 4. Monitor Progress

Keep an eye on:
- Terminal output for current run status
- Sync status messages `[SYNC] ✓ Backed up to Drive`
- `checkpoint.json` for overall progress
- System resources (RAM, disk space)
- Google Drive space (if using auto-sync)

### 5. Batch Size Strategy

For very long simulations:

```bash
# Instead of 100 runs at once
# Do 5 batches of 20 runs

# Batch 1
python run_batch_simulations.py --runs 20 --checkpoint --auto-sync --output batch_001

# Batch 2
python run_batch_simulations.py --runs 20 --checkpoint --auto-sync --output batch_002

# ... and so on
```

### 6. Use Colab Pro for Long Runs

**Free Colab**:
- ~2-4 hour session limit
- Periodic disconnections
- Good for: 5-15 runs
- **With auto-sync**: Can handle 20-30 runs across sessions

**Colab Pro**:
- ~24 hour session limit  
- More stable connections
- Better resources
- Good for: 20-100 runs
- **With auto-sync**: Can handle 100+ runs easily

### 7. Test First, Then Scale

```bash
# Test with small parameters first
python run_batch_simulations.py --runs 2 --checkpoint --auto-sync

# Once confirmed working, scale up
python run_batch_simulations.py --runs 50 --checkpoint --auto-sync
```

### 8. Keep Browser Tab Active

Some platforms (like Colab) may pause execution if the tab is inactive for too long.

**Workaround in Colab**:
```javascript
// Run this in browser console (F12)
function ClickConnect(){
  console.log("Keeping connection alive");
  document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect, 60000)  // Every minute
```

### 9. Estimate Time Requirements

Before starting, estimate:

```python
# Rough estimation
runs = 20
routing_types = 3
minutes_per_run_per_routing = 5

total_minutes = runs * routing_types * minutes_per_run_per_routing
total_hours = total_minutes / 60

print(f"Estimated time: {total_hours:.1f} hours")
```

### 10. Parallel Batches on Multiple Platforms

For very large experiments:

- Run Batch A on Colab
- Run Batch B on Kaggle
- Run Batch C locally
- Combine results later

---

## 📊 Post-Processing Results

After simulations complete:

### Download Results from Colab

**Note**: If you used `--auto-sync`, your results are already in Google Drive!

Check: `/content/drive/MyDrive/Meshtasticator_Results/batch_results_*/`

To download locally anyway:

```python
# Create zip file
!zip -r results.zip batch_results_*

# Download
from google.colab import files
files.download('results.zip')
```

### Analyze Results

```bash
# Compare routing protocols
python compare_reliability.py
python compare_delays.py

# Visualize topologies
python plot_topology.py

# Analyze batches
python analyze_batch_results.py
```

---

## 📝 Summary Checklist

Before running on Colab:

- [ ] Repository cloned
- [ ] Google Drive mounted (highly recommended)
- [ ] Dependencies installed
- [ ] Topology parameters configured
- [ ] Checkpoint support enabled (`--checkpoint`)
- [ ] **Auto-sync enabled (`--auto-sync`)** ← NEW!
- [ ] Estimated time calculated
- [ ] Browser tab will stay active

For resuming:

- [ ] Checkpoint directory identified
- [ ] Using `--resume` flag
- [ ] **Auto-sync re-enabled (`--auto-sync`)** ← Don't forget!
- [ ] Same Python environment restored

---

## 🆘 Need Help?

Common issues and solutions:

| Issue | Solution |
|-------|----------|
| Session timeout | Use `--checkpoint --auto-sync` and `--resume` |
| Out of memory | Reduce nodes or runs |
| Import errors | Run `pip install -r requirements.txt` |
| Can't find checkpoint | Use exact directory name with `--resume` |
| Too slow | Reduce simulation time or network size |
| Sync not working | Check Drive is mounted and accessible |
| Results not in Drive | Verify `--auto-sync` flag was used |

For more help, see:
- [Main README](README.md)
- [Discrete Event Simulation Guide](DISCRETE_EVENT_SIM.md)
- [Interactive Simulation Guide](INTERACTIVE_SIM.md)

---

**Happy Simulating! 🎉**
