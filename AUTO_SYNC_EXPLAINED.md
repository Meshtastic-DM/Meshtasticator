# 🔄 Auto-Sync Feature - How It Works

## What is Auto-Sync?

Auto-sync automatically backs up your simulation results to Google Drive **after each completed run**, ensuring zero data loss even if your Colab session disconnects.

## Visual Comparison

### ❌ Traditional Approach (WITHOUT Auto-Sync)

```
Session Start
    ↓
Run 1 → Run 2 → Run 3 → ... → Run 20
    ↓
[Manual Copy to Drive?]
    ↓
[Session Disconnects] ← 💥 ALL PROGRESS LOST!
```

### ✅ Auto-Sync Approach (WITH Auto-Sync)

```
Session Start
    ↓
Run 1 → [✓ Synced to Drive]
    ↓
Run 2 → [✓ Synced to Drive]
    ↓
Run 3 → [✓ Synced to Drive]
    ↓
[Session Disconnects] ← ✅ All 3 runs safe in Drive!
    ↓
[New Session - Resume]
    ↓
Run 4 → [✓ Synced to Drive]
    ↓
Run 5 → [✓ Synced to Drive]
    ↓
...continues until complete
```

## How to Enable

### On Google Colab:

```bash
# Mount Google Drive first
from google.colab import drive
drive.mount('/content/drive')

# Run with auto-sync
python run_batch_simulations.py --runs 20 --checkpoint --auto-sync
```

That's it! Results will automatically sync after each run.

## What Gets Synced?

After each completed run, the following are synced to Google Drive:

```
/content/drive/MyDrive/Meshtasticator_Results/
└── batch_results_20260214_120000/
    ├── checkpoint.json                    ← Progress tracker
    ├── .checkpoint.pkl                    ← Binary checkpoint
    ├── topology_params_used.yaml          ← Configuration
    ├── batch_summary.json                 ← Summary
    ├── run_001/
    │   ├── config.yaml
    │   ├── run_metadata.json
    │   ├── AODV/
    │   │   ├── sensor_packets_AODV.csv    ← Simulation data
    │   │   ├── dm_packets_AODV.csv
    │   │   ├── broadcast_packets_AODV.csv
    │   │   └── *.png                       ← Plots
    │   ├── MANAGED_FLOOD/
    │   │   └── ...
    │   └── BL_A_AODV/
    │       └── ...
    ├── run_002/
    │   └── ...
    └── ...
```

**Everything** is backed up continuously!

## Sync Timing

```
Simulation Timeline (per run):
├─ [00:00] Run starts
├─ [02:30] Simulation completes
├─ [02:31] Checkpoint saved locally
├─ [02:32] 🔄 Syncing to Drive... (1-5 seconds)
├─ [02:37] ✓ Sync complete!
└─ [02:38] Next run starts
```

**Overhead**: Only 1-5 seconds per run (2-3% of runtime)

## Sync Efficiency

- **Incremental**: Only new/changed files are copied
- **Fast**: Uses efficient file operations
- **Automatic**: No manual intervention needed
- **Safe**: Preserves file timestamps and metadata

## Example Output

When auto-sync is working, you'll see:

```
======================================================================
RUN 5/20 (Progress: 4/20 completed)
======================================================================
  Generating topology (seed=1005)...
  Running simulation with AODV...
    Moved 15 output files
  Running simulation with MANAGED_FLOOD...
    Moved 15 output files
  Running simulation with BL_A_AODV...
    Moved 15 output files

  [OK] Run 5 completed successfully
  [CHECKPOINT] Progress saved (5/20)
  [SYNC] Syncing to Google Drive...
  [SYNC] ✓ Backed up to Drive: /content/drive/MyDrive/Meshtasticator_Results/batch_results_20260214_120000
```

## Checking Sync Status

Verify your backups are working:

```python
from pathlib import Path
import json

# Check Drive backups
drive_path = Path('/content/drive/MyDrive/Meshtasticator_Results')
batches = list(drive_path.glob('batch_results_*'))

print(f"📊 Total batches backed up: {len(batches)}")

# Check latest batch progress
if batches:
    latest = sorted(batches)[-1]
    checkpoint = latest / 'checkpoint.json'
    
    if checkpoint.exists():
        with open(checkpoint) as f:
            data = json.load(f)
            print(f"\n✅ Latest: {latest.name}")
            print(f"   Progress: {data['progress_percentage']:.1f}%")
            print(f"   Completed: {len(data['completed_runs'])}/{data['total_runs']} runs")
            print(f"   Last updated: {data['last_updated_iso']}")
```

## Resuming with Auto-Sync

If your session disconnects:

1. **Reconnect and mount Drive**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Navigate to project**:
   ```bash
   cd /content/Meshtasticator
   ```

3. **Resume with auto-sync enabled**:
   ```bash
   python run_batch_simulations.py \
       --resume batch_results_20260214_120000 \
       --auto-sync
   ```

Your completed runs are already in Drive, and new runs will continue syncing!

## Troubleshooting

### "Drive not mounted" error

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

### "Sync failed" warning

Check:
- Drive is mounted: `ls /content/drive/MyDrive/`
- Enough Drive space: Check Google Drive quota
- Path is writable: Try creating a test file

### Sync seems slow

- Normal: First run takes longer (full copy)
- Subsequent runs are faster (incremental)
- Network speed dependent
- Typical: 1-5 seconds per run

## Comparison Table

| Feature | Without Auto-Sync | With Auto-Sync |
|---------|------------------|----------------|
| **Data Safety** | ❌ Lost if disconnect | ✅ Always backed up |
| **Manual Work** | ❌ Must copy manually | ✅ Automatic |
| **Resume Capability** | ⚠️ Only if local copy intact | ✅ Always from Drive |
| **Peace of Mind** | ❌ Stressful | ✅ Worry-free |
| **Overhead** | None | 1-5 sec/run (~2%) |
| **Recommended** | No | **Yes!** |

## Best Practices

1. ✅ **Always mount Drive first**
2. ✅ **Use `--auto-sync` for all Colab runs**
3. ✅ **Check sync messages** to verify it's working
4. ✅ **Keep Drive mounted** throughout the session
5. ✅ **Use `--auto-sync` when resuming** too

## Example: Long Experiment

```bash
# Day 1: Start with auto-sync (runs 1-15 before timeout)
python run_batch_simulations.py --runs 50 --checkpoint --auto-sync

# [Colab disconnects after 15 runs]
# ✅ All 15 runs safely backed up in Google Drive

# Day 2: Resume with auto-sync (runs 16-30 before timeout)
python run_batch_simulations.py \
    --resume batch_results_20260214_120000 \
    --auto-sync

# [Colab disconnects after 15 more runs]
# ✅ All 30 runs now in Google Drive

# Day 3: Final resume (runs 31-50)
python run_batch_simulations.py \
    --resume batch_results_20260214_120000 \
    --auto-sync

# ✅ All 50 runs completed and backed up!
```

**Total data loss**: ZERO! 🎉

## FAQ

**Q: Do I need to manually copy results anymore?**  
A: No! Auto-sync handles it automatically after each run.

**Q: What if I forget to use `--auto-sync`?**  
A: Results only saved locally. Risk of loss if session ends.

**Q: Can I use auto-sync on local computer?**  
A: Yes, with `--sync-to-drive /path/to/google/drive/folder`

**Q: Does it slow down simulations?**  
A: Minimal - only 1-5 seconds per run (~2% overhead).

**Q: What if Drive is full?**  
A: Sync will fail with warning, but local results preserved.

**Q: Can I check Drive backups during simulation?**  
A: Yes! Open Drive in browser or use file explorer.

## Summary

| Use Case | Command |
|----------|---------|
| Start new with auto-sync | `--runs 20 --checkpoint --auto-sync` |
| Resume with auto-sync | `--resume DIR --auto-sync` |
| Custom Drive path | `--sync-to-drive /path/to/drive` |
| Check auto-sync status | Watch for `[SYNC] ✓` messages |

---

**Remember**: Always use `--checkpoint --auto-sync` together for maximum safety! 🛡️
