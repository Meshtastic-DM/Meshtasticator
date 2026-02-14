# 🚀 Quick Start: Run on Google Colab in 5 Minutes

## Method 1: Use the Notebook (Easiest)

1. Upload `Meshtasticator_Colab.ipynb` to Google Drive
2. Open with Google Colaboratory
3. Click "Runtime" → "Run all"
4. Done! ✅

## Method 2: Manual Commands

### Step 1: Setup (Run once)

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repository (replace with your GitHub URL)
!git clone https://github.com/YOUR_USERNAME/Meshtasticator.git
%cd Meshtasticator

# Install dependencies
!pip install -q -r requirements.txt
```

### Step 2: Run Simulations

```bash
# Basic run with auto-sync to Drive (RECOMMENDED)
!python run_batch_simulations.py --runs 10 --checkpoint --auto-sync

# Manual Drive path
!python run_batch_simulations.py --runs 10 --checkpoint --sync-to-drive /content/drive/MyDrive/Results

# Without Drive sync (less safe)
!python run_batch_simulations.py --runs 10 --checkpoint

# With specific routing protocols
!python run_batch_simulations.py --runs 10 --routing AODV MANAGED_FLOOD --checkpoint --auto-sync

# Custom parameters
!python run_batch_simulations.py --runs 5 --params topology_params.yaml --checkpoint --auto-sync
```

### Step 3: Resume if Disconnected

```bash
# List batch directories
!ls -d batch_results_*/

# Resume from specific batch
!python run_batch_simulations.py --resume batch_results_20260214_120000
```

### Step 4: Download Results

```python
# Note: If you used --auto-sync, results are already in Google Drive!
# Check: /content/drive/MyDrive/Meshtasticator_Results/

# To download locally anyway:
# Create zip
!zip -r results.zip batch_results_*

# Download
from google.colab import files
files.download('results.zip')
```

## Common Options

| Flag | Purpose | Example |
|------|---------|---------|
| `--runs N` | Number of simulation runs | `--runs 20` |
| `--routing` | Routing protocols to test | `--routing AODV ZRP` |
| `--checkpoint` | Enable resume capability | `--checkpoint` |
| `--auto-sync` | Auto-sync to Drive (Colab) | `--auto-sync` |
| `--sync-to-drive` | Sync to specific Drive path | `--sync-to-drive /path/to/drive` |
| `--resume` | Resume interrupted batch | `--resume batch_results_20260214_120000` |
| `--params` | Custom topology file | `--params my_topology.yaml` |

## Time Estimates

| Runs | Routing Types | Estimated Time |
|------|---------------|----------------|
| 5 | 1 | ~15-30 min |
| 10 | 2 | ~30-60 min |
| 20 | 3 | ~1-2 hours |
| 50 | 3 | ~3-5 hours |

## Important Tips

✅ **Always use `--checkpoint`** for runs > 5  
✅ **Always use `--auto-sync`** on Colab (results backed up after each run!)  
✅ **Mount Google Drive** to enable auto-sync  
✅ **Keep browser tab active** to prevent disconnection  
✅ **Use Colab Pro** for long runs (>20 runs)  

## Troubleshooting

**Session disconnected?**
→ Re-run setup cells, then use `--resume`

**Out of memory?**
→ Reduce `--runs` or nodes in `topology_params.yaml`

**Too slow?**
→ Reduce simulation time or network size

**Need help?**
→ See [COLAB_GUIDE.md](COLAB_GUIDE.md) for detailed instructions

---

That's it! You're ready to run simulations on Colab. 🎉
