# Final Experiment Setup - Working Version

## ✅ Fixed: Continuous Logging Issue

**Problem:** Original script redirected output to files only (`> log 2>&1`), causing:
- No terminal output during execution
- Process appears to hang
- Not visible in `ps aux`

**Solution:** Use `tee` for **simultaneous** output to both terminal and log files:
```bash
python script.py 2>&1 | tee logfile.log
exec > >(tee -a main.log) 2>&1  # Redirect all script output
```

## Working Scripts

### 1. Simple Test (Fastest)
```bash
./scripts/run_experiments_simple.sh
```
- Runs Byzantine detection only (~5 seconds)
- Continuous terminal output
- Logs saved to `outputs/experiments_TIMESTAMP/`

### 2. Full Suite (Recommended)
```bash
./scripts/run_all_experiments.sh
```
- All experiments with continuous logging
- Resumable via checkpoint file
- Full logs in real-time

### 3. Byzantine Detection Only
```bash
python scripts/run_benchmark_suite.py \
    --benchmarks byzantine \
    --output outputs/results
```

## Output Structure

```
outputs/experiments_YYYYMMDD_HHMMSS/
├── main.log                    # ✅ CONTINUOUS log of entire run
├── completed.txt               # Checkpoint file
├── logs/
│   ├── exp1_byzantine.log     # Per-experiment logs
│   ├── exp2_defense.log
│   └── ...
└── results/
    ├── benchmark_*.json       # JSON results
    └── ...
```

## Real-Time Monitoring

### Watch live logs
```bash
# Main log (all experiments)
tail -f outputs/experiments_*/main.log

# Specific experiment
tail -f outputs/experiments_*/logs/exp1_byzantine.log
```

### Check progress
```bash
# How many experiments completed?
cat outputs/experiments_*/completed.txt | wc -l

# List completed
cat outputs/experiments_*/completed.txt
```

### View results
```bash
# List all result files
ls -lh outputs/experiments_*/results/*.json

# View specific result
cat outputs/experiments_*/results/benchmark_*.json | jq .
```

## What Was Fixed

### Before (broken):
```bash
python script.py > log.txt 2>&1  # No terminal output!
```

### After (working):
```bash
# Method 1: Per-command
python script.py 2>&1 | tee log.txt

# Method 2: Entire script
exec > >(tee -a main.log) 2>&1
python script.py  # Auto-logged + shown on terminal
```

## Experiment Results

### Byzantine Detection (Working ✅)

**Latest Run:** `outputs/experiments_20260330_163235/`

Results:
```
LINEAR: 100% detection in 1.0ms
MLP:    100% detection in 0.76ms  
CNN:    100% detection in 0.93ms
```

**Log file:** Captures everything in real-time
**JSON results:** Structured data for paper tables

## Key Features

1. **Continuous Logging:** See output as it happens
2. **Dual Output:** Terminal + log files simultaneously
3. **Resumable:** Checkpoint system tracks progress
4. **Debug Mode:** `set -x` shows each command execution
5. **Error Handling:** Captures exit codes and failures

## For Paper Results

All experiments log to JSON files with:
- Byzantine detection rates
- Detection times
- Fingerprint differences
- Model architectures
- Timestamps

Extract results:
```bash
# Get all Byzantine detection results
jq '.benchmarks.byzantine_detection' outputs/experiments_*/results/*.json

# Compile into table
python scripts/compile_paper_results.py outputs/experiments_*/results
```

## Troubleshooting

### No output appearing?
- Check if `tee` is in the command
- Verify log file exists: `ls -la outputs/experiments_*/main.log`
- Try simple version first: `./scripts/run_experiments_simple.sh`

### Process hangs?
- Use Ctrl+C to stop
- Check last log entry: `tail outputs/experiments_*/main.log`
- Run Python script directly to debug

### Can't find results?
```bash
# Find all experiment runs
ls -d outputs/experiments_*/

# Find all results
find outputs/ -name "*.json" -type f
```

## Summary

**Working scripts:**
- ✅ `run_experiments_simple.sh` - Quick test
- ✅ `run_all_experiments.sh` - Full suite with continuous logging
- ✅ `run_benchmark_suite.py` - Direct Python execution

**All logs are now continuous and visible in real-time!**
