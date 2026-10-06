# Shared environment for the Histo grouped retrain (sourced by the .sl files).
H=/nesi/project/aut04653/Manj/Histo
export LD_LIBRARY_PATH=$H/.venv/lib:$LD_LIBRARY_PATH
export HF_HOME=/nesi/nobackup/aut04653/jfd1673/hf_cache
export HF_HUB_OFFLINE=1
export HISTO_DATA_DIR=/nesi/nobackup/aut04653/jfd1673/histo_runs/data
export HISTO_OUTPUT_DIR=/nesi/nobackup/aut04653/jfd1673/histo_runs/grouped_ckpt
export HISTO_EPOCHS=40
export PYTHONUNBUFFERED=1
PY=$H/.venv/bin/python
cd $H/retrain_grouped
export HISTO_PATIENCE=10
