#!/bin/bash -e
#SBATCH --account=aut04653
#SBATCH --job-name=histo_classical
#SBATCH --time=04:00:00
#SBATCH --partition=milan
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/nesi/project/aut04653/Manj/Histo/retrain_grouped/logs/classical-%j.out
# Classical ML baselines (SVM / RF / k-NN on GLCM+LBP+colour features), all seeds and units.
# SMOKE=1 runs the first 3 units into smoke_runs/.
source /nesi/project/aut04653/Manj/Histo/retrain_grouped/common_env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
if [ "${SMOKE:-0}" = 1 ]; then export HISTO_RUNS_DIR=smoke_runs; $PY classical_ml.py --limit 3
else $PY classical_ml.py ${SEEDS:+--seeds $SEEDS}; fi
