#!/bin/bash -e
#SBATCH --account=aut04653
#SBATCH --job-name=histo_cpusmoke
#SBATCH --time=01:00:00
#SBATCH --partition=milan
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=/nesi/project/aut04653/Manj/Histo/retrain_grouped/logs/smoke-cpu-%j.out
# CPU smoke of every model path (tiny data, 1 epoch) into smoke_runs/.
source /nesi/project/aut04653/Manj/Histo/retrain_grouped/common_env.sh
export HISTO_RUNS_DIR=smoke_runs HISTO_CKPT_DIR=/nesi/nobackup/aut04653/jfd1673/histo_runs/grouped_smoke
export OMP_NUM_THREADS=16
for m in ${SMOKE_MODELS:-fusion meanfusion vit resnet inception vgg16 densenet121 efficientnet_b0}; do
  $PY train_grouped.py --model $m --unit final --seed 43 --epochs 1 --limit 32 && echo "SMOKE_OK $m" || echo "SMOKE_FAIL $m"
done
[ -z "$SMOKE_MODELS" ] && $PY train_grouped.py --model resnet --unit fold3 --seed 44 --epochs 1 --limit 32 && echo "SMOKE_OK resnet-fold" || echo "SMOKE_FAIL resnet-fold"
