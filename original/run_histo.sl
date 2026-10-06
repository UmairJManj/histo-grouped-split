#!/bin/bash -e
#SBATCH --account=aut04653
#SBATCH --job-name=histo_ensemble
#SBATCH --time=18:00:00
#SBATCH --partition=genoa,milan        # span both GPU partitions
#SBATCH --gpus-per-node=L4:1           # L4 (24GB, bf16) = safe for BATCH_SIZE=16 tri-backbone ~167M params
#SBATCH --cpus-per-task=4              # matches NUM_WORKERS 4
#SBATCH --mem=32G
#SBATCH --output=/nesi/nobackup/aut04653/jfd1673/histo_runs/slurm-%j.out

# Tri-backbone histopathology ensemble (ResNet50 + InceptionResNetV2 + ViT-B/16,
# attention fusion). Converted from tissue_ensemble_fixed_(1).ipynb.
# EPOCHS raised 10 -> 40; unchanged early-stopping (patience 20) + best-val
# checkpoint restore mean the returned model is still the best val epoch, so the
# extra epochs only add room to improve, they don't degrade the proven result.

# PyTorch 2.x venv (cu121 wheels bundle CUDA; torchstain installed for Macenko)
source /nesi/project/aut04653/Manj/uav_env/bin/activate

# timm pretrained weights are PRE-CACHED here (compute nodes have no internet)
export HF_HOME=/nesi/nobackup/aut04653/jfd1673/hf_cache
export HF_HUB_OFFLINE=1                 # never reach out to the Hub at runtime

OUT=/nesi/nobackup/aut04653/jfd1673/histo_runs/run_${SLURM_JOB_ID}
mkdir -p "$OUT"

export HISTO_DATA_DIR=/nesi/nobackup/aut04653/jfd1673/histo_runs/data
export HISTO_OUTPUT_DIR="$OUT"
export HISTO_EPOCHS=40

nvidia-smi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
python -c "import torch; print('torch', torch.__version__, 'cuda avail', torch.cuda.is_available())"
echo "DATA=$HISTO_DATA_DIR  OUT=$HISTO_OUTPUT_DIR  EPOCHS=$HISTO_EPOCHS"

srun python -u "/nesi/project/aut04653/Manj/Human Histopathological/train_histo.py"
