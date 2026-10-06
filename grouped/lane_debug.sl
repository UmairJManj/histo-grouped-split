#!/bin/bash
#SBATCH --account=aut04653
#SBATCH --job-name=histo_dbg
#SBATCH --qos=debug
#SBATCH --time=01:00:00
#SBATCH --partition=genoa
#SBATCH --gpus-per-node=L4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/nesi/project/aut04653/Manj/Histo/retrain_grouped/logs/lane-debug-%j.out
# High-priority debug lane: one task-claiming loop per allocated GPU, sharing
# tasks.txt with the normal lanes (same mkdir locks). Self-requeues once if
# work remains, giving a fresh 1 h debug window (debug MaxWall is 1 h) under the same job id.
cd /nesi/project/aut04653/Manj/Histo/retrain_grouped
export SLURM_CPUS_PER_TASK=4
# Improvement variants first (they resume from per-epoch state across requeues),
# then tier-1/tier-2. A guard requeues the job 4 min before the 1 h wall.
export TASK_FILES="improve_tasks.txt tasks_t1.txt baseline_variant_tasks.txt tasks_t2.txt"
export MIN_LEFT=600
( sleep 3360 && echo "[guard] requeue before wall" && scontrol requeue $SLURM_JOB_ID ) &
GUARD=$!
NG=$(nvidia-smi -L | wc -l)
[ "$NG" -gt 1 ] && export KEEP_OWN_LOCKS=1   # sibling loops share the job id
for g in $(seq 0 $((NG - 1))); do
  CUDA_VISIBLE_DEVICES=$g bash lane.sl >> logs/lane-debug-${SLURM_JOB_ID}-gpu$g.out 2>&1 &
  PIDS="$PIDS $!"
done
wait $PIDS
kill $GUARD 2>/dev/null
left=$(while read M S U; do [ -f runs/$M/seed$S/$U/result.json ] || echo x; done < ${TASKS:-tasks.txt} | wc -l)
echo "$(date +%T) tasks left: $left"
[ "$left" -gt 0 ] && scontrol requeue $SLURM_JOB_ID
