#!/bin/bash
#SBATCH --account=aut04653
#SBATCH --job-name=histo_lane
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/nesi/project/aut04653/Manj/Histo/retrain_grouped/logs/lane-%j.out
# Work-queue lane: claims the next unclaimed (model, seed, unit[, variant]) from the
# task files in order (improvement variants, tier-1 finals, baselines under the
# chosen variant, tier-2 CV)
# with an atomic mkdir lock and runs it, until none are left or <40 min remain.
# Submit with --partition/--gpus-per-node for any GPU type.
source /nesi/project/aut04653/Manj/Histo/retrain_grouped/common_env.sh
TASK_FILES=${TASK_FILES:-"improve_tasks.txt tasks_t1.txt baseline_variant_tasks.txt tasks_t2.txt"}
mkdir -p locks
END=$(( $(date +%s) + $(squeue -h -j $SLURM_JOB_ID -o %L | awk -F'[-:]' '{n=NF; s=$n+60*$(n-1); if(n>2)s+=3600*$(n-2); if(n>3)s+=86400*$(n-3); print s}') ))
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "lane $SLURM_JOB_ID on $(hostname) GPU=$GPU"
# a requeued job keeps its id: drop locks it held in its previous run
for o in locks/*/owner; do [ -f "$o" ] && [ "$(awk '{print $1}' "$o")" = "$SLURM_JOB_ID" ] && [ -z "${KEEP_OWN_LOCKS:-}" ] && rm -rf "$(dirname "$o")" && echo "=== dropped own stale lock $(dirname "$o")"; done
MIN_LEFT=${MIN_LEFT:-2400}   # seconds of walltime needed to start a task
for TF in $TASK_FILES; do
while read M S U V; do
  V=${V:-base}; if [ $V != base ]; then R=$V/runs; export HISTO_VARIANT=$V; else R=runs; unset HISTO_VARIANT; fi
  need=$MIN_LEFT; [ $V = base ] && [ $need -lt 1800 ] && need=1800   # base runs do not resume: need 30 min
  [ $(( END - $(date +%s) )) -lt $need ] && { echo "less than $((need / 60)) min left for $M $S $U $V, skipping"; continue; }
  [ -f $R/$M/seed$S/$U/result.json ] && continue
  L=locks/${M}_${S}_${U}$([ $V != base ] && echo _$V)
  if ! mkdir $L 2>/dev/null; then
    # reclaim a lock whose owner job is no longer running (crash, preemption, timeout)
    owner=$(awk '{print $1}' $L/owner 2>/dev/null)
    if [ -n "$owner" ] && ! squeue -h -j "$owner" -t RUNNING 2>/dev/null | grep -q .; then
      rm -rf $L; mkdir $L 2>/dev/null || continue; echo "=== reclaimed stale lock of $owner"
    else continue; fi
  fi
  echo "${OWNER_ID:-$SLURM_JOB_ID} $GPU $(date +%FT%T)" > $L/owner
  echo "=== $(date +%T) start $M seed$S $U $V"
  export HISTO_CKPT_DIR=$HISTO_OUTPUT_DIR/$V/$M/seed$S/$U
  if $PY train_grouped.py --model $M --seed $S --unit $U; then
    echo "=== $(date +%T) done $M seed$S $U $V"
  else
    echo "=== $(date +%T) FAILED $M seed$S $U $V"; mv $L locks/FAILED_$V${M}_${S}_${U}_$SLURM_JOB_ID
  fi
done < $TF
done
