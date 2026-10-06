#!/bin/bash
# One-screen Histo progress: tiers done/total, running/failed tasks, queue.
cd /nesi/project/aut04653/Manj/Histo/retrain_grouped
count() { local d=0 n=0; while read M S U; do n=$((n+1)); [ -f runs/$M/seed$S/$U/result.json ] && d=$((d+1)); done < <(sed -n "$1"p tasks.txt); echo "$d/$n"; }
cls=$(ls runs/{svm,rf,knn}/seed*/*/result.json 2>/dev/null | wc -l)
echo "tier1 finals $(count 1,24) | tier2 seed42 CV $(count 25,64) | classical $cls/54"
echo "running: $(for l in locks/*/owner; do [ -f "$l" ] || continue; t=$(basename $(dirname $l)); [[ $t == FAILED_* ]] && continue; M=${t%%_*}; r=${t#*_}; S=${r%%_*}; U=${r#*_}; [ -f runs/$M/seed$S/$U/result.json ] || echo -n "$t "; done)"
echo "failed: $(ls -d locks/FAILED_* 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"
squeue -u $USER -h -o "%i %j %q %b %T %R" | grep histo | awk '{print $2, $3, $4, $5}' | sort | uniq -c
