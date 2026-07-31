#!/bin/bash
#SBATCH --job-name=llm-bench
#SBATCH --account=paceship-simpliearn
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=bench/logs/bench_%j.out
#SBATCH --error=bench/logs/bench_%j.out
#
# Full benchmark capture on one A100 allocation.
#
# Everything that must be compared runs inside THIS job, on THIS node —
# interactive A100 nodes vary in contention and clocks, so absolute tok/s is
# only comparable within a single allocation.
#
#   sbatch scripts/pace_bench.sbatch
#   squeue -u $USER
#   tail -f bench/logs/bench_<jobid>.out
#
# Prerequisite: bench/wikitext2_test.txt must already exist (fetch it from the
# LOGIN node — compute nodes may have no outbound internet):
#   python scripts/fetch_wikitext2.py

set -u   # not -e: one failing stage should not abandon the rest

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p bench/logs bench/results

module load cuda/12.9.1
module load anaconda3
source activate llm

export HF_HOME="$HOME/ps-simpliearn-0/.hf_cache"

echo "=============================================================="
echo " job      : ${SLURM_JOB_ID}"
echo " node     : $(hostname)"
echo " commit   : $(git log --oneline -1)"
echo " started  : $(date -Is)"
echo "=============================================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch, transformers; print(f'torch {torch.__version__}  cuda {torch.version.cuda}  transformers {transformers.__version__}')"
echo

run () {                      # run <label> <command...>
    local label="$1"; shift
    echo "-------------------------------------------------------------"
    echo ">>> ${label}"
    echo ">>> $*"
    echo "-------------------------------------------------------------"
    local t0=$SECONDS
    if "$@"; then
        echo "<<< ${label}: OK ($((SECONDS - t0))s)"
    else
        echo "<<< ${label}: FAILED rc=$? ($((SECONDS - t0))s)"
    fi
    echo
}

# ---------------------------------------------------------------------------
# 1. Throughput baselines — engine vs HF, back to back on this node.
#    attn-impl is pinned explicitly; the June 5 run left it to the library
#    default and never recorded which one it got.
# ---------------------------------------------------------------------------
run "engine baseline (fp16)" \
    python bench/harness.py --backend gpu --max-tokens 128 --n-runs 3

run "HF baseline (fp16, sdpa pinned)" \
    python bench/baseline_hf.py --max-tokens 128 --n-runs 3 --attn-impl sdpa

# ---------------------------------------------------------------------------
# 2. WikiText-2 perplexity — the genuinely new measurement.
#    Sliding window 512 / stride 256: each scored token gets >=256 tokens of
#    left context, which is the standard protocol.
# ---------------------------------------------------------------------------
WIKI=bench/wikitext2_test.txt
if [[ -f "$WIKI" ]]; then
    for mode in fp16 int8 int4; do
        run "perplexity ${mode} (WikiText-2, w=512 s=256)" \
            python -m bench.perplexity --mode "$mode" --text "$WIKI" \
                   --window 512 --stride 256 --max-tokens 100000
    done
else
    echo "!!! ${WIKI} missing — fetch it from the login node:"
    echo "!!!   python scripts/fetch_wikitext2.py"
fi

echo "=============================================================="
echo " finished : $(date -Is)"
echo " elapsed  : ${SECONDS}s"
echo "=============================================================="
ls -la bench/results/
