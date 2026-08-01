#!/bin/bash
# GPU correctness gate — the Phase 0 hard gate for the serving layer.
#
# Runs tests/test_gpu_oracle.py, which is the first thing in this repo to verify
# that the GPU fp16 model produces CORRECT output rather than merely finite,
# correctly-shaped output. Generates the HF oracle fixtures first if absent.
#
#     sbatch scripts/gpu_oracle.sbatch
#     squeue -u $USER
#     tail -f logs/gpu_oracle_<jobid>.out
#
# QOS is `embers`: free and preemptible. Correct for a correctness run, since no
# published number comes from it. Benchmarks use `inferno`.
#
#SBATCH --job-name=gpu_oracle
#SBATCH --account=paceship-simpliearn
#SBATCH --qos=embers
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/gpu_oracle_%j.out
#SBATCH --error=logs/gpu_oracle_%j.out

set -e
mkdir -p logs

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID}  qos: ${SLURM_JOB_QOS} ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

module load cuda/12.9.1
module load anaconda3
conda activate llm

cd "${SLURM_SUBMIT_DIR}"
export HF_HUB_OFFLINE=1          # weights are local; compute nodes have no internet

echo
echo "=== oracle fixtures ==="
if [ ! -f tests/fixtures/oracle_short.pkl ] || [ ! -f tests/fixtures/oracle_medium.pkl ]; then
    echo "Generating (HF fp32 on CPU, a few minutes)..."
    python3 tests/oracle.py
else
    echo "Present, skipping generation."
fi

echo
echo "=== THE GATE: GPU fp16 vs fp32 oracle ==="
python3 -m pytest tests/test_gpu_oracle.py -v -s 2>&1

echo
echo "=== regression: existing GPU tests still pass ==="
python3 -m pytest tests/test_gpu_model.py tests/test_components_gpu.py -q 2>&1 | tail -5

echo
echo "=== BatchMeta invariants (CPU, sanity) ==="
python3 -m pytest tests/test_attention_backend.py -q 2>&1 | tail -3

echo
echo "=== done ==="
