"""
Decode-attention kernel microbenchmark (Phase 4.4).

Times v1/v2/v3 kernels and PyTorch SDPA on decode-shaped inputs (one query token,
varying kv_seq), using CUDA events for accurate GPU timing. Writes a CSV row per
(version, kv_seq).

Usage (PACE):
    python -m bench.bench_attn_kernel
"""

import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "build"))
sys.path.insert(0, str(_ROOT / "kernels"))

N_HEADS, N_KV, HEAD_DIM = 32, 8, 64
SCALE   = 1.0 / (HEAD_DIM ** 0.5)
GROUPS  = N_HEADS // N_KV
DEV     = "cuda:0"
KV_SEQS = [128, 512, 1024, 2048]
N_ITERS = 200
N_WARMUP = 20


def _time_cuda(fn, n_iters=N_ITERS, n_warmup=N_WARMUP) -> float:
    """Return mean latency in microseconds using CUDA events."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / n_iters   # ms->us, per iter


def _sdpa(q, k, v):
    """PyTorch reference: scaled_dot_product_attention on decode shapes.
    q: (n_heads,1,head_dim)  k,v: (n_heads,kv_seq,head_dim) after GQA expand."""
    return F.scaled_dot_product_attention(q, k, v)


def main():
    from attn_reference import attention_decode

    rows = []
    print(f"{'kv_seq':>8} {'v1(us)':>10} {'v2(us)':>10} {'v3(us)':>10} {'sdpa(us)':>10} {'v3 vs sdpa':>12}")

    for kv_seq in KV_SEQS:
        q = (torch.randn(N_HEADS, HEAD_DIM) * 0.1).half().to(DEV)
        k = (torch.randn(N_KV, kv_seq, HEAD_DIM) * 0.1).half().to(DEV)
        v = (torch.randn(N_KV, kv_seq, HEAD_DIM) * 0.1).half().to(DEV)

        t = {}
        for ver in ["v1", "v2", "v3"]:
            t[ver] = _time_cuda(lambda ver=ver: attention_decode(q, k, v, SCALE, version=ver))

        # SDPA on expanded GQA shapes: (n_heads, 1, head_dim) x (n_heads, kv_seq, head_dim)
        q_s = q.unsqueeze(1)                                   # (n_heads,1,head_dim)
        k_s = k.repeat_interleave(GROUPS, dim=0)               # (n_heads,kv_seq,head_dim)
        v_s = v.repeat_interleave(GROUPS, dim=0)
        t["sdpa"] = _time_cuda(lambda: _sdpa(q_s, k_s, v_s))

        speedup = t["sdpa"] / t["v3"]
        print(f"{kv_seq:>8} {t['v1']:>10.2f} {t['v2']:>10.2f} {t['v3']:>10.2f} "
              f"{t['sdpa']:>10.2f} {speedup:>11.2f}x")

        for ver in ["v1", "v2", "v3", "sdpa"]:
            rows.append({"kv_seq": kv_seq, "version": ver, "latency_us": round(t[ver], 3),
                         "v3_vs_sdpa_speedup": round(speedup, 3)})

    out_dir = _ROOT / "bench" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "attn_kernel_microbench.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
