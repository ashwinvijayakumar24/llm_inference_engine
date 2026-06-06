// Decode-attention CUDA kernel (Phase 4.4).
//
// Computes attention for ONE query token (the decode hot path) against the
// full K/V cache. Single query position => NO causal mask needed (the newest
// token legitimately attends to all past positions + itself).
//
// Per query head h:
//   kv_head = h / groups                      (GQA: query heads share KV heads)
//   scores[j] = (q · K[kv_head, j]) * scale   for j in 0..kv_seq-1
//   p = softmax(scores)
//   out[h] = sum_j p[j] * V[kv_head, j]        (length head_dim)
//
// All math accumulates in fp32; inputs/outputs are fp16 (__half).
//
// ---------------------------------------------------------------------------
// v1: one block per head, ONE thread per block. The whole head's attention is
// done by a single thread with serial loops — the CPU reference transcribed to
// one CUDA thread. Uses STREAMING (online) softmax so it needs only head_dim
// floats of state, not kv_seq: maintain running max m, running denom l, and a
// running output accumulator acc[]. This is the Flash-Attention idea in its
// simplest serial form.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

#define MAX_HEAD_DIM 128

// Layouts (row-major, contiguous):
//   Q:   [n_heads, head_dim]
//   K,V: [n_kv_heads, kv_seq, head_dim]
//   out: [n_heads, head_dim]
__global__ void attention_decode_v1_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ out,
    int n_heads, int n_kv_heads, int kv_seq, int head_dim,
    float scale, int groups)
{
    int h = blockIdx.x;                 // one block == one query head
    if (h >= n_heads) return;

    int kv_head = h / groups;           // GQA mapping
    const __half* q   = Q + (size_t)h * head_dim;
    const __half* Kh  = K + (size_t)kv_head * kv_seq * head_dim;
    const __half* Vh  = V + (size_t)kv_head * kv_seq * head_dim;
    __half* o         = out + (size_t)h * head_dim;

    // Load q into registers (fp32).
    float qf[MAX_HEAD_DIM];
    for (int d = 0; d < head_dim; d++) qf[d] = __half2float(q[d]);

    // Streaming softmax state.
    float m   = -INFINITY;              // running max score
    float l   = 0.0f;                   // running sum of exp(score - m)
    float acc[MAX_HEAD_DIM];
    for (int d = 0; d < head_dim; d++) acc[d] = 0.0f;

    for (int j = 0; j < kv_seq; j++) {
        const __half* kj = Kh + (size_t)j * head_dim;
        const __half* vj = Vh + (size_t)j * head_dim;

        // score = (q · K[j]) * scale
        float s = 0.0f;
        for (int d = 0; d < head_dim; d++) s += qf[d] * __half2float(kj[d]);
        s *= scale;

        // Online softmax update.
        float m_new = fmaxf(m, s);
        float corr  = expf(m - m_new);          // rescale prior accumulators
        float p     = expf(s - m_new);          // weight of this position
        l = l * corr + p;
        for (int d = 0; d < head_dim; d++)
            acc[d] = acc[d] * corr + p * __half2float(vj[d]);
        m = m_new;
    }

    // Normalize and write.
    float inv_l = 1.0f / l;
    for (int d = 0; d < head_dim; d++)
        o[d] = __float2half(acc[d] * inv_l);
}

// Host launcher. Pointers are raw device addresses (from torch .data_ptr()).
void launch_attention_decode_v1(
    const __half* Q, const __half* K, const __half* V, __half* out,
    int n_heads, int n_kv_heads, int kv_seq, int head_dim, float scale)
{
    int groups = n_heads / n_kv_heads;
    attention_decode_v1_kernel<<<n_heads, 1>>>(
        Q, K, V, out, n_heads, n_kv_heads, kv_seq, head_dim, scale, groups);
}

// ---------------------------------------------------------------------------
// v2: one block per head, head_dim THREADS per block. Thread d owns output
// element out[d] and holds q[d] in a register. The score q·K[j] is a reduction
// across the head_dim threads, done cooperatively via shared memory + a tree
// reduction. The streaming-softmax scalars (m, l, corr, p) are identical in
// every thread (same inputs), so each thread just updates its own acc[d].
//
// New CUDA concepts vs v1: threadIdx.x, __shared__ memory, __syncthreads(),
// parallel tree reduction.

__global__ void attention_decode_v2_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ out,
    int n_heads, int n_kv_heads, int kv_seq, int head_dim,
    float scale, int groups)
{
    int h = blockIdx.x;                 // one block == one query head
    int d = threadIdx.x;                // one thread == one head dimension
    if (h >= n_heads || d >= head_dim) return;

    int kv_head = h / groups;
    const __half* q   = Q + (size_t)h * head_dim;
    const __half* Kh  = K + (size_t)kv_head * kv_seq * head_dim;
    const __half* Vh  = V + (size_t)kv_head * kv_seq * head_dim;
    __half* o         = out + (size_t)h * head_dim;

    float qd = __half2float(q[d]);      // this thread's slice of q

    __shared__ float sdata[MAX_HEAD_DIM];   // scratch for the per-j reduction

    // Streaming-softmax state (each thread tracks its own copies of the scalars;
    // they stay identical because every thread sees the same reduced score).
    float m = -INFINITY;
    float l = 0.0f;
    float acc = 0.0f;                   // this thread owns acc[d]

    for (int j = 0; j < kv_seq; j++) {
        const __half* kj = Kh + (size_t)j * head_dim;
        const __half* vj = Vh + (size_t)j * head_dim;

        // Cooperative dot product: each thread contributes one term, then a
        // tree reduction sums them into sdata[0].
        sdata[d] = qd * __half2float(kj[d]);
        __syncthreads();
        for (int stride = head_dim >> 1; stride > 0; stride >>= 1) {
            if (d < stride) sdata[d] += sdata[d + stride];
            __syncthreads();
        }
        float s = sdata[0] * scale;     // all threads read the reduced score
        __syncthreads();                // done with sdata before next iteration

        // Online softmax update (identical scalars in every thread).
        float m_new = fmaxf(m, s);
        float corr  = expf(m - m_new);
        float p     = expf(s - m_new);
        l   = l * corr + p;
        acc = acc * corr + p * __half2float(vj[d]);
        m   = m_new;
    }

    o[d] = __float2half(acc / l);
}

void launch_attention_decode_v2(
    const __half* Q, const __half* K, const __half* V, __half* out,
    int n_heads, int n_kv_heads, int kv_seq, int head_dim, float scale)
{
    int groups = n_heads / n_kv_heads;
    attention_decode_v2_kernel<<<n_heads, head_dim>>>(
        Q, K, V, out, n_heads, n_kv_heads, kv_seq, head_dim, scale, groups);
}

// ---------------------------------------------------------------------------
// v3: split-KV (flash-decoding) + warp-shuffle reduction.
//
// Problem with v2: only n_heads (32) blocks launch — an A100 has 108 SMs, so
// most sit idle (low occupancy). v3 splits the kv_seq into chunks and launches
// (n_heads x n_splits) blocks, so many SMs work in parallel. Each block computes
// a PARTIAL streaming-softmax over its chunk; a second kernel merges the partials
// per head using the flash-attention combine rule.
//
// The per-j dot product uses a WARP-SHUFFLE reduction (__shfl_down_sync) instead
// of the shared-memory tree: lanes exchange values in registers, no __syncthreads
// within a warp. head_dim=64 == 2 warps, combined via a 2-element shared array.
//
// Scratch (device): partial_m[h,s], partial_l[h,s], partial_acc[h,s,d].

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

// Pass 1: each block handles (head h, split s), reduces its KV chunk to a partial.
__global__ void attention_decode_v3_partial_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    float* __restrict__ partial_m,     // [n_heads, n_splits]
    float* __restrict__ partial_l,     // [n_heads, n_splits]
    float* __restrict__ partial_acc,   // [n_heads, n_splits, head_dim]
    int n_heads, int n_kv_heads, int kv_seq, int head_dim,
    float scale, int groups, int n_splits, int chunk)
{
    int h = blockIdx.x;
    int s = blockIdx.y;
    int d = threadIdx.x;
    if (h >= n_heads || s >= n_splits || d >= head_dim) return;

    int kv_head = h / groups;
    const __half* q  = Q + (size_t)h * head_dim;
    const __half* Kh = K + (size_t)kv_head * kv_seq * head_dim;
    const __half* Vh = V + (size_t)kv_head * kv_seq * head_dim;

    int j_start = s * chunk;
    int j_end   = min(j_start + chunk, kv_seq);

    float qd = __half2float(q[d]);
    int   warp = d / WARP_SIZE;
    int   lane = d % WARP_SIZE;
    __shared__ float warp_sums[MAX_HEAD_DIM / WARP_SIZE];
    int   n_warps = head_dim / WARP_SIZE;

    float m = -INFINITY, l = 0.0f, acc = 0.0f;

    for (int j = j_start; j < j_end; j++) {
        const __half* kj = Kh + (size_t)j * head_dim;
        const __half* vj = Vh + (size_t)j * head_dim;

        // Warp-shuffle dot product: reduce within each warp, then across warps.
        float partial = qd * __half2float(kj[d]);
        partial = warp_reduce_sum(partial);
        if (lane == 0) warp_sums[warp] = partial;
        __syncthreads();
        float s_score = 0.0f;
        for (int w = 0; w < n_warps; w++) s_score += warp_sums[w];
        s_score *= scale;
        __syncthreads();

        float m_new = fmaxf(m, s_score);
        float corr  = expf(m - m_new);
        float p     = expf(s_score - m_new);
        l   = l * corr + p;
        acc = acc * corr + p * __half2float(vj[d]);
        m   = m_new;
    }

    // Write this block's partial (one acc element per thread).
    size_t base = ((size_t)h * n_splits + s);
    partial_acc[base * head_dim + d] = acc;
    if (d == 0) {                       // m and l are identical across threads
        partial_m[base] = (j_end > j_start) ? m : -INFINITY;
        partial_l[base] = l;
    }
}

// Pass 2: merge the n_splits partials per head via the flash-attention combine.
__global__ void attention_decode_v3_combine_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_acc,
    __half* __restrict__ out,
    int n_heads, int head_dim, int n_splits)
{
    int h = blockIdx.x;
    int d = threadIdx.x;
    if (h >= n_heads || d >= head_dim) return;

    float m = -INFINITY, l = 0.0f, acc = 0.0f;
    for (int s = 0; s < n_splits; s++) {
        size_t base = (size_t)h * n_splits + s;
        float ms = partial_m[base];
        float ls = partial_l[base];
        float as = partial_acc[base * head_dim + d];

        float m_new = fmaxf(m, ms);
        float corr  = expf(m  - m_new);   // rescale running state
        float cs    = expf(ms - m_new);   // rescale this split
        l   = l * corr + ls * cs;
        acc = acc * corr + as * cs;
        m   = m_new;
    }
    out[(size_t)h * head_dim + d] = __float2half(acc / l);
}

void launch_attention_decode_v3(
    const __half* Q, const __half* K, const __half* V, __half* out,
    int n_heads, int n_kv_heads, int kv_seq, int head_dim, float scale)
{
    int groups = n_heads / n_kv_heads;

    // Choose splits: ~256 tokens per chunk, capped so we don't over-split.
    const int CHUNK = 256, MAX_SPLITS = 16;
    int n_splits = (kv_seq + CHUNK - 1) / CHUNK;
    if (n_splits < 1)          n_splits = 1;
    if (n_splits > MAX_SPLITS) n_splits = MAX_SPLITS;
    int chunk = (kv_seq + n_splits - 1) / n_splits;

    float *pm, *pl, *pa;
    cudaMalloc(&pm, (size_t)n_heads * n_splits * sizeof(float));
    cudaMalloc(&pl, (size_t)n_heads * n_splits * sizeof(float));
    cudaMalloc(&pa, (size_t)n_heads * n_splits * head_dim * sizeof(float));

    dim3 grid1(n_heads, n_splits);
    attention_decode_v3_partial_kernel<<<grid1, head_dim>>>(
        Q, K, V, pm, pl, pa,
        n_heads, n_kv_heads, kv_seq, head_dim, scale, groups, n_splits, chunk);

    attention_decode_v3_combine_kernel<<<n_heads, head_dim>>>(
        pm, pl, pa, out, n_heads, head_dim, n_splits);

    cudaFree(pm); cudaFree(pl); cudaFree(pa);
}
