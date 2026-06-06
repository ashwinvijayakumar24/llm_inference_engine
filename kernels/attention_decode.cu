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
