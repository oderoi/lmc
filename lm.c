/*
 * gpt2_edge.c — Dependency-free GPT-2 124M inference engine in pure C99
 * Target: Ultra-constrained edge devices (Raspberry Pi, phones, MCUs)
 * Author: NileAGI Elite ML Engineering Team
 *
 * Architecture: GPT-2 124M
 *   - 12 transformer layers
 *   - 12 attention heads
 *   - 768 embedding dimension
 *   - 3072 FFN hidden dimension
 *   - 50257 vocabulary size
 *   - 1024 max sequence length
 *
 * Compile: gcc -O3 -march=native -ffast-math -fopenmp gpt2_edge.c -o gpt2_edge -lm
 * Usage:   ./gpt2_edge "Your prompt here" [max_tokens] [temperature] [top_p]
 */

/* ============================================================
 * STANDARD HEADERS — THE ONLY DEPENDENCIES WE ALLOW
 * ============================================================ */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ============================================================
 * GPT-2 124M HYPERPARAMETERS
 * ============================================================ */
#define GPT2_VOCAB_SIZE   50257
#define GPT2_SEQ_LEN      1024
#define GPT2_N_LAYERS     12
#define GPT2_N_HEADS      12
#define GPT2_EMBED_DIM    768
#define GPT2_FFN_DIM      3072   /* 4 * EMBED_DIM */
#define GPT2_HEAD_DIM     64     /* EMBED_DIM / N_HEADS */

/* ============================================================
 * BINARY MODEL FILE MAGIC & VERSION
 * ============================================================ */
#define MODEL_MAGIC   0x47505432U  /* "GPT2" in hex */
#define MODEL_VERSION 1

/* ============================================================
 * TOKENIZER CONSTANTS
 * ============================================================ */
#define BPE_MAX_VOCAB      50257
#define BPE_MAX_MERGES     50000
#define BPE_TOKEN_MAX_LEN  256
#define UNICODE_BYTE_RANGE 256

/* ============================================================
 * MEMORY ARENA — avoids malloc fragmentation on edge devices
 * We allocate one giant contiguous buffer for ALL weights.
 * ============================================================ */
typedef struct {
    float *data;
    size_t capacity;   /* total floats allocated */
    size_t used;       /* floats consumed so far  */
} Arena;

static Arena g_arena;

/* Allocate `n` floats from the arena (returns zeroed memory) */
static float* arena_alloc(size_t n) {
    if (g_arena.used + n > g_arena.capacity) {
        fprintf(stderr, "[FATAL] Arena OOM: need %zu, have %zu\n",
                n, g_arena.capacity - g_arena.used);
        exit(1);
    }
    float *ptr = g_arena.data + g_arena.used;
    g_arena.used += n;
    return ptr;
}

/* ============================================================
 * WEIGHT STRUCTURES
 * Each layer's weights are stored contiguously for cache
 * friendliness. We use raw float pointers into the arena.
 * ============================================================ */

/* Per-transformer-block weights */
typedef struct {
    /* --- Layer Norm 1 (before attention) --- */
    float *ln1_weight;   /* [EMBED_DIM] */
    float *ln1_bias;     /* [EMBED_DIM] */

    /* --- Fused QKV projection --- */
    /* Shape: [3 * EMBED_DIM, EMBED_DIM] stored as [out, in]  */
    float *qkv_weight;   /* [3*EMBED_DIM * EMBED_DIM] */
    float *qkv_bias;     /* [3*EMBED_DIM]              */

    /* --- Attention output projection --- */
    float *attn_proj_weight;  /* [EMBED_DIM * EMBED_DIM] */
    float *attn_proj_bias;    /* [EMBED_DIM]             */

    /* --- Layer Norm 2 (before FFN) --- */
    float *ln2_weight;   /* [EMBED_DIM] */
    float *ln2_bias;     /* [EMBED_DIM] */

    /* --- FFN (MLP) --- */
    float *ffn_fc_weight;    /* [FFN_DIM * EMBED_DIM] */
    float *ffn_fc_bias;      /* [FFN_DIM]             */
    float *ffn_proj_weight;  /* [EMBED_DIM * FFN_DIM] */
    float *ffn_proj_bias;    /* [EMBED_DIM]           */
} LayerWeights;

/* Full model weights */
typedef struct {
    /* Token embedding table: [VOCAB_SIZE, EMBED_DIM] */
    float *wte;
    /* Positional embedding table: [SEQ_LEN, EMBED_DIM] */
    float *wpe;

    /* 12 transformer layers */
    LayerWeights layers[GPT2_N_LAYERS];

    /* Final layer norm */
    float *ln_f_weight;  /* [EMBED_DIM] */
    float *ln_f_bias;    /* [EMBED_DIM] */

    /* Language model head weight == wte (tied weights in GPT-2) */
    /* We reuse the wte pointer — no extra memory needed */
} ModelWeights;

/* ============================================================
 * KV CACHE — critical for autoregressive inference speed
 * Instead of recomputing all past keys/values, we cache them.
 * Layout: [N_LAYERS, SEQ_LEN, N_HEADS, HEAD_DIM]
 * ============================================================ */
typedef struct {
    float *k_cache;  /* [N_LAYERS * SEQ_LEN * N_HEADS * HEAD_DIM] */
    float *v_cache;  /* [N_LAYERS * SEQ_LEN * N_HEADS * HEAD_DIM] */
    int    seq_len;  /* how many tokens are currently cached */
} KVCache;

/* ============================================================
 * ACTIVATION BUFFERS — reused across forward passes
 * Preallocate to avoid runtime malloc calls
 * ============================================================ */
typedef struct {
    float *x;           /* [SEQ_LEN, EMBED_DIM] — main residual stream */
    float *x_norm;      /* [SEQ_LEN, EMBED_DIM] — after layer norm     */
    float *qkv;         /* [SEQ_LEN, 3*EMBED_DIM] — fused QKV          */
    float *attn_out;    /* [SEQ_LEN, EMBED_DIM]                         */
    float *ffn_hidden;  /* [SEQ_LEN, FFN_DIM]                           */
    float *logits;      /* [VOCAB_SIZE] — final output logits           */
    float *attn_scores; /* [N_HEADS, SEQ_LEN, SEQ_LEN] — attention map */
} Activations;

/* ============================================================
 * TOKENIZER STRUCTURES
 * We implement a full byte-level BPE tokenizer compatible
 * with the official GPT-2 tokenizer (tiktoken gpt2 encoding).
 * ============================================================ */

/* BPE merge rule: (left_token, right_token) -> merged_token */
typedef struct {
    int left;
    int right;
    int result;
} BPEMerge;

/* Vocabulary entry: token_id -> byte string */
typedef struct {
    uint8_t bytes[BPE_TOKEN_MAX_LEN];
    int     len;
} VocabEntry;

typedef struct {
    /* vocab: token_id -> UTF-8 bytes */
    VocabEntry vocab[BPE_MAX_VOCAB];
    int        vocab_size;

    /* merges: ordered list of BPE merge rules */
    BPEMerge merges[BPE_MAX_MERGES];
    int      n_merges;

    /*
     * GPT-2 byte-level BPE uses a specific 256-char -> token mapping
     * for raw bytes. bytes_to_unicode[b] gives the Unicode codepoint
     * string used to represent raw byte b in the vocabulary.
     */
    char byte_encoder[256][8];  /* byte -> utf8 string */
    int  byte_decoder[0x400];   /* unicode codepoint -> byte */

    /* Reverse vocab lookup: token string -> token_id */
    /* We use a simple hash table for O(1) lookup     */
    /* Hash table: (string_hash % HASH_SIZE) -> list  */
#define VOCAB_HASH_SIZE 131072
    int vocab_hash[VOCAB_HASH_SIZE]; /* -1 = empty */
    int vocab_hash_next[BPE_MAX_VOCAB];
} Tokenizer;

/* ============================================================
 * GLOBAL STATE
 * ============================================================ */
static ModelWeights  g_weights;
static KVCache       g_kv_cache;
static Activations   g_act;
static Tokenizer     g_tokenizer;

/* ============================================================
 * MATH UTILITIES
 * ============================================================ */

/* Fast approximate GeLU used in GPT-2 FFN
 * Original: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 * This is the exact formula GPT-2 uses (not the erf variant).
 */
static inline float gelu(float x) {
    /* sqrt(2/pi) ≈ 0.7978845608 */
    const float c = 0.7978845608f;
    const float k = 0.044715f;
    float x3 = x * x * x;
    float inner = c * (x + k * x3);
    /* tanh approximation: slightly faster than libm tanh on some targets */
    return 0.5f * x * (1.0f + tanhf(inner));
}

/* Softmax in-place over a vector of length n */
static void softmax(float *x, int n) {
    /* Find max for numerical stability */
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    /* exp and sum */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    /* normalize */
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        x[i] *= inv_sum;
    }
}

/* Layer Normalization: y = (x - mean) / sqrt(var + eps) * weight + bias
 * Applied to last dimension of size `dim`.
 * `x` and `out` may alias (in-place safe).
 */
static void layer_norm(float *out, const float *x, const float *weight,
                       const float *bias, int dim) {
    const float eps = 1e-5f;
    /* Compute mean */
    float mean = 0.0f;
    for (int i = 0; i < dim; i++) mean += x[i];
    mean /= (float)dim;
    /* Compute variance */
    float var = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= (float)dim;
    float inv_std = 1.0f / sqrtf(var + eps);
    /* Normalize and scale */
    for (int i = 0; i < dim; i++) {
        out[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

/*
 * General Matrix-Vector multiply: out[M] = weight[M,K] * in[K] + bias[M]
 * weight is stored row-major: weight[row * K + col]
 *
 * This is the hot path. We use blocking and loop unrolling for
 * cache efficiency. The compiler with -O3 -march=native will
 * auto-vectorize the inner loop to use SIMD (SSE/AVX/NEON).
 */
static void matmul_vec(float *out, const float *weight, const float *bias,
                       const float *in, int M, int K) {
    /* Block size tuned for L1 cache (~32KB typical) */
#define BLOCK_K 64

    /* Initialize output with bias */
    if (bias) {
        memcpy(out, bias, M * sizeof(float));
    } else {
        memset(out, 0, M * sizeof(float));
    }

    /* Blocked matrix-vector product */
    for (int kb = 0; kb < K; kb += BLOCK_K) {
        int k_end = kb + BLOCK_K < K ? kb + BLOCK_K : K;

#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(M >= 512)
#endif
        for (int m = 0; m < M; m++) {
            const float *w_row = weight + m * K + kb;
            const float *x_blk = in + kb;
            float acc = 0.0f;
            int k_len = k_end - kb;

            /* Unrolled inner loop — 8x unroll for ILP */
            int k = 0;
            for (; k <= k_len - 8; k += 8) {
                acc += w_row[k+0] * x_blk[k+0]
                     + w_row[k+1] * x_blk[k+1]
                     + w_row[k+2] * x_blk[k+2]
                     + w_row[k+3] * x_blk[k+3]
                     + w_row[k+4] * x_blk[k+4]
                     + w_row[k+5] * x_blk[k+5]
                     + w_row[k+6] * x_blk[k+6]
                     + w_row[k+7] * x_blk[k+7];
            }
            /* Handle remainder */
            for (; k < k_len; k++) {
                acc += w_row[k] * x_blk[k];
            }
            out[m] += acc;
        }
    }
#undef BLOCK_K
}

/*
 * Batch matrix-vector multiply for multiple sequence positions.
 * out[seq_len, M] = in[seq_len, K] * weight^T[M, K] + bias[M]
 * Used for QKV projection, FFN, etc.
 */
static void matmul_seq(float *out, const float *weight, const float *bias,
                       const float *in, int seq_len, int M, int K) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int t = 0; t < seq_len; t++) {
        matmul_vec(out + t * M, weight, bias, in + t * K, M, K);
    }
}

/* ============================================================
 * ATTENTION MECHANISM
 * Causal multi-head self-attention with KV cache.
 *
 * For each new token at position `pos`, we:
 * 1. Compute Q, K, V for the new token only
 * 2. Store K, V in the cache
 * 3. Compute attention over all cached positions [0..pos]
 * 4. Output weighted sum
 * ============================================================ */
static void attention_forward(
    float *out,            /* [EMBED_DIM] output for current token */
    const float *x_norm,   /* [EMBED_DIM] normalized input         */
    const LayerWeights *lw,/* layer weights                        */
    float *k_cache,        /* [SEQ_LEN, N_HEADS, HEAD_DIM]         */
    float *v_cache,        /* [SEQ_LEN, N_HEADS, HEAD_DIM]         */
    int pos,               /* current sequence position             */
    float *qkv_buf,        /* scratch buffer [3*EMBED_DIM]          */
    float *scores_buf      /* scratch buffer [N_HEADS * SEQ_LEN]    */
) {
    const int D = GPT2_EMBED_DIM;
    const int H = GPT2_N_HEADS;
    const int Dh = GPT2_HEAD_DIM;
    const float scale = 1.0f / sqrtf((float)Dh);

    /* Step 1: Compute fused QKV for current token */
    matmul_vec(qkv_buf, lw->qkv_weight, lw->qkv_bias, x_norm, 3 * D, D);

    float *q_vec = qkv_buf;           /* [D] */
    float *k_vec = qkv_buf + D;       /* [D] */
    float *v_vec = qkv_buf + 2 * D;   /* [D] */

    /* Step 2: Store K, V in cache at position `pos` */
    /* Cache layout: [SEQ_LEN, N_HEADS * HEAD_DIM]   */
    float *k_dest = k_cache + pos * H * Dh;
    float *v_dest = v_cache + pos * H * Dh;
    memcpy(k_dest, k_vec, H * Dh * sizeof(float));
    memcpy(v_dest, v_vec, H * Dh * sizeof(float));

    /* Step 3: For each head, compute attention over [0..pos] */
    int ctx_len = pos + 1;  /* number of tokens to attend to */

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int h = 0; h < H; h++) {
        float *q_h = q_vec + h * Dh;
        float *scores = scores_buf + h * GPT2_SEQ_LEN;

        /* Compute Q·K^T / sqrt(Dh) for all past positions */
        for (int t = 0; t < ctx_len; t++) {
            float *k_t = k_cache + t * H * Dh + h * Dh;
            float dot = 0.0f;
            /* Unrolled dot product */
            int d = 0;
            for (; d <= Dh - 8; d += 8) {
                dot += q_h[d+0] * k_t[d+0] + q_h[d+1] * k_t[d+1]
                     + q_h[d+2] * k_t[d+2] + q_h[d+3] * k_t[d+3]
                     + q_h[d+4] * k_t[d+4] + q_h[d+5] * k_t[d+5]
                     + q_h[d+6] * k_t[d+6] + q_h[d+7] * k_t[d+7];
            }
            for (; d < Dh; d++) dot += q_h[d] * k_t[d];
            scores[t] = dot * scale;
        }
        /* Causal masking: future positions = -inf (implicit via ctx_len) */

        /* Softmax over [0..pos] */
        softmax(scores, ctx_len);

        /* Weighted sum of V vectors */
        float *out_h = out + h * Dh;
        memset(out_h, 0, Dh * sizeof(float));
        for (int t = 0; t < ctx_len; t++) {
            float *v_t = v_cache + t * H * Dh + h * Dh;
            float s = scores[t];
            for (int d = 0; d < Dh; d++) {
                out_h[d] += s * v_t[d];
            }
        }
    }
}

/* ============================================================
 * TRANSFORMER BLOCK FORWARD PASS
 * x: [EMBED_DIM] — the residual stream for the current token
 * ============================================================ */
static void transformer_block_forward(
    float *x,              /* [EMBED_DIM] in-place residual stream */
    const LayerWeights *lw,
    float *k_cache,        /* [SEQ_LEN, N_HEADS, HEAD_DIM]         */
    float *v_cache,        /* [SEQ_LEN, N_HEADS, HEAD_DIM]         */
    int pos,
    float *scratch_norm,   /* [EMBED_DIM] temp buffer              */
    float *scratch_qkv,    /* [3*EMBED_DIM] temp buffer            */
    float *scratch_attn,   /* [EMBED_DIM] temp buffer              */
    float *scratch_scores, /* [N_HEADS * SEQ_LEN] temp buffer      */
    float *scratch_ffn     /* [FFN_DIM] temp buffer                */
) {
    const int D = GPT2_EMBED_DIM;
    const int F = GPT2_FFN_DIM;

    /* --- Sub-layer 1: LayerNorm + Multi-Head Attention + Residual --- */

    /* LayerNorm 1 */
    layer_norm(scratch_norm, x, lw->ln1_weight, lw->ln1_bias, D);

    /* Multi-head causal self-attention */
    attention_forward(scratch_attn, scratch_norm, lw,
                      k_cache, v_cache, pos,
                      scratch_qkv, scratch_scores);

    /* Attention output projection */
    float attn_proj_out[GPT2_EMBED_DIM];
    matmul_vec(attn_proj_out, lw->attn_proj_weight, lw->attn_proj_bias,
               scratch_attn, D, D);

    /* Residual connection 1 */
    for (int i = 0; i < D; i++) x[i] += attn_proj_out[i];

    /* --- Sub-layer 2: LayerNorm + FFN + Residual --- */

    /* LayerNorm 2 */
    layer_norm(scratch_norm, x, lw->ln2_weight, lw->ln2_bias, D);

    /* FFN: fc -> GeLU -> proj */
    matmul_vec(scratch_ffn, lw->ffn_fc_weight, lw->ffn_fc_bias,
               scratch_norm, F, D);

    /* GeLU activation */
    for (int i = 0; i < F; i++) scratch_ffn[i] = gelu(scratch_ffn[i]);

    /* FFN projection back to D */
    float ffn_out[GPT2_EMBED_DIM];
    matmul_vec(ffn_out, lw->ffn_proj_weight, lw->ffn_proj_bias,
               scratch_ffn, D, F);

    /* Residual connection 2 */
    for (int i = 0; i < D; i++) x[i] += ffn_out[i];
}

/* ============================================================
 * FULL MODEL FORWARD PASS
 * Processes ONE new token at position `pos`.
 * Returns pointer to logits[VOCAB_SIZE].
 * ============================================================ */
static float* model_forward(int token_id, int pos) {
    const int D = GPT2_EMBED_DIM;
    const int V = GPT2_VOCAB_SIZE;

    /* Scratch buffers — allocated once in g_act */
    float *scratch_norm   = g_act.x_norm;
    float *scratch_qkv    = g_act.qkv;
    float *scratch_attn   = g_act.attn_out;
    float *scratch_scores = g_act.attn_scores;
    float *scratch_ffn    = g_act.ffn_hidden;

    /* Step 1: Token embedding + positional embedding */
    float *tok_emb = g_weights.wte + token_id * D;
    float *pos_emb = g_weights.wpe + pos * D;

    /* x = tok_emb + pos_emb */
    float *x = g_act.x;
    for (int i = 0; i < D; i++) {
        x[i] = tok_emb[i] + pos_emb[i];
    }

    /* Step 2: Pass through all 12 transformer layers */
    for (int l = 0; l < GPT2_N_LAYERS; l++) {
        /* KV cache pointers for this layer */
        float *k_cache_l = g_kv_cache.k_cache
                         + l * GPT2_SEQ_LEN * GPT2_N_HEADS * GPT2_HEAD_DIM;
        float *v_cache_l = g_kv_cache.v_cache
                         + l * GPT2_SEQ_LEN * GPT2_N_HEADS * GPT2_HEAD_DIM;

        transformer_block_forward(x, &g_weights.layers[l],
                                  k_cache_l, v_cache_l, pos,
                                  scratch_norm, scratch_qkv, scratch_attn,
                                  scratch_scores, scratch_ffn);
    }

    /* Step 3: Final layer norm */
    float x_final[GPT2_EMBED_DIM];
    layer_norm(x_final, x, g_weights.ln_f_weight, g_weights.ln_f_bias, D);

    /* Step 4: LM head = x_final @ wte^T  (weight tying) */
    /* This is the most expensive step: [V, D] * [D] -> [V] */
    matmul_vec(g_act.logits, g_weights.wte, NULL, x_final, V, D);

    return g_act.logits;
}

/* ============================================================
 * SAMPLING — Top-p (Nucleus) + Temperature
 *
 * 1. Apply temperature to logits (sharpen/flatten distribution)
 * 2. Softmax
 * 3. Sort descending
 * 4. Find nucleus: smallest set of tokens with cumulative prob >= top_p
 * 5. Sample from nucleus
 * ============================================================ */

/* Simple LCG random number generator (good enough, dependency-free) */
static uint64_t g_rng_state = 0;

static void rng_seed(uint64_t seed) {
    g_rng_state = seed ^ 0xdeadbeefcafeULL;
}

/* Xorshift64 — fast, decent quality */
static uint64_t rng_u64(void) {
    g_rng_state ^= g_rng_state << 13;
    g_rng_state ^= g_rng_state >> 7;
    g_rng_state ^= g_rng_state << 17;
    return g_rng_state;
}

/* Uniform float in [0, 1) */
static float rng_float(void) {
    return (float)(rng_u64() >> 11) / (float)(1ULL << 53);
}

/* Comparison function for sorting (descending by prob) */
typedef struct { float prob; int idx; } ProbIdx;

static int cmp_prob_desc(const void *a, const void *b) {
    const ProbIdx *pa = (const ProbIdx *)a;
    const ProbIdx *pb = (const ProbIdx *)b;
    if (pb->prob > pa->prob) return 1;
    if (pb->prob < pa->prob) return -1;
    return 0;
}

/* Top-p sampling with temperature */
static int sample_top_p(float *logits, float temperature, float top_p) {
    const int V = GPT2_VOCAB_SIZE;

    /* Apply temperature scaling */
    if (temperature < 1e-6f) {
        /* Greedy decoding */
        int best = 0;
        for (int i = 1; i < V; i++) {
            if (logits[i] > logits[best]) best = i;
        }
        return best;
    }

    float inv_temp = 1.0f / temperature;
    for (int i = 0; i < V; i++) {
        logits[i] *= inv_temp;
    }

    /* Softmax */
    softmax(logits, V);

    /* Build sorted index array */
    /* Note: We allocate this on the stack — 50257 * 8 bytes ≈ 400KB
     * For microcontrollers this might be too much; use heap in that case */
    static ProbIdx sorted[GPT2_VOCAB_SIZE];
    for (int i = 0; i < V; i++) {
        sorted[i].prob = logits[i];
        sorted[i].idx  = i;
    }
    qsort(sorted, V, sizeof(ProbIdx), cmp_prob_desc);

    /* Find nucleus cutoff */
    float cumsum = 0.0f;
    int nucleus_size = 0;
    for (int i = 0; i < V; i++) {
        cumsum += sorted[i].prob;
        nucleus_size = i + 1;
        if (cumsum >= top_p) break;
    }

    /* Sample from nucleus */
    /* Re-normalize probabilities within nucleus */
    float nucleus_sum = 0.0f;
    for (int i = 0; i < nucleus_size; i++) nucleus_sum += sorted[i].prob;
    float inv_ns = 1.0f / nucleus_sum;

    float r = rng_float();
    float cdf = 0.0f;
    for (int i = 0; i < nucleus_size; i++) {
        cdf += sorted[i].prob * inv_ns;
        if (r < cdf) return sorted[i].idx;
    }
    /* Fallback */
    return sorted[nucleus_size - 1].idx;
}

/* ============================================================
 * MODEL LOADING
 * Binary format (little-endian):
 *   Header: [magic:u32][version:u32][vocab_size:u32][seq_len:u32]
 *           [n_layers:u32][n_heads:u32][embed_dim:u32]
 *   Weights: all float32 arrays concatenated in order:
 *     wte, wpe,
 *     for each layer: ln1_w, ln1_b, qkv_w, qkv_b,
 *                     ap_w, ap_b, ln2_w, ln2_b,
 *                     ffn_fc_w, ffn_fc_b, ffn_proj_w, ffn_proj_b
 *     ln_f_w, ln_f_b
 * ============================================================ */

static void load_model(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[ERROR] Cannot open model file: %s\n", path);
        fprintf(stderr, "Run the converter script first: python3 converter.py\n");
        exit(1);
    }

    /* Read and validate header */
    uint32_t magic, version, vocab_size, seq_len, n_layers, n_heads, embed_dim;
    fread(&magic,      sizeof(uint32_t), 1, f);
    fread(&version,    sizeof(uint32_t), 1, f);
    fread(&vocab_size, sizeof(uint32_t), 1, f);
    fread(&seq_len,    sizeof(uint32_t), 1, f);
    fread(&n_layers,   sizeof(uint32_t), 1, f);
    fread(&n_heads,    sizeof(uint32_t), 1, f);
    fread(&embed_dim,  sizeof(uint32_t), 1, f);

    if (magic != MODEL_MAGIC) {
        fprintf(stderr, "[ERROR] Invalid model file (bad magic: 0x%08X)\n", magic);
        exit(1);
    }
    if (version != MODEL_VERSION) {
        fprintf(stderr, "[ERROR] Model version mismatch: got %u, expected %u\n",
                version, MODEL_VERSION);
        exit(1);
    }
    if (vocab_size != GPT2_VOCAB_SIZE || seq_len != GPT2_SEQ_LEN ||
        n_layers != GPT2_N_LAYERS || n_heads != GPT2_N_HEADS ||
        embed_dim != GPT2_EMBED_DIM) {
        fprintf(stderr, "[ERROR] Model architecture mismatch!\n");
        fprintf(stderr, "  Expected: V=%d, S=%d, L=%d, H=%d, D=%d\n",
                GPT2_VOCAB_SIZE, GPT2_SEQ_LEN, GPT2_N_LAYERS,
                GPT2_N_HEADS, GPT2_EMBED_DIM);
        fprintf(stderr, "  Got:      V=%u, S=%u, L=%u, H=%u, D=%u\n",
                vocab_size, seq_len, n_layers, n_heads, embed_dim);
        exit(1);
    }

    /*
     * Calculate total parameter count and allocate arena.
     * GPT-2 124M parameter breakdown:
     *   wte:          50257 * 768     = 38,597,376
     *   wpe:           1024 * 768     =    786,432
     *   per layer (×12):
     *     ln1:         2 * 768       =      1,536
     *     qkv:   (3*768)*768 + 3*768 =  1,771,776
     *     attn_proj: 768*768 + 768   =    590,592
     *     ln2:         2 * 768       =      1,536
     *     ffn_fc:  3072*768 + 3072   =  2,362,368
     *     ffn_proj: 768*3072 + 768   =  2,360,064
     *   ln_f:          2 * 768       =      1,536
     *   Total ≈ 124M params
     */
    const int D = GPT2_EMBED_DIM;
    const int V = GPT2_VOCAB_SIZE;
    const int S = GPT2_SEQ_LEN;
    const int L = GPT2_N_LAYERS;
    const int F = GPT2_FFN_DIM;

    size_t total_params = 0;
    total_params += (size_t)V * D;  /* wte */
    total_params += (size_t)S * D;  /* wpe */
    for (int l = 0; l < L; l++) {
        total_params += 2 * D;              /* ln1 */
        total_params += (size_t)3*D*D + 3*D; /* qkv */
        total_params += (size_t)D*D + D;    /* attn_proj */
        total_params += 2 * D;              /* ln2 */
        total_params += (size_t)F*D + F;    /* ffn_fc */
        total_params += (size_t)D*F + D;    /* ffn_proj */
    }
    total_params += 2 * D;  /* ln_f */

    printf("[INFO] Model parameters: %zu (%.1f MB)\n",
           total_params, total_params * 4.0 / (1024*1024));

    /* Allocate arena */
    g_arena.capacity = total_params;
    g_arena.data = (float*)malloc(total_params * sizeof(float));
    g_arena.used = 0;
    if (!g_arena.data) {
        fprintf(stderr, "[FATAL] Cannot allocate %.1f MB for weights\n",
                total_params * 4.0 / (1024*1024));
        exit(1);
    }

    /* Helper macro: allocate n floats and read from file */
#define LOAD(ptr, n) do { \
    (ptr) = arena_alloc(n); \
    if (fread((ptr), sizeof(float), (n), f) != (size_t)(n)) { \
        fprintf(stderr, "[ERROR] Truncated model file at %s:%d\n", \
                __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

    /* Load weights in the same order the converter saves them */
    LOAD(g_weights.wte, (size_t)V * D);
    LOAD(g_weights.wpe, (size_t)S * D);

    for (int l = 0; l < L; l++) {
        LayerWeights *lw = &g_weights.layers[l];
        LOAD(lw->ln1_weight,      D);
        LOAD(lw->ln1_bias,        D);
        LOAD(lw->qkv_weight,      (size_t)3*D*D);
        LOAD(lw->qkv_bias,        3*D);
        LOAD(lw->attn_proj_weight, (size_t)D*D);
        LOAD(lw->attn_proj_bias,  D);
        LOAD(lw->ln2_weight,      D);
        LOAD(lw->ln2_bias,        D);
        LOAD(lw->ffn_fc_weight,   (size_t)F*D);
        LOAD(lw->ffn_fc_bias,     F);
        LOAD(lw->ffn_proj_weight, (size_t)D*F);
        LOAD(lw->ffn_proj_bias,   D);
    }

    LOAD(g_weights.ln_f_weight, D);
    LOAD(g_weights.ln_f_bias,   D);

#undef LOAD

    fclose(f);
    printf("[INFO] Model loaded successfully.\n");
}

/* ============================================================
 * ACTIVATION BUFFER INITIALIZATION
 * ============================================================ */
static void init_activations(void) {
    const int D = GPT2_EMBED_DIM;
    const int V = GPT2_VOCAB_SIZE;
    const int F = GPT2_FFN_DIM;
    const int H = GPT2_N_HEADS;
    const int S = GPT2_SEQ_LEN;

    /* We allocate activations from heap (not arena, to keep arena for weights) */
    g_act.x           = (float*)malloc(D * sizeof(float));
    g_act.x_norm       = (float*)malloc(D * sizeof(float));
    g_act.qkv          = (float*)malloc(3 * D * sizeof(float));
    g_act.attn_out     = (float*)malloc(D * sizeof(float));
    g_act.ffn_hidden   = (float*)malloc(F * sizeof(float));
    g_act.logits       = (float*)malloc(V * sizeof(float));
    g_act.attn_scores  = (float*)malloc(H * S * sizeof(float));

    if (!g_act.x || !g_act.x_norm || !g_act.qkv || !g_act.attn_out ||
        !g_act.ffn_hidden || !g_act.logits || !g_act.attn_scores) {
        fprintf(stderr, "[FATAL] Cannot allocate activation buffers\n");
        exit(1);
    }
}

/* ============================================================
 * KV CACHE INITIALIZATION
 * ============================================================ */
static void init_kv_cache(void) {
    const size_t cache_size = (size_t)GPT2_N_LAYERS
                            * GPT2_SEQ_LEN
                            * GPT2_N_HEADS
                            * GPT2_HEAD_DIM;

    g_kv_cache.k_cache = (float*)calloc(cache_size, sizeof(float));
    g_kv_cache.v_cache = (float*)calloc(cache_size, sizeof(float));
    g_kv_cache.seq_len = 0;

    if (!g_kv_cache.k_cache || !g_kv_cache.v_cache) {
        fprintf(stderr, "[FATAL] Cannot allocate KV cache (%.1f MB)\n",
                cache_size * 2 * 4.0 / (1024*1024));
        exit(1);
    }
    printf("[INFO] KV cache allocated: %.1f MB\n",
           cache_size * 2 * 4.0 / (1024*1024));
}

/* ============================================================
 * BYTE-LEVEL BPE TOKENIZER
 *
 * GPT-2 uses a byte-level BPE where every raw byte appears in
 * the vocabulary. To handle arbitrary Unicode, GPT-2 maps
 * each raw byte to a specific Unicode character before doing
 * BPE, using the bytes_to_unicode() mapping from the original
 * code. This avoids <unk> tokens.
 *
 * The tokenizer loads:
 *   encoder.json  — maps token_string -> token_id
 *   vocab.bpe     — BPE merge rules (one pair per line)
 * ============================================================ */

/*
 * The GPT-2 bytes_to_unicode mapping:
 * Printable ASCII/Latin-1 chars map to themselves.
 * Non-printable bytes get mapped to chars starting at U+0100.
 * We precompute this table at startup.
 */
static void init_byte_encoder(Tokenizer *tok) {
    /* These are the byte values that map to themselves (printable range) */
    /* From the original GPT-2 Python code:
     * bs = list(range(ord('!'), ord('~')+1))   # 33-126
     *    + list(range(ord('¡'), ord('¬')+1))   # 161-172
     *    + list(range(ord('®'), ord('ÿ')+1))   # 174-255
     * cs = bs[:]
     * n = 0
     * for b in range(256):
     *     if b not in bs:
     *         bs.append(b); cs.append(256 + n); n += 1
     */

    int bs[256], cs[256];
    int n_bs = 0;

    /* Fill printable ranges */
    for (int b = 33; b <= 126; b++) { bs[n_bs] = b; cs[n_bs] = b; n_bs++; }
    for (int b = 161; b <= 172; b++) { bs[n_bs] = b; cs[n_bs] = b; n_bs++; }
    for (int b = 174; b <= 255; b++) { bs[n_bs] = b; cs[n_bs] = b; n_bs++; }

    /* Fill non-printable bytes */
    int extra = 256;
    for (int b = 0; b < 256; b++) {
        int found = 0;
        for (int i = 0; i < n_bs; i++) {
            if (bs[i] == b) { found = 1; break; }
        }
        if (!found) {
            bs[n_bs] = b; cs[n_bs] = extra++; n_bs++;
        }
    }

    /* Build encoder: byte -> UTF-8 string of the mapped codepoint */
    for (int i = 0; i < 256; i++) {
        tok->byte_decoder[cs[i]] = bs[i]; /* unicode codepoint -> raw byte */
    }

    /* Encode the unicode codepoints as UTF-8 strings */
    for (int i = 0; i < 256; i++) {
        int cp = cs[i];   /* unicode codepoint */
        char *out = tok->byte_encoder[bs[i]];
        if (cp < 0x80) {
            /* 1-byte UTF-8: ASCII */
            out[0] = (char)cp; out[1] = '\0';
        } else if (cp < 0x800) {
            /* 2-byte UTF-8 */
            out[0] = (char)(0xC0 | (cp >> 6));
            out[1] = (char)(0x80 | (cp & 0x3F));
            out[2] = '\0';
        } else {
            /* 3-byte UTF-8 (shouldn't happen for our range, but be safe) */
            out[0] = (char)(0xE0 | (cp >> 12));
            out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
            out[2] = (char)(0x80 | (cp & 0x3F));
            out[3] = '\0';
        }
    }
}

/* Decode a UTF-8 sequence to a unicode codepoint.
 * Returns the codepoint and advances *s past the character. */
static int utf8_decode(const char **s) {
    unsigned char c = (unsigned char)**s;
    int cp;
    if (c < 0x80) {
        cp = c; (*s)++;
    } else if ((c & 0xE0) == 0xC0) {
        cp = (c & 0x1F) << 6;
        (*s)++;
        cp |= ((unsigned char)**s & 0x3F); (*s)++;
    } else if ((c & 0xF0) == 0xE0) {
        cp = (c & 0x0F) << 12;
        (*s)++;
        cp |= ((unsigned char)**s & 0x3F) << 6; (*s)++;
        cp |= ((unsigned char)**s & 0x3F); (*s)++;
    } else {
        /* 4-byte or invalid: skip */
        cp = '?'; (*s)++;
    }
    return cp;
}

/* Compute a simple hash of a byte string */
static uint32_t str_hash(const uint8_t *s, int len) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) {
        h ^= s[i];
        h *= 16777619u;
    }
    return h;
}

/* Insert a token into the hash table */
static void vocab_hash_insert(Tokenizer *tok, int token_id) {
    uint32_t h = str_hash(tok->vocab[token_id].bytes,
                          tok->vocab[token_id].len);
    uint32_t slot = h % VOCAB_HASH_SIZE;
    tok->vocab_hash_next[token_id] = tok->vocab_hash[slot];
    tok->vocab_hash[slot] = token_id;
}

/* Look up a token string in the hash table. Returns -1 if not found. */
static int vocab_lookup(const Tokenizer *tok, const uint8_t *s, int len) {
    uint32_t h = str_hash(s, len);
    uint32_t slot = h % VOCAB_HASH_SIZE;
    int id = tok->vocab_hash[slot];
    while (id != -1) {
        if (tok->vocab[id].len == len &&
            memcmp(tok->vocab[id].bytes, s, len) == 0) {
            return id;
        }
        id = tok->vocab_hash_next[id];
    }
    return -1;
}

/*
 * Parse encoder.json — a JSON file mapping token_string -> token_id.
 * We write a minimal JSON parser sufficient for this file format.
 * The file looks like: {"!": 0, "\"": 1, ..., "token": 50256}
 */
static void load_encoder_json(Tokenizer *tok, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[ERROR] Cannot open encoder.json: %s\n", path);
        exit(1);
    }

    /* Initialize hash table */
    memset(tok->vocab_hash, -1, sizeof(tok->vocab_hash));
    memset(tok->vocab_hash_next, -1, sizeof(tok->vocab_hash_next));
    tok->vocab_size = 0;

    /* Read entire file into buffer */
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *buf = (char*)malloc(fsize + 1);
    if (!buf) { fprintf(stderr, "[FATAL] OOM loading encoder.json\n"); exit(1); }
    fread(buf, 1, fsize, f);
    buf[fsize] = '\0';
    fclose(f);

    /*
     * Minimal JSON parser:
     * We scan for: "KEY": VALUE pairs
     * KEY is a JSON string (may contain escape sequences)
     * VALUE is an integer
     */
    char *p = buf;

    /* Skip opening '{' */
    while (*p && *p != '{') p++;
    if (*p) p++;  /* skip '{' */

    while (*p) {
        /* Skip whitespace and commas */
        while (*p && (*p == ' ' || *p == '\n' || *p == '\r' ||
                      *p == '\t' || *p == ',')) p++;

        if (*p == '}') break;  /* end of object */
        if (*p != '"') { p++; continue; }  /* skip unexpected chars */

        /* Parse key string */
        p++;  /* skip opening '"' */
        uint8_t key[BPE_TOKEN_MAX_LEN];
        int key_len = 0;

        while (*p && *p != '"' && key_len < BPE_TOKEN_MAX_LEN - 1) {
            if (*p == '\\') {
                p++;
                switch (*p) {
                    case '"':  key[key_len++] = '"';  p++; break;
                    case '\\': key[key_len++] = '\\'; p++; break;
                    case '/':  key[key_len++] = '/';  p++; break;
                    case 'n':  key[key_len++] = '\n'; p++; break;
                    case 'r':  key[key_len++] = '\r'; p++; break;
                    case 't':  key[key_len++] = '\t'; p++; break;
                    case 'b':  key[key_len++] = '\b'; p++; break;
                    case 'f':  key[key_len++] = '\f'; p++; break;
                    case 'u': {
                        /* Unicode escape: \uXXXX */
                        p++;
                        char hex[5] = {0};
                        for (int hi = 0; hi < 4 && *p; hi++) hex[hi] = *p++;
                        int cp = (int)strtol(hex, NULL, 16);
                        /* Encode as UTF-8 */
                        if (cp < 0x80) {
                            key[key_len++] = (uint8_t)cp;
                        } else if (cp < 0x800) {
                            key[key_len++] = (uint8_t)(0xC0 | (cp >> 6));
                            key[key_len++] = (uint8_t)(0x80 | (cp & 0x3F));
                        } else {
                            key[key_len++] = (uint8_t)(0xE0 | (cp >> 12));
                            key[key_len++] = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
                            key[key_len++] = (uint8_t)(0x80 | (cp & 0x3F));
                        }
                        break;
                    }
                    default: key[key_len++] = *p++; break;
                }
            } else {
                key[key_len++] = (uint8_t)*p++;
            }
        }
        if (*p == '"') p++;  /* skip closing '"' */

        /* Skip ':' and whitespace */
        while (*p && (*p == ' ' || *p == ':' || *p == '\t')) p++;

        /* Parse integer value */
        if (*p < '0' || *p > '9') continue;
        int token_id = 0;
        while (*p >= '0' && *p <= '9') {
            token_id = token_id * 10 + (*p - '0');
            p++;
        }

        /* Store in vocabulary */
        if (token_id < BPE_MAX_VOCAB) {
            memcpy(tok->vocab[token_id].bytes, key, key_len);
            tok->vocab[token_id].len = key_len;
            vocab_hash_insert(tok, token_id);
            if (token_id + 1 > tok->vocab_size) {
                tok->vocab_size = token_id + 1;
            }
        }
    }

    free(buf);
    printf("[INFO] Vocabulary loaded: %d tokens\n", tok->vocab_size);
}

/*
 * Parse vocab.bpe — the BPE merge rules file.
 * Format: first line is "#version: ..."
 * Then each line: "TOKEN_A TOKEN_B" (space-separated)
 * The line number (starting from 0 after header) is the merge priority.
 */
static void load_vocab_bpe(Tokenizer *tok, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[ERROR] Cannot open vocab.bpe: %s\n", path);
        exit(1);
    }

    tok->n_merges = 0;
    char line[1024];

    /* Skip first line (version comment) */
    if (!fgets(line, sizeof(line), f)) { fclose(f); return; }

    while (fgets(line, sizeof(line), f) && tok->n_merges < BPE_MAX_MERGES) {
        /* Remove trailing newline */
        int len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';
        if (len == 0) continue;

        /* Find the space separator */
        char *space = strchr(line, ' ');
        if (!space) continue;

        *space = '\0';
        char *left_str  = line;
        char *right_str = space + 1;

        /* Look up token IDs */
        int left_id = vocab_lookup(tok,
                                   (const uint8_t*)left_str, strlen(left_str));
        int right_id = vocab_lookup(tok,
                                    (const uint8_t*)right_str, strlen(right_str));

        if (left_id == -1 || right_id == -1) {
            /* Token not in vocabulary — try to find or create the merged token */
            /* Build merged string */
            int ll = strlen(left_str), rl = strlen(right_str);
            uint8_t merged[BPE_TOKEN_MAX_LEN];
            memcpy(merged, left_str, ll);
            memcpy(merged + ll, right_str, rl);
            int result_id = vocab_lookup(tok, merged, ll + rl);
            if (result_id == -1) continue;
            if (left_id == -1 || right_id == -1) continue;
        }

        /* Build merged string to find result token */
        int ll = strlen(left_str), rl = strlen(right_str);
        if (ll + rl >= BPE_TOKEN_MAX_LEN) continue;

        uint8_t merged[BPE_TOKEN_MAX_LEN];
        memcpy(merged, left_str, ll);
        memcpy(merged + ll, right_str, rl);
        int result_id = vocab_lookup(tok, merged, ll + rl);
        if (result_id == -1) continue;

        tok->merges[tok->n_merges].left   = left_id;
        tok->merges[tok->n_merges].right  = right_id;
        tok->merges[tok->n_merges].result = result_id;
        tok->n_merges++;
    }

    fclose(f);
    printf("[INFO] BPE merges loaded: %d rules\n", tok->n_merges);
}

/*
 * Load tokenizer from encoder.json + vocab.bpe
 */
static void load_tokenizer(const char *encoder_path, const char *bpe_path) {
    init_byte_encoder(&g_tokenizer);
    load_encoder_json(&g_tokenizer, encoder_path);
    load_vocab_bpe(&g_tokenizer, bpe_path);
}

/* ============================================================
 * BPE ENCODING
 * Converts a UTF-8 string to a sequence of token IDs.
 *
 * Algorithm:
 * 1. Pre-tokenize: split on whitespace/punctuation using GPT-2's
 *    regex pattern (approximated). Each "word" is processed separately.
 * 2. For each word:
 *    a. Map each byte to its "unicode" representation
 *    b. Initialize token sequence as individual chars
 *    c. Repeatedly apply the lowest-priority merge rule that applies
 * ============================================================ */

/* GPT-2 regex pattern for pre-tokenization (approximation):
 * Original: r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
 * We approximate by splitting on whitespace boundaries and
 * keeping punctuation separate. */

/* Token sequence for BPE — represents one "word" being tokenized */
#define MAX_WORD_LEN   128
#define MAX_WORD_TOKENS (MAX_WORD_LEN * 4)  /* after byte encoding */

typedef struct {
    int ids[MAX_WORD_TOKENS];
    int len;
} TokenSeq;

/*
 * Apply BPE merges to a token sequence in-place.
 * We use a simple O(n * merges) algorithm suitable for edge devices
 * (words are short; the expensive case is model inference, not tokenization).
 */
static void bpe_apply_merges(TokenSeq *seq, const Tokenizer *tok) {
    /* Repeatedly find and apply the highest-priority (lowest index) merge */
    while (seq->len >= 2) {
        int best_merge = tok->n_merges;  /* lower = higher priority */
        int best_pos   = -1;

        /* Find the pair with the best (lowest index) merge rule */
        for (int i = 0; i < seq->len - 1; i++) {
            int a = seq->ids[i], b = seq->ids[i + 1];
            /* Search for this pair in merge rules */
            /* O(n_merges) per pair — optimize with a hash table for production */
            for (int m = 0; m < tok->n_merges; m++) {
                if (tok->merges[m].left == a && tok->merges[m].right == b) {
                    if (m < best_merge) {
                        best_merge = m;
                        best_pos   = i;
                    }
                    break;
                }
            }
        }

        if (best_pos == -1) break;  /* No more merges apply */

        /* Apply the merge at best_pos: replace (ids[best_pos], ids[best_pos+1])
         * with merges[best_merge].result */
        seq->ids[best_pos] = tok->merges[best_merge].result;
        /* Shift elements left */
        for (int i = best_pos + 1; i < seq->len - 1; i++) {
            seq->ids[i] = seq->ids[i + 1];
        }
        seq->len--;
    }
}

/*
 * Encode a single "word" (pre-tokenized chunk) to token IDs.
 * word_bytes: raw UTF-8 bytes of the word
 * word_len:   number of bytes
 * out_ids:    output token IDs
 * Returns number of tokens produced.
 */
static int encode_word(const Tokenizer *tok,
                       const uint8_t *word_bytes, int word_len,
                       int *out_ids) {
    TokenSeq seq;
    seq.len = 0;

    /* Step 1: Map each byte through byte_encoder to get initial tokens */
    for (int i = 0; i < word_len && seq.len < MAX_WORD_TOKENS; i++) {
        uint8_t b = word_bytes[i];
        const char *encoded = tok->byte_encoder[b];
        /* Look up this single-char token in vocabulary */
        int tid = vocab_lookup(tok, (const uint8_t*)encoded, strlen(encoded));
        if (tid == -1) {
            /* Fallback: use the raw byte value */
            seq.ids[seq.len++] = (int)b;
        } else {
            seq.ids[seq.len++] = tid;
        }
    }

    /* Step 2: Apply BPE merges */
    bpe_apply_merges(&seq, tok);

    /* Copy to output */
    for (int i = 0; i < seq.len; i++) {
        out_ids[i] = seq.ids[i];
    }
    return seq.len;
}

/*
 * Full text encoding.
 * Splits text into "words" using a simplified pre-tokenizer,
 * encodes each word, concatenates token IDs.
 *
 * GPT-2 pre-tokenizer keeps spaces as part of the following word
 * (e.g., " hello" is one token), and splits on punctuation.
 */
static int tokenize(const Tokenizer *tok, const char *text,
                    int *out_ids, int max_tokens) {
    int n_tokens = 0;
    const uint8_t *p = (const uint8_t*)text;
    int text_len = strlen(text);
    int i = 0;

    while (i < text_len && n_tokens < max_tokens) {
        /* Collect a "word": optional leading space + alphanumeric,
         * OR punctuation/other characters (each separately) */
        uint8_t word[MAX_WORD_LEN];
        int wlen = 0;

        /* Include leading space(s) — GPT-2 tokenizer attaches space to next word */
        if (p[i] == ' ' && i + 1 < text_len) {
            word[wlen++] = p[i++];
        }

        if (i >= text_len) {
            /* Just a trailing space */
            if (wlen > 0) {
                int word_ids[MAX_WORD_TOKENS];
                int n = encode_word(tok, word, wlen, word_ids);
                for (int j = 0; j < n && n_tokens < max_tokens; j++) {
                    out_ids[n_tokens++] = word_ids[j];
                }
            }
            break;
        }

        uint8_t c = p[i];

        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
            (c >= '0' && c <= '9') || c >= 0x80) {
            /* Alphanumeric or multi-byte UTF-8: collect the word */
            while (i < text_len && wlen < MAX_WORD_LEN - 1) {
                uint8_t cc = p[i];
                if ((cc >= 'A' && cc <= 'Z') || (cc >= 'a' && cc <= 'z') ||
                    (cc >= '0' && cc <= '9') || cc >= 0x80) {
                    word[wlen++] = p[i++];
                } else break;
            }
        } else if (c == '\n' || c == '\r' || c == '\t') {
            /* Whitespace: one token */
            word[wlen++] = p[i++];
        } else {
            /* Punctuation and other: one character at a time */
            word[wlen++] = p[i++];
        }

        if (wlen > 0) {
            int word_ids[MAX_WORD_TOKENS];
            int n = encode_word(tok, word, wlen, word_ids);
            for (int j = 0; j < n && n_tokens < max_tokens; j++) {
                out_ids[n_tokens++] = word_ids[j];
            }
        }
    }

    return n_tokens;
}

/*
 * Decode token IDs back to UTF-8 text.
 * For each token, look up its byte string in vocab[], then
 * map each byte back through byte_decoder.
 */
static int detokenize_token(const Tokenizer *tok, int token_id,
                            char *out_buf, int buf_size) {
    if (token_id < 0 || token_id >= tok->vocab_size) return 0;

    const VocabEntry *ve = &tok->vocab[token_id];
    const char *s = (const char*)ve->bytes;
    const char *end = s + ve->len;

    int out_len = 0;
    while (s < end && out_len < buf_size - 1) {
        /* Decode UTF-8 codepoint from the token string */
        int cp = utf8_decode(&s);
        /* Map codepoint back to raw byte */
        if (cp >= 0 && cp < 0x400) {
            int raw_byte = tok->byte_decoder[cp];
            out_buf[out_len++] = (char)raw_byte;
        }
    }
    out_buf[out_len] = '\0';
    return out_len;
}

/* ============================================================
 * MAIN GENERATION LOOP
 * ============================================================ */
static void generate(const char *prompt, int max_new_tokens,
                     float temperature, float top_p) {
    /* Step 1: Tokenize the prompt */
    int prompt_tokens[GPT2_SEQ_LEN];
    int n_prompt = tokenize(&g_tokenizer, prompt,
                            prompt_tokens, GPT2_SEQ_LEN);

    if (n_prompt == 0) {
        fprintf(stderr, "[ERROR] Empty prompt after tokenization.\n");
        return;
    }
    printf("[INFO] Prompt tokens: %d\n", n_prompt);

    /* Print the prompt */
    printf("\n--- Generated Text ---\n");
    printf("%s", prompt);
    fflush(stdout);

    /* Step 2: Process prompt tokens through the model */
    /* We feed them one by one, updating the KV cache */
    g_kv_cache.seq_len = 0;
    int total_tokens = n_prompt + max_new_tokens;
    if (total_tokens > GPT2_SEQ_LEN) total_tokens = GPT2_SEQ_LEN;

    float *logits = NULL;
    for (int i = 0; i < n_prompt; i++) {
        logits = model_forward(prompt_tokens[i], i);
        g_kv_cache.seq_len = i + 1;
    }

    /* Step 3: Autoregressive generation */
    int pos = n_prompt;
    char decode_buf[64];

    for (int step = 0; step < max_new_tokens; step++) {
        if (pos >= GPT2_SEQ_LEN) {
            printf("\n[Context window full]\n");
            break;
        }

        /* Sample next token */
        int next_token = sample_top_p(logits, temperature, top_p);

        /* Stop at end-of-text token (50256) */
        if (next_token == 50256) {
            printf("\n[EOS]\n");
            break;
        }

        /* Decode and print the new token */
        int dec_len = detokenize_token(&g_tokenizer, next_token,
                                       decode_buf, sizeof(decode_buf));
        if (dec_len > 0) {
            fwrite(decode_buf, 1, dec_len, stdout);
            fflush(stdout);  /* streaming output */
        }

        /* Forward pass with new token */
        logits = model_forward(next_token, pos);
        g_kv_cache.seq_len = pos + 1;
        pos++;
    }

    printf("\n--- Done: %d tokens generated  ---\n", pos - n_prompt);
}

/* ============================================================
   MAIN
 * COMMAND LINE INTERFACE
 * Usage: ./gpt2_edge "prompt" [max_tokens=64] [temp=0.7] [top_p=0.9]
 * ============================================================ */
int main(int argc, char *argv[]) {
    printf("=================================\n");
    printf("  GPT-2 Edge Inference Engine\n");
    printf("  NileAGI — Pure C99, No Deps\n");
    printf("=================================\n\n");

    /* Default parameters */
    const char *prompt     = "Hello, world!";
    int         max_tokens = 64;
    float       temperature = 0.7f;
    float       top_p       = 0.9f;

    /* Model and tokenizer file paths */
    const char *model_path   = "gpt2_124m.bin";
    const char *encoder_path = "encoder.json";
    const char *bpe_path     = "vocab.bpe";

    /* Parse command line */
    if (argc >= 2) prompt       = argv[1];
    if (argc >= 3) max_tokens   = atoi(argv[2]);
    if (argc >= 4) temperature  = atof(argv[3]);
    if (argc >= 5) top_p        = atof(argv[4]);

    /* Validate args */
    if (max_tokens <= 0) max_tokens = 64;
    if (max_tokens > GPT2_SEQ_LEN - 10) max_tokens = GPT2_SEQ_LEN - 10;
    if (temperature < 0.0f) temperature = 0.7f;
    if (top_p <= 0.0f || top_p > 1.0f) top_p = 0.9f;

    printf("[CONFIG] prompt: \"%s\"\n", prompt);
    printf("[CONFIG] max_tokens=%d, temperature=%.2f, top_p=%.2f\n\n",
           max_tokens, temperature, top_p);

    /* Seed RNG with current time */
    rng_seed((uint64_t)time(NULL));

#ifdef _OPENMP
    printf("[INFO] OpenMP enabled: %d threads\n", omp_get_max_threads());
#else
    printf("[INFO] OpenMP disabled (single-threaded)\n");
#endif

    /* Load model weights */
    printf("[INFO] Loading model from: %s\n", model_path);
    load_model(model_path);

    /* Load tokenizer */
    printf("[INFO] Loading tokenizer...\n");
    load_tokenizer(encoder_path, bpe_path);

    /* Allocate KV cache and activation buffers */
    init_kv_cache();
    init_activations();

    /* Run generation */
    generate(prompt, max_tokens, temperature, top_p);

    /* Cleanup */
    free(g_arena.data);
    free(g_kv_cache.k_cache);
    free(g_kv_cache.v_cache);
    free(g_act.x);
    free(g_act.x_norm);
    free(g_act.qkv);
    free(g_act.attn_out);
    free(g_act.ffn_hidden);
    free(g_act.logits);
    free(g_act.attn_scores);

    return 0;
}
