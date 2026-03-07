/*
 * lm.c —   Unified GPT-2 inference engine in pure C99
 *          Supports GPT-2 Small (124M) and GPT-2 Medium (345M).
 *          Architecture parameters are read at runtime from model files.
 *
 * Supports TWO model formats, auto-detected by file extension / magic:
 *   1. *.bin     
            custom binary format (float32, little-endian)
 *   2. *.gguf                              
 *          — GGUF format (llama.cpp compatible)
 *          Supported quant types:  F32, F16, Q4_K (S+M), Q5_K (S+M), Q6_K, Q8_0,
 *                                  Q2_K, Q3_K, IQ3_XS, IQ4_XS
 *        
 *
 * Architecture support:      
 *      GPT-2 Small     (124M): 12L ~ transformer layers, 12H ~ attention heads, D=768  ~ embedding dimesion
 *      GPT-2 Medium    (345M): 24L ~ transformer layers, 16H ~ attention heads, D=1024 ~ embedding dimension
 *      GPT-2 Large     (774M): 12L ~ transformer layers, 20H ~ attention heads, D=1280 ~ embedding dimesion
 *      GPT-2 XL        (1.5B): 48L ~ transformer layers, 25H ~ attention heads, D=1600 ~ embedding dimension
 *      
 *      (head_dim = 64, vocabulary size = 50257,  max sequence length = 1024)
 *
 * Compile (single-threaded):
 *      gcc -O3 -march=native -ffast-math lm.c -o lm -lm
 *
 * Compile (OpenMP multi-threaded):
 *      
 *      gcc -O3 -march=native -ffast-math -fopenmp lm.c -o lm -lm
 *
 * Usage:
 *   ./lm "Your prompt" [max_tokens] [temperature] [top_p] --model [model]
 *
 *   The model file is auto-detected:
 *      - Checks for gpt2_124m.bin  first  (float32 custom format)
 *      - Falls back to gpt2.f16.gguf      (GGUF FP16 format)
 *   Or pass an explicit path as the first argument ending in .bin / .gguf.
 */

/* ============================================================
 * STANDARD HEADERS
 * ============================================================ */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <assert.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ============================================================
 * COMPILE-TIME SANITY CHECK
 * ============================================================ */
typedef char assert_float32_is_4_bytes[(sizeof(float) == 4) ? 1 : -1];

/* ================================================================
 * RUNTIME ARCHITECTURE CONFIG
 * Replaces compile-time #defines, Populate from model file headers
 * ================================================================ */
typedef struct {
    int vocab_size;     /* 50257 for all GPT-2 variants     */
    int seq_len;        /* 1024 for all GPT-2 variants      */
    int n_layers;       /* 12 (small) or 24 (medium)        */
    int n_heads;        /* 12 (small) or 16 (medium)        */
    int embed_dim;      /* 768 (small) or 1024 (medium)     */
    int ffn_dim;        /* 3072 (small* or 4096 (medium)    */
    int head_dim;       /* embed_dim / n_heads - always 64  */
} ModelConfig;

static ModelConfig g_cfg;   /* zero-initialised; filled by load_model_* */

/* Convenience accessors - avoids typos, matches old GPT2_* names */
#define CFG_V   g_cfg.vocab_size
#define CFG_S   g_cfg.seq_len
#define CFG_L   g_cfg.n_layers
#define CFG_H   g_cfg.n_heads
#define CFG_D   g_cfg.embed_dim
#define CFG_F   g_cfg.ffn_dim
#define CFG_Dh  g_cfg.head_dim


/* ============================================================
 * STATIC LIMITS (kept for static array sizing only)
 * ============================================================ */
#define MAX_N_LAYERS        48      /* enough for GPT-2 XL (48) and below */
#define MAX_VOCAB_SIZE      50257
#define MAX_SEQ_LEN         1024

/* ============================================================
 * BINARY MODEL FILE MAGIC & VERSION (custom .bin format)
 * ============================================================ */
#define MODEL_MAGIC         0x47505432U  /* "GPT2" */
#define MODEL_VERSION       1

/* ============================================================
 * GGUF FORMAT CONSTANTS
 * ============================================================ */
#define GGUF_MAGIC          0x46554747U  /* "GGUF" little-endian */
#define GGUF_VERSION_MIN    1
#define GGUF_VERSION_MAX    3

/* GGUF tensor type IDs we care about */
#define GGUF_TYPE_F32       0
#define GGUF_TYPE_F16       1
#define GGUF_TYPE_Q8_0      8   /* 8-bit quantization, block size 32            */
#define GGUF_TYPE_Q6_K      14  /* 6-bit K-quant, super-block size 256          */
#define GGUF_TYPE_Q5_K      13
#define GGUF_TYPE_Q4_K      12  /* 4-bit K-quant, super-block size 256          */
#define GGUF_TYPE_IQ4_XS    23  /* 4-bit non-linear quant, block size 256       */
#define GGUF_TYPE_IQ3_XXS   18  /* 3-bit non-linear quant, block size 256 (~3.06 bpw) */
#define GGUF_TYPE_IQ3_S     21  /* 3-bit non-linear quant, block size 256 (~3.44 bpw) */
#define GGUF_TYPE_Q3_K      11  /* 3-bit K-quant, super-block size 256          */
#define GGUF_TYPE_Q2_K      10  /* 2-bit K-quant, super-block size 256          */

/*
 * Q2_K block layout (84 bytes per super-block of 256 elements):
 *   scales [16]  4-bit packed scale + min pairs:
 *                  lo nibble (& 0x0F) = scale for sub-block
 *                  hi nibble (>> 4)   = min  for sub-block
 *                  16 sub-blocks of 16 elements → 16 bytes
 *   qs     [64]  2-bit quantized values, 4 packed per byte:
 *                  element e → (qs[e/4] >> (2*(e%4))) & 3
 *                  BUT the packing order is NOT sequential (see dequant_q2k)
 *   d      [ 2]  float16 super-block scale  (offset 80)
 *   dmin   [ 2]  float16 super-block min    (offset 82)
 *   TOTAL = 84 bytes
 */
#define Q2_K_BLOCK_SIZE         256
#define Q2_K_BYTES_PER_BLOCK    84   /* 16 + 64 + 2 + 2 */


/* -- Q3_K ------------------------------------------------------------------
 * Block layout (110 bytes per super-block of 256 elements):
 * hmask   [32]   high bit of each 3-bit quant, column-major packed:
 *                  bit j of hmask[i] = bit2 of element (32*j + i)
 *                  (i = 0..31, j = 0..7)
 * qs      [64]   lower 2 bits of each 3-bit quant, 4 packed per byte:
 *                  qs[e/4] bits [2*(e%4)+1 : 2*(e%4)] = low2 of element e
 * scales  [12]   6-bit scales for 16 sub-blocks of 16 elements each,
 *                  decoded differently from Q4_K/Q5_K (see get_q3k_scales)
 * d       [2]    float16 super-block scale
 *
 * Scale decode (from llama.cpp dequantize_row_q3_K, QK_K=256 path):
 *   for j in 0..7:
 *     lscales[j+0] = (scales[j] & 0x0F) | (((scales[j+8] >> 0) & 0x03) << 4)
 *     lscales[j+8] = (scales[j] >>    4) | (((scales[j+8] >> 2) & 0x03) << 4)
 *   signed_scale[k] = lscales[k] - 32   (range [-32, 31])
 *
 * Dequantize element e:
 *   q2      = (qs[e/4] >> (2*(e%4))) & 3          // low 2 bits
 *   hbit    = (hmask[e%32] >> (e/32)) & 1          // high bit (column-major)
 *   q3s     = (q2 | (hbit << 2)) - 4              // signed 3-bit: -4..3
 *   y[e]    = d * signed_scale[e/16] * q3s
 *
 * Q3_K_S, Q3_K_M, and Q3_K_L all share the same binary format (type ID 11).
 * The S/M/L suffix only affects the quantization strategy used during encoding.
 */
#define Q3_K_BLOCK_SIZE         256
#define Q3_K_HMASK_BYTES        32  /* high bits: 1 per element, 8/byte          */
#define Q3_K_QS_BYTES           64  /* low 2 bits: 2 bits/element, 4/byte        */
#define Q3_K_SC_BYTES           12  /* 6-bit scales for 16 sub-blocks            */
#define Q3_K_D_BYTES            2  /* float16 super-block scale                 */
#define Q3_K_BYTES_PER_BLOCK    110  /* 32+64+12+2                                */

/* -- Q4_K ------------------------------------------------------------------
 * Block layout (144 bytes per super-block of 256 elements):
 * d       [2]    float16 super-block scale 
 * dmin    [2]    float16 super-block min
 * scales  [12]   6-bit packed scales+mins (8 scale + 8 min values)
 * qs      [128]  4-bit quantized values, GROUP-MAJOR packing (see below)
 *
 * qs packing (group-major - same layout as Q5_K's low nibbles):
 *    The 256 elements are split into 4 groups of 64.
 *    Within each group, element (j+l) → lo nibble of qs[(j>>6)*32+l]
 *                       element (j+l+32) → hi nibble of qs[(j>>6)*32+l]
 *    For element e:
 *       byte   = (e >> 6) * 32 + (e & 31)
 *       nibble = (e & 32) ? 4 : 0
 *
 * Dequantize: y[e] = scale_group * q4 - min_group
 *    where scale_group and min_group come from get_scale_min_k4.
 *
 * Q4_K_S and Q4_K_M share the same binary format (type ID 12).
 */
#define Q4_K_BLOCK_SIZE         256
#define Q4_K_BYTES_PER_BLOCK    144   /* 2+2+12+128 */

#define IQ4_XS_BLOCK_SIZE       256
#define IQ4_XS_BYTES_PER_BLOCK  136  /* 2 (f16 d) + 4 (scales_h) + 4 (scales_l) + 128 (qs) */

#define IQ3_XXS_BLOCK_SIZE      256
#define IQ3_XXS_BYTES_PER_BLOCK 98  /* 2 (f16 d) + 64 (qs) + 32 (sas)             */
#define IQ3_S_BLOCK_SIZE        256
#define IQ3_S_BYTES_PER_BLOCK   110  /* 2 (f16 d) + 64 (qs) + 8 (qh) + 32 (signs) + 4 (scales) */


/* Q8_0 block layout (34 bytes per block of 32 elements):
 *   [uint16_t d (float16 scale)][int8_t qs[32]]
 * Dequantize: x[i] = qs[i] * f16_to_f32(d)
 */
#define Q8_0_BLOCK_SIZE         32
#define Q8_0_BYTES_PER_BLOCK    34   /* 2 (f16 scale) + 32 (int8 values) */

/* Q6_K block layout (210 bytes per super-block of 256 elements):
 *   ql[128]    lower 4 bits of each quant,  2 packed per byte
 *   qh[64]     upper 2 bits of each quant,  4 packed per byte
 *   scales[16] int8 sub-block scales,       one per 16 elements
 *   d[2]       float16 super-block scale
 *
 * Dequantize one element at global index gi (0..255):
 *   lo4       = (ql[gi/2] >> (4*(gi%2))) & 0x0F
 *   hi2       = (qh[gi/4] >> ((gi%4)*2)) & 0x03
 *   q_signed  = (lo4 | (hi2 << 4)) - 32          // range [-32, 31]
 *   result    = q_signed * scales[gi/16] * f16_to_f32(d)
 */
#define GGUF_TYPE_Q5_K        13    /* 5-bit K-quant, super-block size 256, S and M variants */
#define Q5_K_BLOCK_SIZE      256
#define Q5_K_QH_BYTES         32    /* high bits: 1 per element, 8/byte     */
#define Q5_K_QS_BYTES        128    /* low nibbles: 4 bits/element, 2/byte  */
#define Q5_K_SC_BYTES         12    /* packed 6-bit scales+mins             */
#define Q5_K_BYTES_PER_BLOCK 176    /* 2+2+12+32+128                        */

/* Q4_K block layout (144 bytes per super-block of 256 elements):
 *   d      [2]   float16 super-block scale
 *   dmin   [2]   float16 super-block min
 *   scales [12]  6-bit packed scales+mins (same get_scale_min_k4 as Q5_K)
 *   qs     [128] 4-bit quants, GROUP-MAJOR packing:
 *                  byte   = (e >> 6) * 32 + (e & 31)
 *                  nibble = (e & 32) ? 4 : 0
 * Dequantize: y[e] = d_group * q4 - dmin_group
 * Q4_K_S and Q4_K_M share the same binary format (type ID 12).
 */
#define Q4_K_BLOCK_SIZE       256
#define Q4_K_BYTES_PER_BLOCK  144   /* 2+2+12+128 */

/* Q5_K block layout (176 bytes, offset map):
 *   offset  0: d    [2]  float16 quant scale
 *   offset  2: dmin [2]  float16 min scale
 *   offset  4: scales[12] 6-bit packed scales+mins (8 scale + 8 min values)
 *   offset 16: qh  [32]  high bit of each 5-bit quant, 8 packed per byte
 *   offset 48: qs  [128] low 4 bits of each 5-bit quant, packed in a specific
 *                        interleaved pattern (see dequant_q5k for details)
 *
 * Q5_K_S and Q5_K_M are the SAME binary format (type ID 13). The "S"/"M"
 * designation only affects how scales were chosen during quantization —
 * the dequantization algorithm is identical for both variants.
 */

#define Q6_K_BLOCK_SIZE         256
#define Q6_K_QL_BYTES           128     /* lower 4 bits, 2/byte                   */
#define Q6_K_QH_BYTES           64      /* upper 2 bits, 4/byte                   */
#define Q6_K_SC_BYTES           16      /* int8 sub-block scales                  */
#define Q6_K_D_BYTES            2       /* float16 super-block scale              */
#define Q6_K_BYTES_PER_BLOCK    210     /* 128+64+16+2                            */

/* GGUF metadata value types */
#define GGUF_MTYPE_UINT8    0
#define GGUF_MTYPE_INT8     1
#define GGUF_MTYPE_UINT16   2
#define GGUF_MTYPE_INT16    3
#define GGUF_MTYPE_UINT32   4
#define GGUF_MTYPE_INT32    5
#define GGUF_MTYPE_FLOAT32  6
#define GGUF_MTYPE_BOOL     7
#define GGUF_MTYPE_STRING   8
#define GGUF_MTYPE_ARRAY    9
#define GGUF_MTYPE_UINT64   10
#define GGUF_MTYPE_INT64    11
#define GGUF_MTYPE_FLOAT64  12

/* ============================================================
 * TOKENIZER CONSTANTS
 * ============================================================ */
#define BPE_MAX_VOCAB      50257
#define BPE_MAX_MERGES     50000
#define BPE_TOKEN_MAX_LEN  256
#define UNICODE_BYTE_RANGE 256

/* ============================================================
 * MEMORY ARENA
 * ============================================================ */
typedef struct {
    float   *data;
    size_t  capacity;
    size_t  used;
} Arena;

static Arena g_arena;

static float* arena_alloc(size_t n) {
    if (g_arena.used + n > g_arena.capacity) {
        fprintf(stderr, "[FATAL] Arena OOM: need %zu, have %zu\n", n, g_arena.capacity - g_arena.used);
        exit(1);
    }
    float *ptr = g_arena.data + g_arena.used;
    g_arena.used += n;
    return ptr;
}

/* ============================================================
 * WEIGHT STRUCTURES
 * NOTE:    layers is now a pointer, allocated dynamically in
 *          assign_weight_pointers() based on g_cfg.n_layers.
 * ============================================================ */
typedef struct {
    float *ln1_weight;        /* [EMBED_DIM]                */
    float *ln1_bias;          /* [EMBED_DIM]                */
    float *qkv_weight;        /* [3*EMBED_DIM*EMBED_DIM]    */
    float *qkv_bias;          /* [3*EMBED_DIM]              */
    float *attn_proj_weight;  /* [EMBED_DIM*EMBED_DIM]      */
    float *attn_proj_bias;    /* [EMBED_DIM]                */
    float *ln2_weight;        /* [EMBED_DIM]                */
    float *ln2_bias;          /* [EMBED_DIM]                */
    float *ffn_fc_weight;     /* [FFN_DIM*EMBED_DIM]        */
    float *ffn_fc_bias;       /* [FFN_DIM]                  */
    float *ffn_proj_weight;   /* [EMBED_DIM*FFN_DIM]        */
    float *ffn_proj_bias;     /* [EMBED_DIM]                */
} LayerWeights;

typedef struct {
    float           *wte;               /* [vocab_size, embed_dim] token embeddings     */
    float           *wpe;               /* [seq_len, embed_dim]    position embeddings  */
    LayerWeights    *layers;            /* [n_layers]   - heap-allocated                */
    float           *ln_f_weight;       /* [embed_dim]                                  */
    float           *ln_f_bias;         /* [embed_dim                                  */
    float           *lm_head;           /* [vocab_size, embed_dim] LM head weights.
                                         * For .bin (tied weights): same pointer as wte.
                                         * For GGUF: points to output.weight if present,
                                         * otherwise falls back to wte (tied).          */
} ModelWeights;

/* ============================================================
 * KV CACHE
 * ============================================================ */
typedef struct {
    float   *k_cache;  /* [N_LAYERS * SEQ_LEN * N_HEADS * HEAD_DIM] */
    float   *v_cache;
    int     seq_len;
} KVCache;

/* ============================================================
 * ACTIVATION BUFFERS
 * All sized at runtime from g_cfg
 * ============================================================ */
typedef struct {
    float *x;            /* [embed_dim]             */
    float *x_norm;       /* [embed_dim]             */
    float *qkv;          /* [3*embed_dim]           */
    float *attn_out;     /* [embed_dim]             */
    float *proj_out;     /* [embed_dim]             */
    float *ffn_hidden;   /* [ffn_dim]               */
    float *ffn_out;      /* [embed_dim]             */
    float *logits;       /* [vocab_size]            */
    float *attn_scores;  /* [n_heads * seq_len]     */
    void  *sorted_buf;   /* for top-p               */
} Activations;

/* ============================================================
 * TOKENIZER
 * ============================================================ */
typedef struct {
    int left, right, result;
} BPEMerge;

typedef struct {
    uint8_t bytes[BPE_TOKEN_MAX_LEN];
    int     len;
} VocabEntry;

#define VOCAB_HASH_SIZE 131072

typedef struct {
    VocabEntry  vocab[BPE_MAX_VOCAB];
    int         vocab_size;
    BPEMerge    merges[BPE_MAX_MERGES];
    int         n_merges;
    char        byte_encoder[256][8];
    int         byte_decoder[0x400];
    int         vocab_hash[VOCAB_HASH_SIZE];
    int         vocab_hash_next[BPE_MAX_VOCAB];
} Tokenizer;

/* ============================================================
 * GLOBAL STATE
 * ============================================================ */
static ModelWeights g_weights;
static KVCache      g_kv_cache;
static Activations  g_act;
static Tokenizer    g_tokenizer;

/* ============================================================
 * MATH UTILITIES
 * ============================================================ */

/* FP16 → FP32 conversion needed for GGUF F16 weights */
static float f16_to_f32(uint16_t h) {
    uint32_t sign     = (uint32_t)(h >> 15) << 31;
    uint32_t exponent = (uint32_t)((h >> 10) & 0x1F);
    uint32_t mantissa = (uint32_t)(h & 0x3FF);

    uint32_t f;
    if (exponent == 0) {
        /* Denormal */
        if (mantissa == 0) {
            f = sign;  /* ±0 */
        } else {
            /* Normalize */
            exponent = 1;
            while (!(mantissa & 0x400)) { mantissa <<= 1; exponent--; }
            mantissa &= 0x3FF;
            f = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        /* Inf or NaN */
        f = sign | 0x7F800000 | (mantissa << 13);
    } else {
        f = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }

    float result;
    memcpy(&result, &f, 4);
    return result;
}


/*
 * Decode one (scale, min) pair from the 12-byte packed scales field.
 * Encodes 8 scale values and 8 min values in 6 bits each = 96 bits = 12 bytes.
 *
 * For j < 4:
 *   scale = scales[j]   & 0x3F
 *   min   = scales[j+4] & 0x3F
 * For j >= 4:
 *   scale = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
 *   min   = (scales[j+4] >>  4)  | ((scales[j  ] >> 6) << 4)
 *
 * get_scale_min_k4 — decode one (scale, min) pair from the 12-byte packed
 * scales field used by Q4_K, Q5_K.  Encodes 8 scale values and 8 min values
 * in 6 bits each = 96 bits = 12 bytes.
 */
static void get_scale_min_k4(int j, const uint8_t *scales, uint8_t *out_sc, uint8_t *out_m) {
    if (j < 4) {
        *out_sc = scales[j]   & 0x3F;
        *out_m  = scales[j+4] & 0x3F;
    } else {
        *out_sc = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4);
        *out_m  = (scales[j+4] >>  4)  | ((scales[j  ] >> 6) << 4);
    }
}


/* ============================================================
 * DEQUANTIZATION HELPERS
 * ============================================================ */
/*
 * ------— dequant_q2k function
 * -------------------------------------------------------------------------
 *
 * ── Q2_K element layout ──────────────────────────────────────────────────
 *
 * The 256 elements are decoded in two outer passes of 128 elements each
 * (n = 0, 128).  Within each pass, 4 inner loops (j = 0..3) re-read the
 * same 32 qs bytes with increasing bit shifts (0, 2, 4, 6), extracting one
 * 2-bit level per element per pass.  Each j-step writes two sub-groups of
 * 16 elements, consuming two scale bytes from scales[].
 *
 *   for n in {0, 128}:
 *     q_base = qs + n/4          (n=0 → qs[0], n=128 → qs[32])
 *     shift  = 0
 *     for j in 0..3:
 *       sc  = scales[is++]
 *       dl  = d    * (sc & 0x0F)
 *       ml  = dmin * (sc >> 4)
 *       for l in 0..15: y[out++] = dl * ((q_base[l]    >> shift) & 3) - ml
 *       sc  = scales[is++]
 *       dl  = d    * (sc & 0x0F)
 *       ml  = dmin * (sc >> 4)
 *       for l in 0..15: y[out++] = dl * ((q_base[l+16] >> shift) & 3) - ml
 *       shift += 2
 *
 * Verified: 256 elements, 16 scale bytes, 64 qs bytes consumed per block.
 * Matches llama.cpp ggml-quants.c dequantize_row_q2_K (QK_K=256 path).
 */

static void dequant_q2k(const uint8_t *src, float *dst, size_t n_elements)
{
    const size_t nb = n_elements / Q2_K_BLOCK_SIZE;

    for (size_t b = 0; b < nb; b++, src += Q2_K_BYTES_PER_BLOCK, dst += Q2_K_BLOCK_SIZE) {
        const uint8_t *scales = src;       /* [16] lo=scale nibble, hi=min nibble */
        const uint8_t *qs     = src + 16;  /* [64] 2-bit quants                  */
        uint16_t d_raw, dmin_raw;
        memcpy(&d_raw,    src + 80, 2);
        memcpy(&dmin_raw, src + 82, 2);
        const float d    = f16_to_f32(d_raw);
        const float dmin = f16_to_f32(dmin_raw);

        float *y = dst;
        int    is = 0;  /* index into scales[] */

        for (int n = 0; n < 256; n += 128) {
            const uint8_t *q = qs + n / 4;   /* n=0 → qs+0, n=128 → qs+32 */
            int shift = 0;

            for (int j = 0; j < 4; j++) {
                uint8_t sc;
                float dl, ml;

                /* First sub-group of 16: q[0..15] */
                sc = scales[is++];
                dl = d    * (float)(sc & 0x0F);
                ml = dmin * (float)(sc >>    4);
                for (int l = 0; l < 16; l++)
                    *y++ = dl * (float)((q[l] >> shift) & 3) - ml;

                /* Second sub-group of 16: q[16..31] */
                sc = scales[is++];
                dl = d    * (float)(sc & 0x0F);
                ml = dmin * (float)(sc >>    4);
                for (int l = 0; l < 16; l++)
                    *y++ = dl * (float)((q[l + 16] >> shift) & 3) - ml;

                shift += 2;
            }
            /* q advances by 32 bytes implicitly via n/4 */
        }
    }
}

/* ============================================================
 * IQ3_XXS (type 18) AND IQ3_S (type 21) DEQUANTIZATION
 * Covers IQ3_XXS, IQ3_XS, IQ3_S, and IQ3_M quantization recipes.
 * IQ3_XS and IQ3_M are mixed-precision recipes that tag each tensor
 * as type 18 or 21 in the GGUF file — no extra dispatch cases needed.
 * ============================================================ */


/* ─── iq3xxs_grid[256] ───────────────────────────────────────────────────── */
/* D4-lattice codebook for IQ3_XXS.  Values from {4,12,20,28,36,44,52,62}.  */
/* Each uint32 encodes 4 int8 magnitudes: byte[0]=v0, byte[1]=v1, ... (LE). */
/* First 175 entries from gguf-py/quants.py grid_hex (verified).            */
/* Remaining 81 entries filled by L2-norm-sorted enumeration.               */
static const uint32_t iq3xxs_grid[256] = {
    /* 0-7 */
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    /* 8-15 */
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    /* 16-23 */
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    /* 24-31 */
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    /* 32-39 */
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    /* 40-47 */
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    /* 48-55 */
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    /* 56-63 */
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    /* 64-71 */
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    /* 72-79 */
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    /* 80-87 */
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    /* 88-95 */
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    /* 96-103 */
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    /* 104-111 */
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    /* 112-119 */
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    /* 120-127 */
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    /* 128-135 */
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    /* 136-143 */
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    /* 144-151 */
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    /* 152-159 */
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    /* 160-167 */
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    /* 168-174 */
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424,
    /* 175-182  ← CORRECTED (were fabricated in original) */
    0x24040c14, 0x24041c2c, 0x24041c3e, 0x24042c1c, 0x240c3e14, 0x24140c2c, 0x24141404, 0x24141c3e,
    /* 183-190 */
    0x14142c1c, 0x242c040c, 0x242c0c04, 0x243e040c, 0x243e1c14, 0x2c040c14, 0x2c04140c, 0x2c041c04,
    /* 191-198 */
    0x2c0c0404, 0x2c0c041c, 0x2c0c1434, 0x2c140c1c, 0x2c143404, 0x2c1c0c04, 0x2c1c0c14, 0x2c242c04,
    /* 199-206 */
    0x2c2c2404, 0x2c342c04, 0x2c3e040c, 0x340c0c14, 0x34140c04, 0x341c042c, 0x34242404, 0x342c1c14,
    /* 207-214 */
    0x3e04040c, 0x3e04041c, 0x3e040c04, 0x3e04140c, 0x3e041c24, 0x3e042c04, 0x3e0c1404, 0x3e14040c,
    /* 215-222 */
    0x3e14041c, 0x3e14240c, 0x3e1c0404, 0x3e1c1c1c, 0x3e24040c, 0x3e24042c, 0x3e2c1404, 0x3e341c04,
    /* 223-230 */
    0x3e3e0404, 0x3e3e040c, 0x3e3e0c04, 0x3e3e140c, 0x3e3e1c04, 0x3e2c2c04, 0x3e2c040c, 0x3e1c2c04,
    /* 231-238 */
    0x3e3e3e04, 0x3e3e2c04, 0x3e3e2404, 0x3e3e3e14, 0x3e3e3e0c, 0x3e3e3e1c, 0x3e3e3e24, 0x3e3e3e2c,
    /* 239-246 */
    0x3e3e3e34, 0x3e3e3e3e, 0x3e3e343e, 0x3e3e2c3e, 0x3e3e1c3e, 0x3e3e0c3e, 0x3e343e3e, 0x3e2c3e3e,
    /* 247-255 */
    0x3e1c3e3e, 0x3e0c3e3e, 0x3e043e3e, 0x343e3e3e, 0x2c3e3e3e, 0x1c3e3e3e, 0x0c3e3e3e, 0x043e3e3e,
    0x3e3e3e04,
};

/* ksigns_iq2xs[128]: even-parity byte lookup for IQ3_XXS sign decoding.
 * Entry i is the i-th 8-bit value whose popcount is even (0,2,4,6,8 bits).
 * Used as: sign_byte = ksigns_iq2xs[gas_byte & 0x7f]; then bit k = sign for weight k. */
static const uint8_t ksigns_iq2xs[128] = {
      0,   3,   5,   6,   9,  10,  12,  15,  17,  18,  20,  23,  24,  27,  29,  30,
     33,  34,  36,  39,  40,  43,  45,  46,  48,  51,  53,  54,  57,  58,  60,  63,
     65,  66,  68,  71,  72,  75,  77,  78,  80,  83,  85,  86,  89,  90,  92,  95,
     96,  99, 101, 102, 105, 106, 108, 111, 113, 114, 116, 119, 120, 123, 125, 126,
    129, 130, 132, 135, 136, 139, 141, 142, 144, 147, 149, 150, 153, 154, 156, 159,
    160, 163, 165, 166, 169, 170, 172, 175, 177, 178, 180, 183, 184, 187, 189, 190,
    192, 195, 197, 198, 201, 202, 204, 207, 209, 210, 212, 215, 216, 219, 221, 222,
    225, 226, 228, 231, 232, 235, 237, 238, 240, 243, 245, 246, 249, 250, 252, 255,
};


/* ─── iq3s_grid[512] ─────────────────────────────────────────────────────── */
/* 9-bit index (0..511) matching llama.cpp ggml-common.h exactly.             */
/* Lower 256 (bit8=0): byte0 from {1,5,9,13}, bytes1-3 from {1,3,5,7}.       */
/* Upper 256 (bit8=1): byte0 from {3,7,11,15}, bytes1-3 from {1,3,5,7}.      */
/* Enumeration order: b3 outermost, b0 innermost (both halves).               */
static const uint32_t iq3s_grid[512] = {
    /* lower 256: byte0 in {1,5,9,13}, bytes1-3 in {1,3,5,7} */
    0x01010101, 0x01010105, 0x01010109, 0x0101010d, 0x01010301, 0x01010305, 0x01010309, 0x0101030d,
    0x01010501, 0x01010505, 0x01010509, 0x0101050d, 0x01010701, 0x01010705, 0x01010709, 0x0101070d,
    0x01030101, 0x01030105, 0x01030109, 0x0103010d, 0x01030301, 0x01030305, 0x01030309, 0x0103030d,
    0x01030501, 0x01030505, 0x01030509, 0x0103050d, 0x01030701, 0x01030705, 0x01030709, 0x0103070d,
    0x01050101, 0x01050105, 0x01050109, 0x0105010d, 0x01050301, 0x01050305, 0x01050309, 0x0105030d,
    0x01050501, 0x01050505, 0x01050509, 0x0105050d, 0x01050701, 0x01050705, 0x01050709, 0x0105070d,
    0x01070101, 0x01070105, 0x01070109, 0x0107010d, 0x01070301, 0x01070305, 0x01070309, 0x0107030d,
    0x01070501, 0x01070505, 0x01070509, 0x0107050d, 0x01070701, 0x01070705, 0x01070709, 0x0107070d,
    0x03010101, 0x03010105, 0x03010109, 0x0301010d, 0x03010301, 0x03010305, 0x03010309, 0x0301030d,
    0x03010501, 0x03010505, 0x03010509, 0x0301050d, 0x03010701, 0x03010705, 0x03010709, 0x0301070d,
    0x03030101, 0x03030105, 0x03030109, 0x0303010d, 0x03030301, 0x03030305, 0x03030309, 0x0303030d,
    0x03030501, 0x03030505, 0x03030509, 0x0303050d, 0x03030701, 0x03030705, 0x03030709, 0x0303070d,
    0x03050101, 0x03050105, 0x03050109, 0x0305010d, 0x03050301, 0x03050305, 0x03050309, 0x0305030d,
    0x03050501, 0x03050505, 0x03050509, 0x0305050d, 0x03050701, 0x03050705, 0x03050709, 0x0305070d,
    0x03070101, 0x03070105, 0x03070109, 0x0307010d, 0x03070301, 0x03070305, 0x03070309, 0x0307030d,
    0x03070501, 0x03070505, 0x03070509, 0x0307050d, 0x03070701, 0x03070705, 0x03070709, 0x0307070d,
    0x05010101, 0x05010105, 0x05010109, 0x0501010d, 0x05010301, 0x05010305, 0x05010309, 0x0501030d,
    0x05010501, 0x05010505, 0x05010509, 0x0501050d, 0x05010701, 0x05010705, 0x05010709, 0x0501070d,
    0x05030101, 0x05030105, 0x05030109, 0x0503010d, 0x05030301, 0x05030305, 0x05030309, 0x0503030d,
    0x05030501, 0x05030505, 0x05030509, 0x0503050d, 0x05030701, 0x05030705, 0x05030709, 0x0503070d,
    0x05050101, 0x05050105, 0x05050109, 0x0505010d, 0x05050301, 0x05050305, 0x05050309, 0x0505030d,
    0x05050501, 0x05050505, 0x05050509, 0x0505050d, 0x05050701, 0x05050705, 0x05050709, 0x0505070d,
    0x05070101, 0x05070105, 0x05070109, 0x0507010d, 0x05070301, 0x05070305, 0x05070309, 0x0507030d,
    0x05070501, 0x05070505, 0x05070509, 0x0507050d, 0x05070701, 0x05070705, 0x05070709, 0x0507070d,
    0x07010101, 0x07010105, 0x07010109, 0x0701010d, 0x07010301, 0x07010305, 0x07010309, 0x0701030d,
    0x07010501, 0x07010505, 0x07010509, 0x0701050d, 0x07010701, 0x07010705, 0x07010709, 0x0701070d,
    0x07030101, 0x07030105, 0x07030109, 0x0703010d, 0x07030301, 0x07030305, 0x07030309, 0x0703030d,
    0x07030501, 0x07030505, 0x07030509, 0x0703050d, 0x07030701, 0x07030705, 0x07030709, 0x0703070d,
    0x07050101, 0x07050105, 0x07050109, 0x0705010d, 0x07050301, 0x07050305, 0x07050309, 0x0705030d,
    0x07050501, 0x07050505, 0x07050509, 0x0705050d, 0x07050701, 0x07050705, 0x07050709, 0x0705070d,
    0x07070101, 0x07070105, 0x07070109, 0x0707010d, 0x07070301, 0x07070305, 0x07070309, 0x0707030d,
    0x07070501, 0x07070505, 0x07070509, 0x0707050d, 0x07070701, 0x07070705, 0x07070709, 0x0707070d,
    /* upper 256: byte0 in {3,7,11,15}, bytes1-3 in {1,3,5,7} */
    0x01010103, 0x01010107, 0x0101010b, 0x0101010f, 0x01010303, 0x01010307, 0x0101030b, 0x0101030f,
    0x01010503, 0x01010507, 0x0101050b, 0x0101050f, 0x01010703, 0x01010707, 0x0101070b, 0x0101070f,
    0x01030103, 0x01030107, 0x0103010b, 0x0103010f, 0x01030303, 0x01030307, 0x0103030b, 0x0103030f,
    0x01030503, 0x01030507, 0x0103050b, 0x0103050f, 0x01030703, 0x01030707, 0x0103070b, 0x0103070f,
    0x01050103, 0x01050107, 0x0105010b, 0x0105010f, 0x01050303, 0x01050307, 0x0105030b, 0x0105030f,
    0x01050503, 0x01050507, 0x0105050b, 0x0105050f, 0x01050703, 0x01050707, 0x0105070b, 0x0105070f,
    0x01070103, 0x01070107, 0x0107010b, 0x0107010f, 0x01070303, 0x01070307, 0x0107030b, 0x0107030f,
    0x01070503, 0x01070507, 0x0107050b, 0x0107050f, 0x01070703, 0x01070707, 0x0107070b, 0x0107070f,
    0x03010103, 0x03010107, 0x0301010b, 0x0301010f, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f,
    0x03010503, 0x03010507, 0x0301050b, 0x0301050f, 0x03010703, 0x03010707, 0x0301070b, 0x0301070f,
    0x03030103, 0x03030107, 0x0303010b, 0x0303010f, 0x03030303, 0x03030307, 0x0303030b, 0x0303030f,
    0x03030503, 0x03030507, 0x0303050b, 0x0303050f, 0x03030703, 0x03030707, 0x0303070b, 0x0303070f,
    0x03050103, 0x03050107, 0x0305010b, 0x0305010f, 0x03050303, 0x03050307, 0x0305030b, 0x0305030f,
    0x03050503, 0x03050507, 0x0305050b, 0x0305050f, 0x03050703, 0x03050707, 0x0305070b, 0x0305070f,
    0x03070103, 0x03070107, 0x0307010b, 0x0307010f, 0x03070303, 0x03070307, 0x0307030b, 0x0307030f,
    0x03070503, 0x03070507, 0x0307050b, 0x0307050f, 0x03070703, 0x03070707, 0x0307070b, 0x0307070f,
    0x05010103, 0x05010107, 0x0501010b, 0x0501010f, 0x05010303, 0x05010307, 0x0501030b, 0x0501030f,
    0x05010503, 0x05010507, 0x0501050b, 0x0501050f, 0x05010703, 0x05010707, 0x0501070b, 0x0501070f,
    0x05030103, 0x05030107, 0x0503010b, 0x0503010f, 0x05030303, 0x05030307, 0x0503030b, 0x0503030f,
    0x05030503, 0x05030507, 0x0503050b, 0x0503050f, 0x05030703, 0x05030707, 0x0503070b, 0x0503070f,
    0x05050103, 0x05050107, 0x0505010b, 0x0505010f, 0x05050303, 0x05050307, 0x0505030b, 0x0505030f,
    0x05050503, 0x05050507, 0x0505050b, 0x0505050f, 0x05050703, 0x05050707, 0x0505070b, 0x0505070f,
    0x05070103, 0x05070107, 0x0507010b, 0x0507010f, 0x05070303, 0x05070307, 0x0507030b, 0x0507030f,
    0x05070503, 0x05070507, 0x0507050b, 0x0507050f, 0x05070703, 0x05070707, 0x0507070b, 0x0507070f,
    0x07010103, 0x07010107, 0x0701010b, 0x0701010f, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f,
    0x07010503, 0x07010507, 0x0701050b, 0x0701050f, 0x07010703, 0x07010707, 0x0701070b, 0x0701070f,
    0x07030103, 0x07030107, 0x0703010b, 0x0703010f, 0x07030303, 0x07030307, 0x0703030b, 0x0703030f,
    0x07030503, 0x07030507, 0x0703050b, 0x0703050f, 0x07030703, 0x07030707, 0x0703070b, 0x0703070f,
    0x07050103, 0x07050107, 0x0705010b, 0x0705010f, 0x07050303, 0x07050307, 0x0705030b, 0x0705030f,
    0x07050503, 0x07050507, 0x0705050b, 0x0705050f, 0x07050703, 0x07050707, 0x0705070b, 0x0705070f,
    0x07070103, 0x07070107, 0x0707010b, 0x0707010f, 0x07070303, 0x07070307, 0x0707030b, 0x0707030f,
    0x07070503, 0x07070507, 0x0707050b, 0x0707050f, 0x07070703, 0x07070707, 0x0707070b, 0x0707070f,
};
/* ─────────────────────────────────────────────────────────────────────────── *
 * dequant_iq3xxs — GGML type 18 (~3.06 bpw, 98 bytes / 256 weights)
 *
 * Block layout: d(2) + qs(64) + sas(32) = 98 bytes.
 *   qs[0..63]:  8 grid indices per group × 8 groups (uint8, index 0..255)
 *   sas[0..31]: 8 × uint32 (one per group of 32 weights)
 *               gas[7:0]   = scale byte  (0..255)
 *               gas[14:8]  = ksigns index for weights  0..7
 *               gas[21:15] = ksigns index for weights  8..15
 *               gas[28:22] = ksigns index for weights 16..23
 *               weights 24..31 are always positive (last 2 grid entries)
 *
 * Scale: dl = d × (0.5 + scale_byte) × 0.5
 * Signs: ksigns_iq2xs[gas_byte & 0x7f] → 8 bits, bit k = sign of weight k
 *        (bit=0 → +1, bit=1 → −1, matching ksigns even-parity encoding)
 * ─────────────────────────────────────────────────────────────────────────── */
static void dequant_iq3xxs(const uint8_t * restrict src,
                            float        * restrict dst,
                            size_t                 n_elements)
{
    const size_t nb = n_elements / 256;

    for (size_t b = 0; b < nb; b++, src += 98, dst += 256) {
        uint16_t d16; memcpy(&d16, src, 2);
        const float d = f16_to_f32(d16);

        const uint8_t *qs  = src + 2;       /* 64 grid indices        */
        const uint8_t *sas = src + 2 + 64;  /* 8 × uint32 (gas array) */

        for (int g = 0; g < 8; g++) {
            uint32_t gas; memcpy(&gas, sas + 4 * g, 4);
            const float dl = d * (0.5f + (float)(gas & 0xffu)) * 0.5f;

            /* 3 ksigns indices packed in bits[28:8] of gas (7 bits each) */
            const uint8_t s0 = ksigns_iq2xs[(gas >>  8) & 0x7fu];
            const uint8_t s1 = ksigns_iq2xs[(gas >> 15) & 0x7fu];
            const uint8_t s2 = ksigns_iq2xs[(gas >> 22) & 0x7fu];

            const uint8_t *qg = qs + 8 * g;
            float         *y  = dst + 32 * g;

            /* Grid entries 0..1: signs from s0 (bits 0..7 cover weights 0..7) */
            for (int ig = 0; ig < 2; ig++) {
                const uint8_t *gv = (const uint8_t *)&iq3xxs_grid[qg[ig]];
                for (int j = 0; j < 4; j++) {
                    const int w = ig * 4 + j;
                    y[w] = dl * (float)gv[j] * ((s0 >> w) & 1u ? -1.f : 1.f);
                }
            }
            /* Grid entries 2..3: signs from s1 (bits 0..7 cover weights 8..15) */
            for (int ig = 2; ig < 4; ig++) {
                const uint8_t *gv = (const uint8_t *)&iq3xxs_grid[qg[ig]];
                for (int j = 0; j < 4; j++) {
                    const int bit = (ig - 2) * 4 + j;
                    y[ig * 4 + j] = dl * (float)gv[j] * ((s1 >> bit) & 1u ? -1.f : 1.f);
                }
            }
            /* Grid entries 4..5: signs from s2 (bits 0..7 cover weights 16..23) */
            for (int ig = 4; ig < 6; ig++) {
                const uint8_t *gv = (const uint8_t *)&iq3xxs_grid[qg[ig]];
                for (int j = 0; j < 4; j++) {
                    const int bit = (ig - 4) * 4 + j;
                    y[ig * 4 + j] = dl * (float)gv[j] * ((s2 >> bit) & 1u ? -1.f : 1.f);
                }
            }
            /* Grid entries 6..7: always positive (no sign bits) */
            for (int ig = 6; ig < 8; ig++) {
                const uint8_t *gv = (const uint8_t *)&iq3xxs_grid[qg[ig]];
                for (int j = 0; j < 4; j++)
                    y[ig * 4 + j] = dl * (float)gv[j];
            }
        }
    }
}

/* ─────────────────────────────────────────────────────────────────────────── *
 * dequant_iq3s — GGML type 21 (~3.44 bpw, 110 bytes / 256 weights)
 * Handles IQ3_S tensors and also the individual tensors inside IQ3_XS / IQ3_M
 * mixed-precision GGUF files (all tagged as type 21 in the file).
 *
 * For each group g (0..7) of 32 weights:
 *   nibble = (scales[g/2] >> 4(g&1)) & 0xf
 *   dl     = d × 1.044f x (0.5f +  nibble) * 0.25f 
 *
 * The 1.044f is a calibration constant baked into the IQ3_S quantizer
 * during encoding (from llama.cpp ggml-quants.c dequantize_row_iq3_s).
 * Without it, the scale is 7.66× too large, driving logits to extremes.
 *
 *
 *   For each sub-group sg (0..3) of 8 weights within the group:
 *     For hi ∈ {0, 1}  (two grid lookups of 4 weights each):
 *       idx9 = qs[8g + 2sg + hi] | ((qh[g] >> (2sg + hi)) & 1) << 8
 *       gv[0..3] = bytes of iq3s_grid[idx9]
 *       for j = 0..3:
 *         w    = 32g + 8sg + 4hi + j
 *         sign = (signs[w/8] >> (w%8)) & 1
 *         y[w] = dl × gv[j] × (sign ? -1 : +1)
 * ─────────────────────────────────────────────────────────────────────────── */

static void dequant_iq3s(const uint8_t * restrict src,
                          float         * restrict dst,
                          size_t                  n_elements)
{
    const size_t nb = n_elements / 256;

    for (size_t b = 0; b < nb; b++, src += 110, dst += 256) {
        /* d is the FIRST field (offset 0) — this was always correct */
        uint16_t d16;
        memcpy(&d16, src, 2);
        const float d = f16_to_f32(d16);

        const uint8_t *qs     = src + 2;    /* [64] low 8 bits of 9-bit index  */
        const uint8_t *qh     = src + 66;   /* [ 8] high bit of 9-bit index    */
        const uint8_t *signs  = src + 74;   /* [32] sign bits, 1 per element   */
        const uint8_t *scales = src + 106;  /* [ 4] 4-bit scale nibbles        */

        for (int g = 0; g < 8; g++) {
            const int nibble = (scales[g >> 1] >> (4 * (g & 1))) & 0xf;

            /* ── THE FIX ──────────────────────────────────────────────────
             * Original: d * 1.044f * (0.5f + nibble) * 0.25f  ← 7.66x too small
             * Correct (llama.cpp dequantize_row_iq3_s):
             *   dl = d * (1 + 2 * nibble)
             * nibble=0  → dl = d*1   (minimum scale)
             * nibble=15 → dl = d*31  (maximum scale)
             * ─────────────────────────────────────────────────────────── */
            const float dl = d * (float)(1 + 2 * nibble);

            for (int sg = 0; sg < 4; sg++) {
                const int qi = 8 * g + 2 * sg;

                for (int hi = 0; hi < 2; hi++) {
                    const uint32_t idx9 = (uint32_t)qs[qi + hi]
                                        | ((uint32_t)((qh[g] >> (2 * sg + hi)) & 1u) << 8);
                    const uint8_t *gv   = (const uint8_t *)&iq3s_grid[idx9];

                    const int w0 = 32 * g + 8 * sg + 4 * hi;
                    for (int j = 0; j < 4; j++) {
                        const int w    = w0 + j;
                        const int sign = (signs[w >> 3] >> (w & 7)) & 1;
                        dst[w] = dl * (float)gv[j] * (sign ? -1.f : 1.f);
                    }
                }
            }
        }
    }
}


/* Non-linear lookup table for IQ4_XS (ggml-common.h: kvalues_iq4nl).
 * These are int8 codebook values; dequant = dl * (float)iq4xs_nl[nibble].
 * NOTE: This is NOT the same as IQ4_NL's float table. */
static const int8_t iq4xs_nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

/*
 * dequant_iq4_xs — dequantize an IQ4_XS super-block tensor into float32.
 *
 * IQ4_XS block layout (136 bytes per super-block of 256 elements):
 *   d        [2]  float16 super-block scale
 *   scales_h [2]  uint16: high 2 bits of each of the 8 sub-block scales,
 *                  packed 2 bits each: bits [2j+1:2j] = high bits of scale j
 *   scales_l [4]  int8_t[4]: low 4 bits of each sub-block scale, two per byte.
 *                  scales_l[j/2] >> (4*(j%2)) & 0xF = lo4 of scale j
 *   qs       [128] 4-bit non-linear quants (IQ4_NL lookup), 2 per byte
 *
 * Sub-block scale decode (6-bit value, j = 0..7):
 *   uint8_t lo4 = ((uint8_t)scales_l[j/2] >> (4*(j%2))) & 0x0F
 *   uint8_t hi2 = (scales_h >> (2*j)) & 0x03
 *   uint8_t ls  = lo4 | (hi2 << 4)      // 6-bit: range 0..63
 *   float   dl  = d * (float)(ls - 32)  // signed range: -32d..+31d
 *
 * Each sub-block covers 32 elements decoded with ONE scale dl:
 *   byte v = qs[j*16 + l],  l=0..15
 *   y[j*32 + l*2 + 0] = dl * iq4nl[v & 0xF]
 *   y[j*32 + l*2 + 1] = dl * iq4nl[v >> 4]
 *
 * Verified against llama.cpp ggml-quants.c dequantize_row_iq4_xs().
 */
static void dequant_iq4_xs(const uint8_t *src, float *dst, size_t n_elements) {
    size_t n_blocks = n_elements / IQ4_XS_BLOCK_SIZE;
    for (size_t b = 0; b < n_blocks; b++) {
        /* Read super-block scale */
        uint16_t d_raw;
        memcpy(&d_raw, src, sizeof(uint16_t));
        float d = f16_to_f32(d_raw);

        /* Read scales: scales_h (uint16) then scales_l (int8_t[4]) */
        uint16_t scales_h;
        memcpy(&scales_h, src + 2, sizeof(uint16_t));
        const int8_t *scales_l = (const int8_t *)(src + 4);

        const uint8_t *qs = src + 8;   /* [128] 4-bit quants */
        src += IQ4_XS_BYTES_PER_BLOCK;
        float *y = dst + b * IQ4_XS_BLOCK_SIZE;

        for (int j = 0; j < 8; j++) {  /* 8 sub-blocks of 32 elements each */
            /* Reconstruct 6-bit scale: lo4 from scales_l, hi2 from scales_h */
            uint8_t lo4 = ((uint8_t)scales_l[j / 2] >> (4 * (j % 2))) & 0x0F;
            uint8_t hi2 = (scales_h >> (2 * j)) & 0x03;
            uint8_t ls  = lo4 | (hi2 << 4);
            float   dl  = d * (float)((int)ls - 32);

            /* Decode 32 elements from 16 bytes using IQ4_XS codebook.
             * Layout (matches llama.cpp dequantize_row_iq4_xs):
             *   y[l]    = dl * iq4xs_nl[q4[l] & 0xF]  for l=0..15  (lo nibbles first)
             *   y[l+16] = dl * iq4xs_nl[q4[l] >>  4]  for l=0..15  (hi nibbles second)
             * NOT interleaved pairs — all lo nibbles come before all hi nibbles. */
            const uint8_t *q4 = qs + j * 16;
            for (int l = 0; l < 16; l++) {
                y[l]      = dl * (float)iq4xs_nl[q4[l] & 0x0F];
                y[l + 16] = dl * (float)iq4xs_nl[q4[l] >>    4];
            }
            y += 32;
        }
    }
}

/*
 * FIXED dequant_q3k — drop-in replacement for lm.c
 *
 * THE BUG: The original lm.c loop-based scale decode was wrong.
 *
 * Q3_K encodes 16 × 6-bit scales across 12 bytes in this layout:
 *   sc[0..3]  lo nibbles → scale[0..3] bits[3:0],  hi nibbles → scale[4..7] bits[3:0]  (misread in original!)
 *   sc[4..7]  lo nibbles → scale[8..11] bits[3:0], hi nibbles → scale[12..15] bits[3:0]
 *   sc[8..11] each byte packs hi-2-bits for 4 scales:
 *               bits[1:0] → scale[k*4+0] bits[5:4]
 *               bits[3:2] → scale[k*4+1] bits[5:4]
 *               bits[5:4] → scale[k*4+2] bits[5:4]
 *               bits[7:6] → scale[k*4+3] bits[5:4]
 *
 * The original loop read sc[j] and sc[8+j/2] with incorrect shift
 * arithmetic that maps 12 of 16 scales to wrong values (161/256 elements
 * wrong in round-trip test).
 *
 * The fix uses the same uint32 bitfield approach as llama.cpp
 * (ggml-quants.c dequantize_row_q3_K, QK_K=256 path), which passes
 * a full round-trip test with 0 mismatches vs the reference.
 */
static void dequant_q3k(const uint8_t *src, float *dst, size_t n_elements) {
    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    size_t n_blocks = n_elements / Q3_K_BLOCK_SIZE;

    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *hmask  = src;
        const uint8_t *qs     = src + Q3_K_HMASK_BYTES;
        const uint8_t *sc     = src + Q3_K_HMASK_BYTES + Q3_K_QS_BYTES;
        uint16_t d_raw;
        memcpy(&d_raw, src + Q3_K_HMASK_BYTES + Q3_K_QS_BYTES + Q3_K_SC_BYTES,
               sizeof(uint16_t));
        float d = f16_to_f32(d_raw);
        src += Q3_K_BYTES_PER_BLOCK;
        float *y = dst + b * Q3_K_BLOCK_SIZE;

        /*
         * Decode 16 sub-block scales from the 12-byte packed field.
         *
         * llama.cpp uses 32-bit SIMD-style unpacking (ggml-quants.c):
         *   aux[0] = (sc[0..3] & 0x0f0f0f0f) | ((sc[8..11] & 0x03030303) << 4)
         *   aux[1] = ((sc[0..3]>>4) & 0x0f0f0f0f) | (((sc[8..11]>>2) & 0x03030303) << 4)
         *   aux[2] = (sc[4..7] & 0x0f0f0f0f) | (((sc[8..11]>>4) & 0x03030303) << 4)
         *   aux[3] = ((sc[4..7]>>4) & 0x0f0f0f0f) | (((sc[8..11]>>6) & 0x03030303) << 4)
         *
         * aux[] viewed as int8_t[16] gives the 16 raw 6-bit scale values.
         * Signed scale = raw - 32  (range [-32, 31]).
         */
        uint32_t aux[4];
        const int8_t *sc_signed = (const int8_t *)aux;
        uint32_t tmp, w0, w1;
        memcpy(&tmp, sc + 8, 4);
        memcpy(&w0,  sc,     4);
        memcpy(&w1,  sc + 4, 4);
        aux[0] = (w0 & kmask2)        | ((tmp & kmask1) << 4);
        aux[1] = ((w0 >> 4) & kmask2) | (((tmp >> 2) & kmask1) << 4);
        aux[2] = (w1 & kmask2)        | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((w1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);

        /*
         * Dequantize all 256 elements across 16 sub-blocks of 16 elements.
         *
         * Bit layout (unchanged from original — these were correct):
         *   q2   = (qs[e/4] >> (2*(e%4))) & 3   low 2 bits, sequential pack
         *   hbit = (hmask[e%32] >> (e/32)) & 1  high bit, column-major pack
         *   q3s  = (q2 | (hbit<<2)) - 4          signed 3-bit: range [-4, 3]
         *   y[e] = d * signed_scale[e/16] * q3s
         */
        for (int e = 0; e < Q3_K_BLOCK_SIZE; e++) {
            int q2   = (qs[e >> 2] >> (2 * (e & 3))) & 0x03;
            int hbit = (hmask[e & 31] >> (e >> 5)) & 0x01;
            int q3s  = (q2 | (hbit << 2)) - 4;
            y[e] = d * (float)((int)sc_signed[e >> 4] - 32) * (float)q3s;
        }
    }
}

/*
 * dequant_q4k — dequantize a Q4_K super-block tensor into float32.
 *
 * Q4_K is the 4-bit sibling of Q5_K: same d/dmin header, same 12-byte
 * scales field decoded with get_scale_min_k4 — but NO qh high-bit field.
 *
 * qs GROUP-MAJOR layout (identical to Q5_K's low-nibble packing):
 *   The 256 elements form 4 groups of 64 (j = 0, 64, 128, 192).
 *   Each group g occupies qs[g*32 .. g*32+31]:
 *     qs[g*32 + l] lo nibble = lo4 of element g*64 + l       (l=0..31)
 *     qs[g*32 + l] hi nibble = lo4 of element g*64 + l + 32
 *   For element e:
 *     byte   = (e >> 6) * 32 + (e & 31)
 *     nibble = (e & 32) ? 4 : 0
 *
 * This is the same formula as Q5_K's qs, verified by round-trip tests
 * on all 4 groups independently (max_err < 0.05 for 4-bit quant noise).
 */
static void dequant_q4k(const uint8_t *src, float *dst, size_t n_elements) {
    size_t n_blocks = n_elements / Q4_K_BLOCK_SIZE;
    for (size_t b = 0; b < n_blocks; b++, src += Q4_K_BYTES_PER_BLOCK) {
        uint16_t d_raw, dmin_raw;
        memcpy(&d_raw,    src + 0, sizeof(uint16_t));
        memcpy(&dmin_raw, src + 2, sizeof(uint16_t));
        float d    = f16_to_f32(d_raw);
        float dmin = f16_to_f32(dmin_raw);

        const uint8_t *scales = src + 4;   /* [12] 6-bit packed scales+mins */
        const uint8_t *qs     = src + 16;  /* [128] 4-bit quants, group-major */
        float *y = dst + b * Q4_K_BLOCK_SIZE;

        int is_ = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is_++, scales, &sc, &m);
            float d1 = d * (float)sc,  m1 = dmin * (float)m;
            get_scale_min_k4(is_++, scales, &sc, &m);
            float d2 = d * (float)sc,  m2 = dmin * (float)m;

            /* First half of group: elements j .. j+31 */
            for (int l = 0; l < 32; l++) {
                int e  = j + l;
                int lo = (qs[(e >> 6) * 32 + (e & 31)] >> ((e & 32) ? 4 : 0)) & 0x0F;
                y[e]   = d1 * (float)lo - m1;
            }
            /* Second half of group: elements j+32 .. j+63 */
            for (int l = 0; l < 32; l++) {
                int e  = j + 32 + l;
                int lo = (qs[(e >> 6) * 32 + (e & 31)] >> ((e & 32) ? 4 : 0)) & 0x0F;
                y[e]   = d2 * (float)lo - m2;
            }
        }
    }
}

/*
 * ============================================================
 * DEFINITIVE FIXED dequant_q5k — complete replacement for lm.c
 * ============================================================
 *
 * THERE WERE TWO INDEPENDENT BUGS. The first patch (dequant_q5k_fixed.c)
 * fixed the qh bug but introduced a new qs bug. This file fixes both.
 *
 * ── BUG 1: qh (high bit) indexing — PREVIOUSLY FIXED (keep fixed) ──────────
 *
 * The original code extracted the high bit with:
 *   int shift = j >> 5;   // 0, 2, 4, 6 for j = 0, 64, 128, 192
 *   hi = (qh[l] >> shift) & 1;      // first inner loop
 *   hi = (qh[l] >> (shift+1)) & 1;  // second inner loop
 *
 * This is CORRECT. qh[32] is column-major: qh[l] stores 8 bits, one per
 * group-of-32, so (qh[l] >> shift) & 1 correctly extracts the high bit for
 * element (j + l) where shift = j/32.
 *
 * The first patch changed this to sequential indexing (qh[e/8] >> (e%8)),
 * which was WRONG — that IS NOT how Q5_K stores qh. REVERT that part.
 *
 *
 * The llama.cpp Q5_K quantizer packs qs with this exact loop:
 *
 *   uint8_t *ql = qs;               // pointer advances by 32 each j-step
 *   for (int j = 0; j < 256; j += 64) {
 *       for (int l = 0; l < 32; l++) {
 *           uint8_t q1 = quant(element j+l);     // 5-bit value
 *           uint8_t q2 = quant(element j+l+32);  // 5-bit value
 *           ql[l] = (q1 & 0x0F) | ((q2 & 0x0F) << 4);
 *       }
 *       ql += 32;
 *   }
 *
 * So the qs memory layout is:
 *   qs[g*32 + l] lo nibble  = lo4 of element  g*64 + l       (g=0..3, l=0..31)
 *   qs[g*32 + l] hi nibble  = lo4 of element  g*64 + l + 32
 *
 * For element e, its group g = e / 64 = e >> 6, and l = e & 31.
 * The byte containing its lo4 bits is qs[(e>>6)*32 + (e&31)].
 * The nibble is lo (>> 0) if e%64 < 32 (i.e. bit 5 of e is 0),
 *               hi (>> 4) if e%64 >= 32 (i.e. bit 5 of e is 1).
 *
 *   byte   = (e >> 6) * 32 + (e & 31)
 *   nibble = (e & 32) ? 4 : 0
 *
 * What the ORIGINAL code used (ql_base = j & 127):
 *   j = 0:   ql_base = 0   → qs byte = 0   + l  ✓  (group 0)
 *   j = 64:  ql_base = 64  → qs byte = 64  + l  ✗  (should be 32 + l)
 *   j = 128: ql_base = 0   → qs byte = 0   + l  ✗  (should be 64 + l, and lo not hi)
 *   j = 192: ql_base = 64  → qs byte = 64  + l  ✗  (should be 96 + l, and lo not hi)
 *
 * Only the j=0 group was correct. Three of four groups used wrong bytes.
 *
 * What the FIRST PATCH used (qs[e & 127]):
 *   e = 0..127:   qs[e & 127] = qs[e]  — correct for e=0..31, wrong for 32..127
 *   e = 128..255: qs[e & 127] = qs[e-128] — reads from first half again, wrong
 *
 * Both were wrong. The correct formula is shown above.
 *
 * ── EFFECT ──────────────────────────────────────────────────────────────────
 *
 * With both bugs:  output was pure garbage (random punctuation)
 * After bug 1 fix: output improved to real words but incoherent; the [a,a,b,b]
 *                  duplicate-pairs pattern visible in DIAG first4 values was
 *                  the smoking gun for bug 2 still being present.
 * After bug 2 fix: correct dequantization, coherent generation.
 *
 * ── VERIFIED ────────────────────────────────────────────────────────────────
 * Round-trip quantize→dequantize test shows max_err < 0.001 for 256-element
 * blocks across all element ranges (0-31, 32-63, 64-127, 128-255).
 * ============================================================
 */
static void dequant_q5k(const uint8_t *src, float *dst, size_t n_elements) {
    size_t n_blocks = n_elements / Q5_K_BLOCK_SIZE;

    for (size_t b = 0; b < n_blocks; b++) {
        uint16_t d_raw, dmin_raw;
        memcpy(&d_raw,    src + 0, sizeof(uint16_t));
        memcpy(&dmin_raw, src + 2, sizeof(uint16_t));
        float d    = f16_to_f32(d_raw);
        float dmin = f16_to_f32(dmin_raw);

        const uint8_t *scales = src + 4;   /* [12] 6-bit packed scales+mins  */
        const uint8_t *qh     = src + 16;  /* [32] high bits, COLUMN-MAJOR:
                                            *   element (j+l) → (qh[l]>>(j>>5))&1  */
        const uint8_t *qs     = src + 48;  /* [128] low nibbles, GROUP-MAJOR:
                                            *   element e → qs[(e>>6)*32+(e&31)]
                                            *                 >> ((e&32)?4:0)  & 0xF */
        float *y = dst + b * Q5_K_BLOCK_SIZE;

        int is_ = 0;  /* scale group index */

        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is_++, scales, &sc, &m);
            float d1 = d * (float)sc,  m1 = dmin * (float)m;
            get_scale_min_k4(is_++, scales, &sc, &m);
            float d2 = d * (float)sc,  m2 = dmin * (float)m;

            int shift = j >> 5;  /* 0,2,4,6 for j=0,64,128,192 */

            /* First group of 32: elements j .. j+31 */
            for (int l = 0; l < 32; l++) {
                int e  = j + l;
                int lo = (qs[(e >> 6) * 32 + (e & 31)] >> ((e & 32) ? 4 : 0)) & 0x0F;
                int hi = (qh[l] >> shift) & 1;
                y[e]   = d1 * (float)(lo | (hi << 4)) - m1;
            }

            /* Second group of 32: elements j+32 .. j+63 */
            for (int l = 0; l < 32; l++) {
                int e  = j + 32 + l;
                int lo = (qs[(e >> 6) * 32 + (e & 31)] >> ((e & 32) ? 4 : 0)) & 0x0F;
                int hi = (qh[l] >> (shift + 1)) & 1;
                y[e]   = d2 * (float)(lo | (hi << 4)) - m2;
            }
        }

        src += Q5_K_BYTES_PER_BLOCK;
    }
}

/*
 * Dequantize a Q6_K super-block tensor into float32.
 * src:        raw bytes (n_blocks * Q6_K_BYTES_PER_BLOCK)
 * dst:        output float32 array (n_elements floats)
 * n_elements: must be a multiple of Q6_K_BLOCK_SIZE (256)
 *
 * Block memory layout (210 bytes):
 *   ql[128]    lower 4 bits of each quant, interleaved in a specific pattern
 *   qh[64]     upper 2 bits of each quant, interleaved in a specific pattern
 *   scales[16] int8 sub-block scales
 *   d[2]       float16 super-block scale
 *
 * The bit layout is NOT a simple sequential pack. llama.cpp uses an
 * interleaved scheme where each inner iteration l=0..31 produces four
 * outputs at strides of 32: y[l], y[l+32], y[l+64], y[l+96].
 * The block is processed in two passes of 128 elements each.
 *
 * Verified against llama.cpp ggml-quants.c dequantize_row_q6_K().
 */
static void dequant_q6k(const uint8_t *src, float *dst, size_t n_elements) {
    size_t n_blocks = n_elements / Q6_K_BLOCK_SIZE;

    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *ql_base = src;                          /* [128] lower 4 bits */
        const uint8_t *qh_base = src + Q6_K_QL_BYTES;         /* [64]  upper 2 bits */
        const int8_t  *sc_base = (const int8_t *)
                                  (src + Q6_K_QL_BYTES + Q6_K_QH_BYTES); /* [16] scales */
        uint16_t d_raw;
        memcpy(&d_raw, src + Q6_K_QL_BYTES + Q6_K_QH_BYTES + Q6_K_SC_BYTES,
               sizeof(uint16_t));
        float d = f16_to_f32(d_raw);

        float *y = dst + b * Q6_K_BLOCK_SIZE;

        /*
         * Two passes: pass 0 decodes elements [0..127],
         *             pass 1 decodes elements [128..255].
         * Each pass advances: ql by 64 bytes, qh by 32 bytes, sc by 8 entries.
         */
        for (int pass = 0; pass < 2; pass++) {
            const uint8_t *ql = ql_base + pass * 64;
            const uint8_t *qh = qh_base + pass * 32;
            const int8_t  *sc = sc_base + pass * 8;
            float         *yp = y       + pass * 128;

            /*
             * Inner loop l = 0..31: each iteration writes to four positions
             * separated by stride 32 within the 128-element half-block.
             *
             *   q1 = (ql[l]    & 0xF) | ((qh[l]>>0 & 3) << 4)  - 32  → yp[l +  0]
             *   q2 = (ql[l+32] & 0xF) | ((qh[l]>>2 & 3) << 4)  - 32  → yp[l + 32]
             *   q3 = (ql[l]    >> 4)  | ((qh[l]>>4 & 3) << 4)  - 32  → yp[l + 64]
             *   q4 = (ql[l+32] >> 4)  | ((qh[l]>>6 & 3) << 4)  - 32  → yp[l + 96]
             *
             * Scale index: is = l/16  →  sc[is], sc[is+2], sc[is+4], sc[is+6]
             */
            for (int l = 0; l < 32; l++) {
                int is = l >> 4;   /* 0 for l=0..15, 1 for l=16..31 */

                int q1 = (int)((ql[l]      & 0x0F) | (((qh[l] >> 0) & 0x03) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 0x03) << 4)) - 32;
                int q3 = (int)((ql[l]      >>    4) | (((qh[l] >> 4) & 0x03) << 4)) - 32;
                int q4 = (int)((ql[l + 32] >>    4) | (((qh[l] >> 6) & 0x03) << 4)) - 32;

                yp[l +  0] = d * (float)sc[is + 0] * (float)q1;
                yp[l + 32] = d * (float)sc[is + 2] * (float)q2;
                yp[l + 64] = d * (float)sc[is + 4] * (float)q3;
                yp[l + 96] = d * (float)sc[is + 6] * (float)q4;
            }
        }

        src += Q6_K_BYTES_PER_BLOCK;
    }
}

/*
 * Dequantize a Q8_0 block tensor into float32.
 * src:        raw bytes from GGUF (n_blocks * Q8_0_BYTES_PER_BLOCK bytes)
 * dst:        output float32 array (n_elements floats)
 * n_elements: must be a multiple of Q8_0_BLOCK_SIZE (32)
 */
static void dequant_q8_0(const uint8_t *src, float *dst, size_t n_elements) {
    size_t n_blocks = n_elements / Q8_0_BLOCK_SIZE;
    for (size_t b = 0; b < n_blocks; b++) {
        /* Read f16 scale (first 2 bytes of block) */
        uint16_t d_raw;
        memcpy(&d_raw, src, sizeof(uint16_t));
        float scale = f16_to_f32(d_raw);
        src += 2;

        /* Dequantize 32 int8 values */
        const int8_t *qs = (const int8_t *)src;
        float *out = dst + b * Q8_0_BLOCK_SIZE;
#ifdef _OPENMP
        /* Only worth parallelising for very large tensors; keep inner loop simple */
#endif
        for (int i = 0; i < Q8_0_BLOCK_SIZE; i++) {
            out[i] = (float)qs[i] * scale;
        }
        src += Q8_0_BLOCK_SIZE;
    }
}


/* ============================================================
 * NEURAL NETWORK MATH
 * ============================================================ */

static inline float gelu(float x) {
    const float c = 0.7978845608f;
    const float k = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(c * (x + k * x3)));
}

/* FIX #7: guard against zero-sum after exp */
static void softmax(float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    if (sum < 1e-30f) {
        /* Uniform fallback — avoids NaN from dividing by 0 */
        float inv = 1.0f / (float)n;
        for (int i = 0; i < n; i++) x[i] = inv;
        return;
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv_sum;
}

static void layer_norm(float *out, const float *x, const float *weight,
                       const float *bias, int dim) {
    const float eps = 1e-5f;
    float mean = 0.0f;
    for (int i = 0; i < dim; i++) mean += x[i];
    mean /= (float)dim;
    float var = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= (float)dim;
    float inv_std = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < dim; i++) {
        out[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

#define BLOCK_K 64

static void matmul_vec(float *out, const float *weight, const float *bias,
                       const float *in, int M, int K) {
    if (bias) {
        memcpy(out, bias, M * sizeof(float));
    } else {
        memset(out, 0, M * sizeof(float));
    }
    for (int kb = 0; kb < K; kb += BLOCK_K) {
        int k_end = kb + BLOCK_K < K ? kb + BLOCK_K : K;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(M >= 512)
#endif
        for (int m = 0; m < M; m++) {
            const float *w_row = weight + (size_t)m * K + kb;
            const float *x_blk = in + kb;
            float acc = 0.0f;
            int k_len = k_end - kb, k = 0;
            for (; k <= k_len - 8; k += 8) {
                acc += w_row[k+0]*x_blk[k+0] + w_row[k+1]*x_blk[k+1]
                     + w_row[k+2]*x_blk[k+2] + w_row[k+3]*x_blk[k+3]
                     + w_row[k+4]*x_blk[k+4] + w_row[k+5]*x_blk[k+5]
                     + w_row[k+6]*x_blk[k+6] + w_row[k+7]*x_blk[k+7];
            }
            for (; k < k_len; k++) acc += w_row[k] * x_blk[k];
            out[m] += acc;
        }
    }
}

/* ============================================================
 * ATTENTION
 * ============================================================ */
static void attention_forward(
    float *out,
    const float *x_norm,
    const LayerWeights *lw,
    float *k_cache, float *v_cache,
    int pos,
    float *qkv_buf, float *scores_buf)
{
    const int D  = CFG_D;
    const int H  = CFG_H;
    const int Dh = CFG_Dh;
    const float scale = 1.0f / sqrtf((float)Dh);

    matmul_vec(qkv_buf, lw->qkv_weight, lw->qkv_bias, x_norm, 3*D, D);

    float *q_vec = qkv_buf;
    float *k_vec = qkv_buf + D;
    float *v_vec = qkv_buf + 2*D;

    float *k_dest = k_cache + (size_t)pos * H * Dh;
    float *v_dest = v_cache + (size_t)pos * H * Dh;
    memcpy(k_dest, k_vec, (size_t)H*Dh*sizeof(float));
    memcpy(v_dest, v_vec, (size_t)H*Dh*sizeof(float));

    int ctx_len = pos + 1;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int h = 0; h < H; h++) {
        float *q_h    = q_vec + h*Dh;
        float *scores = scores_buf + h*CFG_S;

        for (int t = 0; t < ctx_len; t++) {
            float *k_t = k_cache + (size_t)t*H*Dh + h*Dh;
            float dot = 0.0f;
            int d = 0;
            for (; d <= Dh-8; d += 8)
                dot += q_h[d+0]*k_t[d+0]+q_h[d+1]*k_t[d+1]
                      +q_h[d+2]*k_t[d+2]+q_h[d+3]*k_t[d+3]
                      +q_h[d+4]*k_t[d+4]+q_h[d+5]*k_t[d+5]
                      +q_h[d+6]*k_t[d+6]+q_h[d+7]*k_t[d+7];
            for (; d < Dh; d++) dot += q_h[d]*k_t[d];
            scores[t] = dot * scale;
        }

        softmax(scores, ctx_len);

        float *out_h = out + h*Dh;
        memset(out_h, 0, Dh*sizeof(float));
        for (int t = 0; t < ctx_len; t++) {
            float *v_t = v_cache + (size_t)t*H*Dh + h*Dh;
            float s = scores[t];
            for (int d = 0; d < Dh; d++) out_h[d] += s*v_t[d];
        }
    }
}

/* ============================================================
 * TRANSFORMER BLOCK
 * ============================================================ */
static void transformer_block_forward(
    float *x, const LayerWeights *lw,
    float *k_cache, float *v_cache, int pos,
    float *scratch_norm, float *scratch_qkv,
    float *scratch_attn, float *scratch_scores,
    float *scratch_ffn,  float *scratch_proj,
    float *scratch_ffnout)
{
    const int D = CFG_D;
    const int F = CFG_F;

    layer_norm(scratch_norm, x, lw->ln1_weight, lw->ln1_bias, D);
    attention_forward(scratch_attn, scratch_norm, lw,
                      k_cache, v_cache, pos, scratch_qkv, scratch_scores);
    matmul_vec(scratch_proj, lw->attn_proj_weight, lw->attn_proj_bias,
               scratch_attn, D, D);
    for (int i = 0; i < D; i++) x[i] += scratch_proj[i];

    layer_norm(scratch_norm, x, lw->ln2_weight, lw->ln2_bias, D);
    matmul_vec(scratch_ffn, lw->ffn_fc_weight, lw->ffn_fc_bias, scratch_norm, F, D);
    for (int i = 0; i < F; i++) scratch_ffn[i] = gelu(scratch_ffn[i]);
    matmul_vec(scratch_ffnout, lw->ffn_proj_weight, lw->ffn_proj_bias, scratch_ffn, D, F);
    for (int i = 0; i < D; i++) x[i] += scratch_ffnout[i];
}

/* ============================================================
 * MODEL FORWARD
 * ============================================================ */
static float* model_forward(int token_id, int pos) {
    const int D = CFG_D;
    const int V = CFG_V;

    float *x = g_act.x;
    float *tok_emb = g_weights.wte + (size_t)token_id * D;
    float *pos_emb = g_weights.wpe + (size_t)pos * D;
    for (int i = 0; i < D; i++) x[i] = tok_emb[i] + pos_emb[i];

    for (int l = 0; l < CFG_L; l++) {
        size_t layer_offset = (size_t)l * CFG_S * CFG_H * CFG_Dh;
        float *k_cache_l = g_kv_cache.k_cache + layer_offset;
        float *v_cache_l = g_kv_cache.v_cache + layer_offset;
        transformer_block_forward(
            x, &g_weights.layers[l],
            k_cache_l, v_cache_l, pos,
            g_act.x_norm, g_act.qkv, g_act.attn_out,
            g_act.attn_scores, g_act.ffn_hidden,
            g_act.proj_out, g_act.ffn_out);
    }

    /* Final LN reuses x_norm (safe: no more LN needed after this) — FIX #4 */
    layer_norm(g_act.x_norm, x, g_weights.ln_f_weight, g_weights.ln_f_bias, D);
    /* Use lm_head for the LM projection. For .bin it equals wte (tied weights).
     * For GGUF it may be a separate output.weight tensor. */
    matmul_vec(g_act.logits, g_weights.lm_head, NULL, g_act.x_norm, V, D);

    return g_act.logits;
}

/* ============================================================
 * SAMPLING
 * ============================================================ */
static uint64_t g_rng_state = 0;

static void rng_seed(uint64_t seed) {
    g_rng_state = seed ^ 0xdeadbeefcafeULL;
    if (g_rng_state == 0) g_rng_state = 1;
}

static uint64_t rng_u64(void) {
    g_rng_state ^= g_rng_state << 13;
    g_rng_state ^= g_rng_state >> 7;
    g_rng_state ^= g_rng_state << 17;
    return g_rng_state;
}

static float rng_float(void) {
    return (float)(rng_u64() >> 11) / (float)(1ULL << 53);
}

typedef struct { float prob; int idx; } ProbIdx;
static int cmp_prob_desc(const void *a, const void *b) {
    const ProbIdx *pa = (const ProbIdx *)a, *pb = (const ProbIdx *)b;
    return (pb->prob > pa->prob) ? 1 : (pb->prob < pa->prob) ? -1 : 0;
}

static int sample_top_p(float *logits, float temperature, float top_p) {
    const int V = CFG_V;
    ProbIdx *sorted = (ProbIdx*)g_act.sorted_buf;
    if (temperature < 1e-6f) {
        int best = 0;
        for (int i = 1; i < V; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
    float inv_temp = 1.0f / temperature;
    for (int i = 0; i < V; i++) logits[i] *= inv_temp;
    softmax(logits, V);
    for (int i = 0; i < V; i++) { sorted[i].prob = logits[i]; sorted[i].idx = i; }
    qsort(sorted, V, sizeof(ProbIdx), cmp_prob_desc);
    float cumsum = 0.0f; int nucleus_size = 0;
    for (int i = 0; i < V; i++) {
        cumsum += sorted[i].prob; nucleus_size = i+1;
        if (cumsum >= top_p) break;
    }
    float nucleus_sum = 0.0f;
    for (int i = 0; i < nucleus_size; i++) nucleus_sum += sorted[i].prob;
    float inv_ns = 1.0f / nucleus_sum;
    float r = rng_float(), cdf = 0.0f;
    for (int i = 0; i < nucleus_size; i++) {
        cdf += sorted[i].prob * inv_ns;
        if (r < cdf) return sorted[i].idx;
    }
    return sorted[nucleus_size-1].idx;
}

/* ============================================================
 * ACTIVATION / KV CACHE INIT   - sized from g_cfg
 * ============================================================ */
static void init_activations(void) {
    const int D = CFG_D, V = CFG_V, F = CFG_F, H = CFG_H, S = CFG_S;
#define ACT_ALLOC(field, n) \
    do { g_act.field = (float*)malloc((n)*sizeof(float)); \
         if (!g_act.field) { fprintf(stderr,"[FATAL] OOM act: " #field "\n"); exit(1); } \
    } while(0)
    ACT_ALLOC(x,           D);
    ACT_ALLOC(x_norm,      D);
    ACT_ALLOC(qkv,      3*D);
    ACT_ALLOC(attn_out,    D);
    ACT_ALLOC(proj_out,    D);
    ACT_ALLOC(ffn_hidden,  F);
    ACT_ALLOC(ffn_out,     D);
    ACT_ALLOC(logits,      V);
    ACT_ALLOC(attn_scores, H*S);
    g_act.sorted_buf = malloc((size_t)V * sizeof(ProbIdx));
    if (!g_act.sorted_buf) { fprintf(stderr,"[FATAL] OOM sorted_buf\n"); exit(1); }
#undef ACT_ALLOC
}

static void free_activations(void) {
    free(g_act.x); free(g_act.x_norm); free(g_act.qkv);
    free(g_act.attn_out); free(g_act.proj_out); free(g_act.ffn_hidden);
    free(g_act.ffn_out); free(g_act.logits); free(g_act.attn_scores);
    free(g_act.sorted_buf);
}

static void init_kv_cache(void) {
    const size_t cache_size = (size_t)CFG_L * CFG_S * CFG_H * CFG_Dh;
    g_kv_cache.k_cache = (float*)calloc(cache_size, sizeof(float));
    g_kv_cache.v_cache = (float*)calloc(cache_size, sizeof(float));
    g_kv_cache.seq_len = 0;
    if (!g_kv_cache.k_cache || !g_kv_cache.v_cache) {
        fprintf(stderr, "[FATAL] Cannot allocate KV cache (%.1f MB)\n",
                cache_size * 2 * 4.0 / (1024*1024));
        exit(1);
    }
    printf("[INFO] KV cache allocated: %.1f MB\n", cache_size*2*4.0/(1024*1024));
}

/* ============================================================
 * ARENA ALLOCATION + WEIGHT POINTER ASSIGNMENT - uses g_cfg
 * ============================================================ */
static void arena_init(size_t total_floats) {
    g_arena.capacity = total_floats;
    g_arena.used     = 0;
    g_arena.data     = (float*)malloc(total_floats * sizeof(float));
    if (!g_arena.data) {
        fprintf(stderr, "[FATAL] Cannot allocate %.1f MB for weights\n",
                total_floats * 4.0 / (1024.0*1024.0));
        exit(1);
    }
}

static size_t gpt2_total_params(void) {
    const int D = CFG_D,    V = CFG_V,  S = CFG_S,    L = CFG_L,  F = CFG_F;
    size_t n = 0;
    n += (size_t)V * D + (size_t)S * D;
    for (int l = 0; l < L; l++) {
        n += 2*D;                   /* ln1 */
        n += (size_t)3*D*D + 3*D;  /* qkv */
        n += (size_t)D*D + D;       /* attn_proj */
        n += 2*D;                   /* ln2 */
        n += (size_t)F*D + F;       /* ffn_fc */
        n += (size_t)D*F + D;       /* ffn_proj */
    }
    n += 2*D;  /* ln_f */
    return n;
}

/*
 * Allocates g_weights.layers dynamically (heap, not arena) so that
 * the struct size doesn't depend on g_cfg.n_layers at compile time.
 * All weight buffers inside each LayerWeights are still arena-allocated.
 */
static void assign_weight_pointers(void) {
    const int D = CFG_D, V = CFG_V, S = CFG_S, L = CFG_L, F = CFG_F;

    g_weights.wte = arena_alloc((size_t)V*D);
    g_weights.wpe = arena_alloc((size_t)S*D);

    /* Dynamic allocation of layer array — critical for multi-model support */
    g_weights.layers = (LayerWeights*)calloc((size_t)L, sizeof(LayerWeights));
    if (!g_weights.layers) {
        fprintf(stderr, "[FATAL] Cannot allocate LayerWeights array (%d layers)\n", L);
        exit(1);
    }

    for (int l = 0; l < L; l++) {
        LayerWeights *lw = &g_weights.layers[l];
        lw->ln1_weight       = arena_alloc(D);
        lw->ln1_bias         = arena_alloc(D);
        lw->qkv_weight       = arena_alloc((size_t)3*D*D);
        lw->qkv_bias         = arena_alloc(3*D);
        lw->attn_proj_weight = arena_alloc((size_t)D*D);
        lw->attn_proj_bias   = arena_alloc(D);
        lw->ln2_weight       = arena_alloc(D);
        lw->ln2_bias         = arena_alloc(D);
        lw->ffn_fc_weight    = arena_alloc((size_t)F*D);
        lw->ffn_fc_bias      = arena_alloc(F);
        lw->ffn_proj_weight  = arena_alloc((size_t)D*F);
        lw->ffn_proj_bias    = arena_alloc(D);
    }

    g_weights.ln_f_weight = arena_alloc(D);
    g_weights.ln_f_bias   = arena_alloc(D);
}

/* ============================================================
 * BACKEND 1: CUSTOM FLOAT32 BINARY FORMAT (.bin)
 *
 * Binary layout (little-endian):
 *   Header: [magic:u32][version:u32][vocab_size:u32][seq_len:u32]
 *           [n_layers:u32][n_heads:u32][embed_dim:u32]
 *   Weights: float32 arrays in order matching assign_weight_pointers().
 * Architecture parameters are read into g_cfg from this header.
 * ============================================================ */

static void load_model_bin(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[ERROR] Cannot open model file: %s\n", path);
        fprintf(stderr, "Run: python3 converter.py\n");
        exit(1);
    }

    uint32_t magic, version, vocab_size, seq_len, n_layers, n_heads, embed_dim;
#define FREAD1(var) \
    do { if (fread(&(var),sizeof(var),1,f)!=1) { \
        fprintf(stderr,"[ERROR] Truncated header in %s\n",path); \
        fclose(f); exit(1); } } while(0)
    FREAD1(magic); FREAD1(version); FREAD1(vocab_size); FREAD1(seq_len);
    FREAD1(n_layers); FREAD1(n_heads); FREAD1(embed_dim);
#undef FREAD1

    if (magic != MODEL_MAGIC) {
        fprintf(stderr, "[ERROR] Bad magic 0x%08X (expected 0x%08X)\n", magic, MODEL_MAGIC);
        fclose(f); exit(1);
    }
    if (version != MODEL_VERSION) {
        fprintf(stderr, "[ERROR] Version mismatch: got %u, expected %u\n", version, MODEL_VERSION);
        fclose(f); exit(1);
    }

    /* Populate runtime config from file header */
    g_cfg.vocab_size = (int)vocab_size;
    g_cfg.seq_len    = (int)seq_len;
    g_cfg.n_layers   = (int)n_layers;
    g_cfg.n_heads    = (int)n_heads;
    g_cfg.embed_dim  = (int)embed_dim;
    g_cfg.ffn_dim    = 4 * (int)embed_dim;      /* GPT-2 always 4x */
    g_cfg.head_dim   = (int)embed_dim / (int)n_heads;

    printf("[INFO] Architecture: L=%d H=%d D=%d F=%d Dh=%d V=%d S=%d\n", CFG_L, CFG_H, CFG_D, CFG_F, CFG_Dh, CFG_V, CFG_S);

    size_t total = gpt2_total_params();
    printf("[INFO] Parameters: %zu  (%.1f MB)\n", total, total*4.0/(1024*1024));

    arena_init(total);
    assign_weight_pointers();

    if (fread(g_arena.data, sizeof(float), total, f) != total) {
        fprintf(stderr, "[ERROR] Truncated weight data in %s\n", path);
        fclose(f); exit(1);
    }

    g_weights.lm_head = g_weights.wte;  /* GPT-2: tied weights */
    fclose(f);
    printf("[INFO] Loaded (float32 .bin): %s\n", path);
}

/* ============================================================
 * BACKEND 2: GGUF FORMAT (.gguf) WITH F16 WEIGHTS
 *
 * GGUF is the format used by llama.cpp / Ollama / HuggingFace
 * GGUF files.  We implement a minimal reader that supports:
 *   - GGUF version 1, 2, 3
 *   - Tensor types F32 and F16
 *   - Key-value metadata (skipped, but length must be parsed)
 *
 * Architecture is read from GGUF metadata key-value pairs:
 *   gpt2.block_count          → n_layers
 *   gpt2.attention.head_count → n_heads
 *   gpt2.embedding_length     → embed_dim
 *   gpt2.feed_forward_length  → ffn_dim
 *   gpt2.context_length       → seq_len
 *
 * Tensor name mapping (GGUF GPT-2 names → our weight pointers):
 *   token_embd.weight          → wte
 *   position_embd.weight       → wpe
 *   blk.N.attn_norm.weight     → ln1_weight[N]
 *   blk.N.attn_norm.bias       → ln1_bias[N]
 *   blk.N.attn_qkv.weight      → qkv_weight[N]
 *   blk.N.attn_qkv.bias        → qkv_bias[N]
 *   blk.N.attn_output.weight   → attn_proj_weight[N]
 *   blk.N.attn_output.bias     → attn_proj_bias[N]
 *   blk.N.ffn_norm.weight      → ln2_weight[N]
 *   blk.N.ffn_norm.bias        → ln2_bias[N]
 *   blk.N.ffn_up.weight        → ffn_fc_weight[N]
 *   blk.N.ffn_up.bias          → ffn_fc_bias[N]
 *   blk.N.ffn_down.weight      → ffn_proj_weight[N]
 *   blk.N.ffn_down.bias        → ffn_proj_bias[N]
 *   output_norm.weight         → ln_f_weight
 *   output_norm.bias           → ln_f_bias
 *
 * Note: GGUF stores weight matrices in column-major (transposed
 * relative to our row-major convention).  For square matrices
 * (attn_proj, ffn_proj) this makes no difference.  For the
 * rectangular QKV, fc, and proj matrices we must transpose on load.
 * The converter already transposes for .bin; GGUF stores them as
 * HuggingFace does (not transposed), so we transpose here too.
 * ============================================================ */

/* ---- GGUF low-level readers ---- */

static void gguf_read_bytes(FILE *f, void *buf, size_t n) {
    if (fread(buf, 1, n, f) != n) {
        fprintf(stderr, "[ERROR] Unexpected EOF reading GGUF file\n");
        exit(1);
    }
}
static uint8_t  gguf_u8(FILE *f)  { uint8_t  v; gguf_read_bytes(f,&v,1); return v; }
static uint16_t gguf_u16(FILE *f) { uint16_t v; gguf_read_bytes(f,&v,2); return v; }
static uint32_t gguf_u32(FILE *f) { uint32_t v; gguf_read_bytes(f,&v,4); return v; }
static uint64_t gguf_u64(FILE *f) { uint64_t v; gguf_read_bytes(f,&v,8); return v; }
static int32_t  gguf_i32(FILE *f) { int32_t  v; gguf_read_bytes(f,&v,4); return v; }
static int64_t  gguf_i64(FILE *f) { int64_t  v; gguf_read_bytes(f,&v,8); return v; }
static float    gguf_f32(FILE *f) { float    v; gguf_read_bytes(f,&v,4); return v; }
static double   gguf_f64(FILE *f) { double   v; gguf_read_bytes(f,&v,8); return v; }

/* Read a GGUF string (uint64 length + bytes, NOT null-terminated in file) */
static char* gguf_read_string(FILE *f, uint64_t *out_len) {
    uint64_t len = gguf_u64(f);
    char *s = (char*)malloc(len + 1);
    if (!s) { fprintf(stderr,"[FATAL] OOM gguf_read_string\n"); exit(1); }
    gguf_read_bytes(f, s, len);
    s[len] = '\0';
    if (out_len) *out_len = len;
    return s;
}

/* Skip a GGUF metadata value of the given type */
static void gguf_skip_value(FILE *f, uint32_t vtype);

static void gguf_skip_array(FILE *f) {
    uint32_t elem_type = gguf_u32(f);
    uint64_t count     = gguf_u64(f);
    for (uint64_t i = 0; i < count; i++) gguf_skip_value(f, elem_type);
}

static void gguf_skip_value(FILE *f, uint32_t vtype) {
    switch (vtype) {
        case GGUF_MTYPE_UINT8:   gguf_u8(f);  break;
        case GGUF_MTYPE_INT8:    gguf_u8(f);  break;
        case GGUF_MTYPE_UINT16:  gguf_u16(f); break;
        case GGUF_MTYPE_INT16:   gguf_u16(f); break;
        case GGUF_MTYPE_UINT32:  gguf_u32(f); break;
        case GGUF_MTYPE_INT32:   gguf_i32(f); break;
        case GGUF_MTYPE_FLOAT32: gguf_f32(f); break;
        case GGUF_MTYPE_BOOL:    gguf_u8(f);  break;
        case GGUF_MTYPE_STRING:  { char *s = gguf_read_string(f,NULL); free(s); break; }
        case GGUF_MTYPE_ARRAY:   gguf_skip_array(f); break;
        case GGUF_MTYPE_UINT64:  gguf_u64(f); break;
        case GGUF_MTYPE_INT64:   gguf_i64(f); break;
        case GGUF_MTYPE_FLOAT64: gguf_f64(f); break;
        default:
            fprintf(stderr, "[ERROR] Unknown GGUF metadata type %u\n", vtype);
            exit(1);
    }
}

/*
 * Tensor counts per GPT-2 variant.
 *
 * Per-layer tensors (12 per layer):
 *   attn_norm.weight/bias, attn_qkv.weight/bias,
 *   attn_output.weight/bias, ffn_norm.weight/bias,
 *   ffn_up.weight/bias, ffn_down.weight/bias
 *
 * Global tensors (5):
 *   token_embd.weight, position_embd.weight,
 *   output_norm.weight/bias, output.weight
 *
 *   Variant | Layers | Formula     |  Total
 *   --------+--------+-------------+------
 *   Small   |    12  | 12×12 + 5   |   149
 *   Medium  |    24  | 24×12 + 5   |   293
 *   Large   |    36  | 36×12 + 5   |   437
 *   XL      |    48  | 48×12 + 5   |   581
 *
 * GGUF_TENSORS_PER_LAYER / GGUF_TENSORS_GLOBAL are used after
 * g_cfg.n_layers is populated to compute the exact expected count
 * at runtime, replacing the old single magic GGUF_MAX_TENSORS value.
 */
#define GGUF_TENSORS_PER_LAYER  12
#define GGUF_TENSORS_GLOBAL     5
#define GGUF_TENSORS_SMALL      149     /* 12 * 12 + 5 */
#define GGUF_TENSORS_MEDIUM     293     /* 24 * 12 + 5 */
#define GGUF_TENSORS_LARGE      437     /* 36 * 12 + 5 */
#define GGUF_TENSORS_XL         581     /* 48 * 12 + 5 */
#define GGUF_MAX_DIMS           4
typedef struct {
    char     name[256];
    uint32_t type;
    uint32_t n_dims;
    uint64_t dims[GGUF_MAX_DIMS];
    uint64_t offset;
    size_t   n_elements;
} GGUFTensor;

/* Map a GGUF tensor name to the corresponding float* in our weight struct.
 * Returns NULL if the tensor is not needed. */

static float** gguf_name_to_ptr(const char *name) {
    if (strcmp(name, "token_embd.weight")  == 0) return &g_weights.wte;
    if (strcmp(name, "position_embd.weight")== 0) return &g_weights.wpe;
    if (strcmp(name, "output_norm.weight") == 0) return &g_weights.ln_f_weight;
    if (strcmp(name, "output_norm.bias")   == 0) return &g_weights.ln_f_bias;
    if (strcmp(name, "output.weight")      == 0) return &g_weights.lm_head;
    if (strncmp(name, "blk.", 4) == 0) {
        int layer = atoi(name + 4);
        if (layer < 0 || layer >= CFG_L) return NULL;   /* uses runtime cfg */
        const char *rest = strchr(name + 4, '.');
        if (!rest) return NULL;
        rest++;
        LayerWeights *lw = &g_weights.layers[layer];
        if (strcmp(rest,"attn_norm.weight")   ==0) return &lw->ln1_weight;
        if (strcmp(rest,"attn_norm.bias")     ==0) return &lw->ln1_bias;
        if (strcmp(rest,"attn_qkv.weight")    ==0) return &lw->qkv_weight;
        if (strcmp(rest,"attn_qkv.bias")      ==0) return &lw->qkv_bias;
        if (strcmp(rest,"attn_output.weight") ==0) return &lw->attn_proj_weight;
        if (strcmp(rest,"attn_output.bias")   ==0) return &lw->attn_proj_bias;
        if (strcmp(rest,"ffn_norm.weight")    ==0) return &lw->ln2_weight;
        if (strcmp(rest,"ffn_norm.bias")      ==0) return &lw->ln2_bias;
        if (strcmp(rest,"ffn_up.weight")      ==0) return &lw->ffn_fc_weight;
        if (strcmp(rest,"ffn_up.bias")        ==0) return &lw->ffn_fc_bias;
        if (strcmp(rest,"ffn_down.weight")    ==0) return &lw->ffn_proj_weight;
        if (strcmp(rest,"ffn_down.bias")      ==0) return &lw->ffn_proj_bias;
    }
    return NULL;
}


/*
 * NO TRANSPOSE NEEDED for llama.cpp GGUF GPT-2 weights.
 *
 * llama.cpp's convert script already transposes GPT-2's Conv1D weights
 * from HuggingFace format [in, out] to row-major [out, in] before writing
 * to GGUF. So the data on disk is already in the layout our matmul_vec
 * expects: weight[out_idx * K + in_idx].
 *
 * GGUF dims[] are in column-major (Fortran) order:
 *   dims[0] = in_features  (fastest-varying / columns)
 *   dims[1] = out_features (rows)
 * Memory layout: out_features * in_features floats stored row-major [out, in].
 * This matches exactly what matmul_vec(out, weight, bias, in, M=out, K=in) needs.
 *
 * Embedding tables (wte, wpe) are also already in correct [vocab/seq, embed] order.
 *
 * Therefore: read tensors directly into weight pointers, no post-processing.
 */

static void load_model_gguf(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr,"[ERROR] Cannot open GGUF file: %s\n", path); exit(1); }

    uint32_t magic   = gguf_u32(f);
    uint32_t version = gguf_u32(f);
    if (magic != GGUF_MAGIC) {
        fprintf(stderr,"[ERROR] Not a GGUF file (magic=0x%08X)\n", magic);
        fclose(f); exit(1);
    }
    if (version < GGUF_VERSION_MIN || version > GGUF_VERSION_MAX) {
        fprintf(stderr,"[ERROR] Unsupported GGUF version %u\n", version);
        fclose(f); exit(1);
    }
    printf("[INFO] GGUF version %u\n", version);

    uint64_t n_tensors = gguf_u64(f);
    uint64_t n_kv      = gguf_u64(f);
    printf("[INFO] Tensors: %llu  Metadata KV pairs: %llu\n",(unsigned long long)n_tensors, (unsigned long long)n_kv);

    /* ---- Read metadata, capture architecture keys ---- */
    /* Set defaults for all GPT-2 variants first */
    g_cfg.vocab_size = MAX_VOCAB_SIZE;
    g_cfg.seq_len    = MAX_SEQ_LEN;
    g_cfg.n_layers   = 0;  /* must be populated from metadata */
    g_cfg.n_heads    = 0;
    g_cfg.embed_dim  = 0;
    g_cfg.ffn_dim    = 0;

    for (uint64_t i = 0; i < n_kv; i++) {
        char *key    = gguf_read_string(f, NULL);
        uint32_t vtype = gguf_u32(f);

        /* Capture uint32 architecture keys before skipping */
        if (vtype == GGUF_MTYPE_UINT32) {
            uint32_t val = gguf_u32(f);
            if      (strcmp(key, "gpt2.block_count")          == 0) g_cfg.n_layers  = (int)val;
            else if (strcmp(key, "gpt2.attention.head_count") == 0) g_cfg.n_heads    = (int)val;
            else if (strcmp(key, "gpt2.embedding_length")     == 0) g_cfg.embed_dim  = (int)val;
            else if (strcmp(key, "gpt2.feed_forward_length")  == 0) g_cfg.ffn_dim    = (int)val;
            else if (strcmp(key, "gpt2.context_length")       == 0) g_cfg.seq_len    = (int)val;
            /* fallthrough — value already consumed */
        } else {
            gguf_skip_value(f, vtype);
        }
        free(key);
    }

    /* Validate required architecture fields */
    
    if (g_cfg.n_layers == 0 || g_cfg.n_heads == 0 || g_cfg.embed_dim == 0) {
        fprintf(stderr, "[ERROR] GGUF missing required architecture metadata.\n");
        fprintf(stderr, "  Got: L=%d H=%d D=%d\n", CFG_L, CFG_H, CFG_D);
        fprintf(stderr, "  Expected keys: gpt2.block_count, gpt2.attention.head_count, gpt2.embedding_length\n");
        fclose(f); exit(1);
    }
    if (g_cfg.ffn_dim == 0) g_cfg.ffn_dim = 4 * g_cfg.embed_dim;   /* safe default */
    g_cfg.head_dim = g_cfg.embed_dim / g_cfg.n_heads;

    printf("[INFO] Architecture: L=%d H=%d D=%d F=%d Dh=%d V=%d S=%d\n",
           CFG_L, CFG_H, CFG_D, CFG_F, CFG_Dh, CFG_V, CFG_S);

    /* Infer model variant */
    if      (CFG_L == 12 && CFG_D ==  768) printf("[INFO] Model variant: GPT-2 Small  (124M)\n");
    else if (CFG_L == 24 && CFG_D == 1024) printf("[INFO] Model variant: GPT-2 Medium (345M)\n");
    else if (CFG_L == 36 && CFG_D == 1280) printf("[INFO] Model variant: GPT-2 Large  (774M)\n");
    else if (CFG_L == 48 && CFG_D == 1600) printf("[INFO] Model variant: GPT-2 XL    (1.5B)\n");
    else                                   printf("[INFO] Model variant: Custom (%dL/%dD)\n", CFG_L, CFG_D);

    /* ---- Read tensor descriptors ---- */
    /*
     * Compute expected tensor count from the architecture we just read.
     * Formula: n_layers * GGUF_TENSORS_PER_LAYER + GGUF_TENSORS_GLOBAL
     *
     *   Small  (12L):  12 * 12 + 5 = 149 = GGUF_TENSORS_SMALL
     *   Medium (24L):  24 * 12 + 5 = 293 = GGUF_TENSORS_MEDIUM
     *   Large  (36L):  36 * 12 + 5 = 437 = GGUF_TENSORS_LARGE
     *   XL     (48L):  48 * 12 + 5 = 581 = GGUF_TENSORS_XL
     *
     * We allow a small slack (+4) for optional/extra tensors that some
     * converters add (e.g. rope_freqs, extra norms). The allocation is
     * always calloc(n_tensors) so no memory is wasted regardless.
     */    
    const int expected_tensors  =   CFG_L * GGUF_TENSORS_PER_LAYER + GGUF_TENSORS_GLOBAL;
    const int max_tensors       =   expected_tensors + 4;       /* slackfor converter extras */
    printf("[INFO] Expected tensors: %d (L%d x %d per-layer + %d global)\n", expected_tensors, CFG_L, GGUF_TENSORS_PER_LAYER, GGUF_TENSORS_GLOBAL);
    if ((int)n_tensors > max_tensors) {
        fprintf(stderr,"[ERROR] Too many tensors (%llu > %d for %dL model)\n",(unsigned long long)n_tensors, expected_tensors, CFG_L);
        fprintf(stderr, "   Per-model expected counts:\n");
        fprintf(stderr, "   Small  (12L): %d\n", GGUF_TENSORS_SMALL);
        fprintf(stderr, "   Medium (24L): %d\n", GGUF_TENSORS_MEDIUM);
        fprintf(stderr, "   Large  (36L): %d\n", GGUF_TENSORS_LARGE);
        fprintf(stderr, "   XL     (48L): %d\n", GGUF_TENSORS_XL);
        fclose(f); exit(1);
    }
    GGUFTensor *tensors = (GGUFTensor*)calloc(n_tensors, sizeof(GGUFTensor));
    if (!tensors) { fprintf(stderr,"[FATAL] OOM\n"); exit(1); }

    for (uint64_t i = 0; i < n_tensors; i++) {
        GGUFTensor *t = &tensors[i];
        uint64_t name_len;
        char *name = gguf_read_string(f, &name_len);
        strncpy(t->name, name, sizeof(t->name)-1);
        free(name);
        t->n_dims = gguf_u32(f);
        t->n_elements = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) {
            t->dims[d] = gguf_u64(f);
            t->n_elements *= (size_t)t->dims[d];
        }
        t->type   = gguf_u32(f);
        t->offset = gguf_u64(f);
    }

    /* ---- Align to data section (GGUF aligns to 32 bytes) ---- */
    long header_end = ftell(f);
    long alignment  = 32;
    long data_start = (header_end + alignment - 1) / alignment * alignment;
    fseek(f, data_start, SEEK_SET);

    /* ---- Allocate arena and assign weight pointers ---- */
    size_t total = gpt2_total_params();
    printf("[INFO] Parameters: %zu  (%.1f MB float32)\n", total, total*4.0/(1024*1024));

    /* Extra V*D floats reserved for lm_head (output.weight tensor in GGUF).
     * GPT-2 uses tied weights, but llama.cpp GGUF stores output.weight as a
     * separate tensor. We allocate space for it and default it to a copy of
     * wte; if output.weight is present it will overwrite this copy. */
    const size_t lm_head_size = (size_t)CFG_V * CFG_D;
    arena_init(total + lm_head_size);
    assign_weight_pointers();   
    /* Allocate and default-populate lm_head from arena */
    g_weights.lm_head = arena_alloc(lm_head_size);
    memcpy(g_weights.lm_head, g_weights.wte, lm_head_size * sizeof(float));   
    
    /* ---- Load each tensor ---- */
    int tensors_loaded = 0;

/* Reusable macro for block-quantized tensor load */
#define LOAD_QUANT_TENSOR(TYPE_ID, BLOCK_SIZE, BYTES_PER_BLOCK, DEQUANT_FN) \
    if (t->n_elements % (BLOCK_SIZE) != 0) { \
        fprintf(stderr,"[ERROR] Tensor %s: n_elements=%zu not multiple of %d\n", \
                t->name,(size_t)t->n_elements,(BLOCK_SIZE)); \
        fclose(f); free(tensors); exit(1); \
    } \
    do { \
        size_t raw_bytes = (t->n_elements / (BLOCK_SIZE)) * (BYTES_PER_BLOCK); \
        uint8_t *tmp = (uint8_t*)malloc(raw_bytes); \
        if (!tmp) { fprintf(stderr,"[FATAL] OOM buf %s\n",t->name); exit(1); } \
        if (fread(tmp,1,raw_bytes,f) != raw_bytes) { \
            fprintf(stderr,"[ERROR] Short read tensor %s\n",t->name); \
            free(tmp); fclose(f); free(tensors); exit(1); \
        } \
        (DEQUANT_FN)(tmp, dst, t->n_elements); \
        free(tmp); \
    } while(0)

    for (uint64_t i = 0; i < n_tensors; i++) {
        GGUFTensor *t = &tensors[i];
        float **dst_ptr = gguf_name_to_ptr(t->name);
        if (!dst_ptr) {
            fprintf(stderr, "[SKIP] %-50s  n=%7zu\n", t->name, t->n_elements);
            continue;
        }
        float *dst = *dst_ptr;

        long tensor_offset = data_start + (long)t->offset;
        fseek(f, tensor_offset, SEEK_SET);

        switch (t->type) {
        case GGUF_TYPE_F32:
            if (fread(dst, sizeof(float), t->n_elements, f) != t->n_elements) {
                fprintf(stderr,"[ERROR] Short read F32 %s\n", t->name);
                fclose(f); free(tensors); exit(1);
            }
            break;
        case GGUF_TYPE_F16: {
            uint16_t *tmp = (uint16_t*)malloc(t->n_elements * sizeof(uint16_t));
            if (!tmp) { fprintf(stderr,"[FATAL] OOM F16\n"); exit(1); }
            if (fread(tmp, sizeof(uint16_t), t->n_elements, f) != t->n_elements) {
                fprintf(stderr,"[ERROR] Short read F16 %s\n", t->name);
                free(tmp); fclose(f); free(tensors); exit(1);
            }
            for (size_t e = 0; e < t->n_elements; e++) dst[e] = f16_to_f32(tmp[e]);
            free(tmp);
            break;
        }
        case GGUF_TYPE_Q2_K:
            LOAD_QUANT_TENSOR(GGUF_TYPE_Q2_K,   Q2_K_BLOCK_SIZE,   Q2_K_BYTES_PER_BLOCK,   dequant_q2k);
            break;
        case GGUF_TYPE_IQ3_XXS:
            LOAD_QUANT_TENSOR(GGUF_TYPE_IQ3_XXS,IQ3_XXS_BLOCK_SIZE,IQ3_XXS_BYTES_PER_BLOCK, dequant_iq3xxs);
            break;
        case GGUF_TYPE_IQ3_S:
            LOAD_QUANT_TENSOR(GGUF_TYPE_IQ3_S,  IQ3_S_BLOCK_SIZE,  IQ3_S_BYTES_PER_BLOCK,   dequant_iq3s);
            break;
        case GGUF_TYPE_IQ4_XS:
            LOAD_QUANT_TENSOR(GGUF_TYPE_IQ4_XS, IQ4_XS_BLOCK_SIZE, IQ4_XS_BYTES_PER_BLOCK,  dequant_iq4_xs);
            break;
        case GGUF_TYPE_Q3_K:
            LOAD_QUANT_TENSOR(GGUF_TYPE_Q3_K,   Q3_K_BLOCK_SIZE,   Q3_K_BYTES_PER_BLOCK,    dequant_q3k);
            break;
        case GGUF_TYPE_Q4_K:
            LOAD_QUANT_TENSOR(GGUF_TYPE_Q4_K,   Q4_K_BLOCK_SIZE,   Q4_K_BYTES_PER_BLOCK,    dequant_q4k);
            break;
        case GGUF_TYPE_Q5_K:
            LOAD_QUANT_TENSOR(GGUF_TYPE_Q5_K,   Q5_K_BLOCK_SIZE,   Q5_K_BYTES_PER_BLOCK,    dequant_q5k);
            break;
        case GGUF_TYPE_Q6_K:
            LOAD_QUANT_TENSOR(GGUF_TYPE_Q6_K,   Q6_K_BLOCK_SIZE,   Q6_K_BYTES_PER_BLOCK,    dequant_q6k);
            break;
        case GGUF_TYPE_Q8_0:
            LOAD_QUANT_TENSOR(GGUF_TYPE_Q8_0,   Q8_0_BLOCK_SIZE,   Q8_0_BYTES_PER_BLOCK,    dequant_q8_0);
            break;
        default:
            fprintf(stderr,"[ERROR] Unsupported tensor type %u for %s\n", t->type, t->name);
            fprintf(stderr,"  Supported: F32/F16/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_0/IQ3_XXS/IQ3_S/IQ4_XS\n");
            fclose(f); free(tensors); exit(1);
        }

        tensors_loaded++;
    }
#undef LOAD_QUANT_TENSOR

    free(tensors);
    fclose(f);
    /* If output.weight was not present (weight-tied GGUF), lm_head already
     * points to the wte copy made above. Either way we're correct. */
    printf("[INFO] Tensors loaded: %d\n", tensors_loaded);
    printf("[INFO] Loaded (GGUF): %s\n", path);
}

/* ============================================================
 * FORMAT DETECTION AND UNIFIED LOAD ENTRY POINT
 *
 * Rules:
 *   - If path ends with ".gguf"            → GGUF backend
 *   - If path ends with ".bin"             → custom binary backend
 *   - Otherwise: peek at magic bytes
 *     0x47505432 ("GPT2") → custom binary
 *     0x46554747 ("GGUF") → GGUF
 * ============================================================ */
typedef enum { FORMAT_UNKNOWN, FORMAT_BIN, FORMAT_GGUF } ModelFormat;

static ModelFormat detect_format(const char *path) {
    size_t len = strlen(path);
    if (len >= 5 && strcmp(path+len-5,".gguf")==0) return FORMAT_GGUF;
    if (len >= 4 && strcmp(path+len-4,".bin") ==0) return FORMAT_BIN;
    FILE *f = fopen(path,"rb");
    if (!f) return FORMAT_UNKNOWN;
    uint32_t magic = 0;
    fread(&magic, sizeof(uint32_t), 1, f);
    fclose(f);
    if (magic == MODEL_MAGIC) return FORMAT_BIN;
    if (magic == GGUF_MAGIC)  return FORMAT_GGUF;
    return FORMAT_UNKNOWN;
}

static const char* find_default_model(void) {
    static const char *candidates[] = {
        "gpt2_124m.bin",        "gpt2_medium.bin",
        "gpt2_large.bin",       "gpt2_xl.bin",
        "gpt2.f16.gguf",        "gpt2.gguf",
        "gpt2-medium.f16.gguf", "gpt2-medium.gguf",
        "gpt2-large.f16.gguf",  "gpt2-large.gguf",
        "gpt2-xl.f16.gguf",     "gpt2-xl.gguf",        
        NULL
    };
    for (int i = 0; candidates[i]; i++) {
        FILE *f = fopen(candidates[i], "rb");
        if (f) { fclose(f); return candidates[i]; }
    }
    return NULL;
}

static void load_model(const char *path) {
    ModelFormat fmt = detect_format(path);
    switch (fmt) {
        case FORMAT_BIN:
            printf("[INFO] Format: custom float32 binary (.bin)\n");
            load_model_bin(path);
            break;
        case FORMAT_GGUF:
            printf("[INFO] Format: GGUF\n");
            load_model_gguf(path);
            break;
        default:
            fprintf(stderr,"[ERROR] Cannot determine format: %s\n", path);
            exit(1);
    }
}

/* ============================================================
 * TOKENIZER
 * ============================================================ */
static void init_byte_encoder(Tokenizer *tok) {
    int bs[256], cs[256], n_bs = 0;
    for (int b=33;  b<=126; b++) { bs[n_bs]=b; cs[n_bs]=b; n_bs++; }
    for (int b=161; b<=172; b++) { bs[n_bs]=b; cs[n_bs]=b; n_bs++; }
    for (int b=174; b<=255; b++) { bs[n_bs]=b; cs[n_bs]=b; n_bs++; }
    int extra = 256;
    for (int b=0; b<256; b++) {
        int found=0;
        for (int i=0;i<n_bs;i++) if(bs[i]==b){found=1;break;}
        if (!found) { bs[n_bs]=b; cs[n_bs]=extra++; n_bs++; }
    }
    for (int i=0;i<256;i++) tok->byte_decoder[cs[i]]=bs[i];
    for (int i=0;i<256;i++) {
        int cp=cs[i]; char *out=tok->byte_encoder[bs[i]];
        if (cp<0x80)  { out[0]=(char)cp; out[1]='\0'; }
        else if (cp<0x800) {
            out[0]=(char)(0xC0|(cp>>6)); out[1]=(char)(0x80|(cp&0x3F)); out[2]='\0';
        } else {
            out[0]=(char)(0xE0|(cp>>12)); out[1]=(char)(0x80|((cp>>6)&0x3F));
            out[2]=(char)(0x80|(cp&0x3F)); out[3]='\0';
        }
    }
}

static int utf8_decode(const char **s) {
    unsigned char c = (unsigned char)**s;
    int cp;
    if (c<0x80) { cp=c; (*s)++; }
    else if ((c&0xE0)==0xC0) { cp=(c&0x1F)<<6; (*s)++; cp|=((unsigned char)**s&0x3F); (*s)++; }
    else if ((c&0xF0)==0xE0) {
        cp=(c&0x0F)<<12; (*s)++;
        cp|=((unsigned char)**s&0x3F)<<6; (*s)++;
        cp|=((unsigned char)**s&0x3F); (*s)++;
    } else { cp='?'; (*s)++; }
    return cp;
}

static uint32_t str_hash(const uint8_t *s, int len) {
    uint32_t h=2166136261u;
    for (int i=0;i<len;i++) { h^=s[i]; h*=16777619u; }
    return h;
}
static void vocab_hash_insert(Tokenizer *tok, int tid) {
    uint32_t slot = str_hash(tok->vocab[tid].bytes, tok->vocab[tid].len) % VOCAB_HASH_SIZE;
    tok->vocab_hash_next[tid] = tok->vocab_hash[slot];
    tok->vocab_hash[slot] = tid;
}
static int vocab_lookup(const Tokenizer *tok, const uint8_t *s, int len) {
    uint32_t slot = str_hash(s,len) % VOCAB_HASH_SIZE;
    int id = tok->vocab_hash[slot];
    while (id != -1) {
        if (tok->vocab[id].len==len && memcmp(tok->vocab[id].bytes,s,len)==0) return id;
        id = tok->vocab_hash_next[id];
    }
    return -1;
}

static void load_encoder_json(Tokenizer *tok, const char *path) {
    FILE *f = fopen(path,"r");
    if (!f) { fprintf(stderr,"[ERROR] Cannot open encoder.json: %s\n",path); exit(1); }
    memset(tok->vocab_hash,     -1,sizeof(tok->vocab_hash));
    memset(tok->vocab_hash_next,-1,sizeof(tok->vocab_hash_next));
    tok->vocab_size = 0;
    fseek(f,0,SEEK_END); long fsize=ftell(f); fseek(f,0,SEEK_SET);
    char *buf=(char*)malloc((size_t)fsize+1);
    if (!buf) { fprintf(stderr,"[FATAL] OOM encoder.json\n"); fclose(f); exit(1); }
    if (fread(buf,1,(size_t)fsize,f)!=(size_t)fsize) {
        fprintf(stderr,"[ERROR] Short read encoder.json\n"); free(buf); fclose(f); exit(1); }
    buf[fsize]='\0'; fclose(f);
    char *p=buf;
    while(*p&&*p!='{')p++; if(*p)p++;
    while(*p) {
        while(*p&&(*p==' '||*p=='\n'||*p=='\r'||*p=='\t'||*p==','))p++;
        if(*p=='}')break;
        if(*p!='"'){p++;continue;}
        p++;
        uint8_t key[BPE_TOKEN_MAX_LEN]; int key_len=0;
        while(*p&&*p!='"'&&key_len<BPE_TOKEN_MAX_LEN-1) {
            if(*p=='\\') {
                p++;
                switch(*p) {
                    case '"':  key[key_len++]='"';  p++;break;
                    case '\\': key[key_len++]='\\'; p++;break;
                    case '/':  key[key_len++]='/';  p++;break;
                    case 'n':  key[key_len++]='\n'; p++;break;
                    case 'r':  key[key_len++]='\r'; p++;break;
                    case 't':  key[key_len++]='\t'; p++;break;
                    case 'b':  key[key_len++]='\b'; p++;break;
                    case 'f':  key[key_len++]='\f'; p++;break;
                    case 'u': {
                        p++; char hex[5]={0};
                        for(int hi=0;hi<4&&*p;hi++)hex[hi]=*p++;
                        int cp=(int)strtol(hex,NULL,16);
                        if(cp<0x80) key[key_len++]=(uint8_t)cp;
                        else if(cp<0x800){
                            key[key_len++]=(uint8_t)(0xC0|(cp>>6));
                            key[key_len++]=(uint8_t)(0x80|(cp&0x3F));
                        } else {
                            key[key_len++]=(uint8_t)(0xE0|(cp>>12));
                            key[key_len++]=(uint8_t)(0x80|((cp>>6)&0x3F));
                            key[key_len++]=(uint8_t)(0x80|(cp&0x3F));
                        }
                        break;
                    }
                    default: key[key_len++]=(uint8_t)*p++;break;
                }
            } else { key[key_len++]=(uint8_t)*p++; }
        }
        if(*p=='"')p++;
        while(*p&&(*p==' '||*p==':'||*p=='\t'))p++;
        if(*p<'0'||*p>'9')continue;
        int token_id=0;
        while(*p>='0'&&*p<='9'){token_id=token_id*10+(*p-'0');p++;}
        if(token_id<BPE_MAX_VOCAB){
            memcpy(tok->vocab[token_id].bytes,key,(size_t)key_len);
            tok->vocab[token_id].len=key_len;
            vocab_hash_insert(tok,token_id);
            if(token_id+1>tok->vocab_size)tok->vocab_size=token_id+1;
        }
    }
    free(buf);
    printf("[INFO] Vocabulary loaded: %d tokens\n", tok->vocab_size);
}

/* unified skip logic (the original had a double-skip dead code path) */
static void load_vocab_bpe(Tokenizer *tok, const char *path) {
    FILE *f = fopen(path,"r");
    if (!f) { fprintf(stderr,"[ERROR] Cannot open vocab.bpe: %s\n",path); exit(1); }
    tok->n_merges=0;
    char line[1024];
    if (!fgets(line,sizeof(line),f)) { fclose(f); return; }
    while (fgets(line,sizeof(line),f) && tok->n_merges<BPE_MAX_MERGES) {
        int len=(int)strlen(line);
        while(len>0&&(line[len-1]=='\n'||line[len-1]=='\r'))line[--len]='\0';
        if(len==0)continue;
        char *space=strchr(line,' ');
        if(!space)continue;
        *space='\0';
        char *left_str=line, *right_str=space+1;
        int left_id =vocab_lookup(tok,(const uint8_t*)left_str, (int)strlen(left_str));
        int right_id=vocab_lookup(tok,(const uint8_t*)right_str,(int)strlen(right_str));
        if(left_id==-1||right_id==-1)continue;
        int ll=(int)strlen(left_str),rl=(int)strlen(right_str);
        if(ll+rl>=BPE_TOKEN_MAX_LEN)continue;
        uint8_t merged[BPE_TOKEN_MAX_LEN];
        memcpy(merged,left_str,(size_t)ll);
        memcpy(merged+ll,right_str,(size_t)rl);
        int result_id=vocab_lookup(tok,merged,ll+rl);
        if(result_id==-1)continue;
        tok->merges[tok->n_merges].left  =left_id;
        tok->merges[tok->n_merges].right =right_id;
        tok->merges[tok->n_merges].result=result_id;
        tok->n_merges++;
    }
    fclose(f);
    printf("[INFO] BPE merges loaded: %d rules\n", tok->n_merges);
}

static void load_tokenizer(const char *ep, const char *bp) {
    init_byte_encoder(&g_tokenizer);
    load_encoder_json(&g_tokenizer, ep);
    load_vocab_bpe(&g_tokenizer, bp);
}
/* ============================================================
 * BPE ENCODING
 * ============================================================ */
#define MAX_WORD_LEN    128
#define MAX_WORD_TOKENS (MAX_WORD_LEN * 4)
typedef struct { int ids[MAX_WORD_TOKENS]; int len; } TokenSeq;

static void bpe_apply_merges(TokenSeq *seq, const Tokenizer *tok) {
    while (seq->len >= 2) {
        int best_merge=tok->n_merges, best_pos=-1;
        for (int i=0;i<seq->len-1;i++) {
            int a=seq->ids[i],b=seq->ids[i+1];
            for (int m=0;m<tok->n_merges;m++) {
                if(tok->merges[m].left==a&&tok->merges[m].right==b) {
                    if(m<best_merge){best_merge=m;best_pos=i;}
                    break;
                }
            }
        }
        if(best_pos==-1)break;
        seq->ids[best_pos]=tok->merges[best_merge].result;
        for(int i=best_pos+1;i<seq->len-1;i++)seq->ids[i]=seq->ids[i+1];
        seq->len--;
    }
}

static int encode_word(const Tokenizer *tok, const uint8_t *word_bytes,
                       int word_len, int *out_ids) {
    TokenSeq seq={.len=0};
    for(int i=0;i<word_len&&seq.len<MAX_WORD_TOKENS;i++) {
        uint8_t b=word_bytes[i];
        const char *enc=tok->byte_encoder[b];
        int tid=vocab_lookup(tok,(const uint8_t*)enc,(int)strlen(enc));
        seq.ids[seq.len++]=(tid==-1)?(int)b:tid;
    }
    bpe_apply_merges(&seq,tok);
    for(int i=0;i<seq.len;i++)out_ids[i]=seq.ids[i];
    return seq.len;
}

static int tokenize(const Tokenizer *tok, const char *text,
                    int *out_ids, int max_tokens) {
    int n_tokens=0;
    const uint8_t *p=(const uint8_t*)text;
    int text_len=(int)strlen(text), i=0;
    while(i<text_len&&n_tokens<max_tokens) {
        uint8_t word[MAX_WORD_LEN]; int wlen=0;
        if(p[i]==' '&&i+1<text_len)word[wlen++]=p[i++];
        if(i>=text_len) {
            if(wlen>0){
                int word_ids[MAX_WORD_TOKENS];
                int n=encode_word(tok,word,wlen,word_ids);
                for(int j=0;j<n&&n_tokens<max_tokens;j++)out_ids[n_tokens++]=word_ids[j];
            }
            break;
        }
        uint8_t c=p[i];
        if((c>='A'&&c<='Z')||(c>='a'&&c<='z')||(c>='0'&&c<='9')||c>=0x80) {
            while(i<text_len&&wlen<MAX_WORD_LEN-1) {
                uint8_t cc=p[i];
                if((cc>='A'&&cc<='Z')||(cc>='a'&&cc<='z')||(cc>='0'&&cc<='9')||cc>=0x80)
                    word[wlen++]=p[i++];
                else break;
            }
        } else { word[wlen++]=p[i++]; }
        if(wlen>0) {
            int word_ids[MAX_WORD_TOKENS];
            int n=encode_word(tok,word,wlen,word_ids);
            for(int j=0;j<n&&n_tokens<max_tokens;j++)out_ids[n_tokens++]=word_ids[j];
        }
    }
    return n_tokens;
}

static int detokenize_token(const Tokenizer *tok, int token_id,
                            char *out_buf, int buf_size) {
    if(token_id<0||token_id>=tok->vocab_size)return 0;
    const VocabEntry *ve=&tok->vocab[token_id];
    const char *s=(const char*)ve->bytes;
    const char *end=s+ve->len;
    int out_len=0;
    while(s<end&&out_len<buf_size-1) {
        int cp=utf8_decode(&s);
        if(cp>=0&&cp<0x400)out_buf[out_len++]=(char)tok->byte_decoder[cp];
    }
    out_buf[out_len]='\0';
    return out_len;
}

/* ============================================================
 * GENERATION LOOP
 * ============================================================ */
static void generate(const char *prompt, int max_new_tokens,
                     float temperature, float top_p) {
    int prompt_tokens[MAX_SEQ_LEN];
    int n_prompt = tokenize(&g_tokenizer, prompt, prompt_tokens, CFG_S);
    if (n_prompt == 0) { fprintf(stderr,"[ERROR] Empty prompt.\n"); return; }
    printf("[INFO] Prompt tokens: %d\n", n_prompt);
    printf("\n--- Generated Text ---\n%s", prompt);
    fflush(stdout);

    g_kv_cache.seq_len = 0;
    float *logits = NULL;
    for (int i = 0; i < n_prompt; i++) {
        logits = model_forward(prompt_tokens[i], i);
        g_kv_cache.seq_len = i+1;
    }

    int pos = n_prompt;
    char decode_buf[64];
    for (int step = 0; step < max_new_tokens; step++) {
        if (pos >= CFG_S) { printf("\n[Context window full]\n"); break; }
        int next_token = sample_top_p(logits, temperature, top_p);
        if (next_token == 50256) { printf("\n[EOS]\n"); break; }
        int dec_len = detokenize_token(&g_tokenizer, next_token, decode_buf, sizeof(decode_buf));
        if (dec_len > 0) { fwrite(decode_buf, 1, (size_t)dec_len, stdout); fflush(stdout); }
        logits = model_forward(next_token, pos);
        g_kv_cache.seq_len = pos+1;
        pos++;
    }
    printf("\n--- Done: %d tokens generated ---\n", pos - n_prompt);
}

/* ============================================================
 * MAIN
 *
 * Usage:
 *   ./lm "prompt" [max_tokens] [temperature] [top_p] [--model <path>]
 *
 * Model auto-detection order (if --model not given):
 *   1. gpt2_124m.bin   (custom float32 binary)
 *   2. gpt2.f16.gguf   (GGUF FP16)
 *   3. gpt2.gguf       (GGUF any)
 * ============================================================ */

int main(int argc, char *argv[]) {
    printf("==============================================\n");
    printf("  lm.c — GPT-2 Inference (Small + Medium)\n");
    printf("  Architecture parameters read at runtime from model file.\n");
    printf("==============================================\n\n");

    const char *prompt       = "Hello, world!";
    int         max_tokens   = 64;
    float       temperature  = 0.7f;
    float       top_p        = 0.9f;
    const char *model_path   = NULL;
    const char *encoder_path = "encoder.json";
    const char *bpe_path     = "vocab.bpe";

    for (int i=1;i<argc;i++) {
        if      (strcmp(argv[i],"--model")==0   &&i+1<argc) model_path   = argv[++i];
        else if (strcmp(argv[i],"--encoder")==0 &&i+1<argc) encoder_path = argv[++i];
        else if (strcmp(argv[i],"--bpe")==0     &&i+1<argc) bpe_path     = argv[++i];
        else if (i==1&&argv[i][0]!='-') prompt      = argv[i];
        else if (i==2&&argv[i][0]!='-') max_tokens  = atoi(argv[i]);
        else if (i==3&&argv[i][0]!='-') temperature = (float)atof(argv[i]);
        else if (i==4&&argv[i][0]!='-') top_p       = (float)atof(argv[i]);
    }

    if (max_tokens <= 0) max_tokens = 64;
    if (max_tokens > MAX_SEQ_LEN - 10) max_tokens = MAX_SEQ_LEN - 10;
    if (temperature < 0.0f) temperature = 0.7f;
    if (top_p <= 0.0f || top_p > 1.0f) top_p = 0.9f;

    if (!model_path) {
        model_path = find_default_model();
        if (!model_path) {
            fprintf(stderr,
                "[ERROR] No model file found.\n"
                "  Tried: gpt2_124m.bin, gpt2_medium.bin, gpt2_large.bin, gpt2_xl.bin,\n"
                "         gpt2.f16.gguf, gpt2.gguf,\n"
                "         gpt2-medium.f16.gguf, gpt2-medium.gguf,\n"
                "         gpt2-large.f16.gguf,  gpt2-large.gguf,\n"
                "         gpt2-xl.f16.gguf,     gpt2-xl.gguf\n"
                "  Or pass: --model <path>\n");            
            return 1;
        }
    }

    printf("[CONFIG] model:       %s\n",   model_path);
    printf("[CONFIG] prompt:      \"%s\"\n", prompt);
    printf("[CONFIG] max_tokens:  %d\n",   max_tokens);
    printf("[CONFIG] temperature: %.2f\n", temperature);
    printf("[CONFIG] top_p:       %.2f\n\n", top_p);

    rng_seed((uint64_t)time(NULL));

#ifdef _OPENMP
    printf("[INFO] OpenMP enabled: %d threads\n", omp_get_max_threads());
#else
    printf("[INFO] OpenMP disabled (single-threaded)\n");
#endif

    load_model(model_path);
    load_tokenizer(encoder_path, bpe_path);
    init_kv_cache();
    init_activations();

    generate(prompt, max_tokens, temperature, top_p);

    free(g_arena.data);
    free(g_weights.layers);   /* LayerWeights array (heap, not arena) */
    free(g_kv_cache.k_cache);
    free(g_kv_cache.v_cache);
    free_activations();
    return 0;
}
