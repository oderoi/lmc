/*
 * lm.c — Unified GPT-2 124M inference engine in pure C99
 *
 * Supports TWO model formats, auto-detected by file extension / magic:
 *   1. gpt2_124m.bin      — custom binary format (float32, little-endian)
 *   2. *.gguf             — GGUF format (llama.cpp compatible)
 *        Supported quant types: F32, F16, Q8_0, Q5_K (S+M), Q6_K
 *        Examples: gpt2.f16.gguf, gpt2.Q8_0.gguf
 *
 * Architecture: GPT-2 124M
 *   - 12 transformer layers, 12 attention heads
 *   - 768 embedding dimension, 3072 FFN hidden dimension
 *   - 50257 vocabulary size, 1024 max sequence length
 *
 * Compile (single-threaded):
 *   gcc -O3 -march=native -ffast-math lm.c -o lm -lm
 *
 * Compile (OpenMP multi-threaded):
 *   gcc -O3 -march=native -ffast-math -fopenmp lm.c -o lm -lm
 *
 * Usage:
 *   ./lm "Your prompt" [max_tokens] [temperature] [top_p]
 *
 *   The model file is auto-detected:
 *     - Checks for gpt2_124m.bin  first  (float32 custom format)
 *     - Falls back to gpt2.f16.gguf      (GGUF FP16 format)
 *   Or pass an explicit path as the first argument ending in .bin / .gguf.
 *
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
  * BINARY MODEL FILE MAGIC & VERSION (custom .bin format)
  * ============================================================ */
 #define MODEL_MAGIC   0x47505432U  /* "GPT2" */
 #define MODEL_VERSION 1
 
 /* ============================================================
  * GGUF FORMAT CONSTANTS
  * ============================================================ */
 #define GGUF_MAGIC        0x46554747U  /* "GGUF" little-endian */
 #define GGUF_VERSION_MIN  1
 #define GGUF_VERSION_MAX  3
 
 /* GGUF tensor type IDs we care about */
 #define GGUF_TYPE_F32   0
 #define GGUF_TYPE_F16   1
 #define GGUF_TYPE_Q8_0  8   /* 8-bit quantization, block size 32            */
 #define GGUF_TYPE_Q6_K  14  /* 6-bit K-quant, super-block size 256          */
 
 /* Q8_0 block layout (34 bytes per block of 32 elements):
  *   [uint16_t d (float16 scale)][int8_t qs[32]]
  * Dequantize: x[i] = qs[i] * f16_to_f32(d)
  */
 #define Q8_0_BLOCK_SIZE      32
 #define Q8_0_BYTES_PER_BLOCK 34   /* 2 (f16 scale) + 32 (int8 values) */
 
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
 #define GGUF_TYPE_Q5_K  13  /* 5-bit K-quant, super-block size 256, S and M variants */
 #define Q5_K_BLOCK_SIZE      256
 #define Q5_K_QH_BYTES         32  /* high bits: 1 per element, 8/byte    */
 #define Q5_K_QS_BYTES        128  /* low nibbles: 4 bits/element, 2/byte */
 #define Q5_K_SC_BYTES         12  /* packed 6-bit scales+mins            */
 #define Q5_K_BYTES_PER_BLOCK 176  /* 2+2+12+32+128                       */
 
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
 
 #define Q6_K_BLOCK_SIZE      256
 #define Q6_K_QL_BYTES        128  /* lower 4 bits, 2/byte                   */
 #define Q6_K_QH_BYTES         64  /* upper 2 bits, 4/byte                   */
 #define Q6_K_SC_BYTES         16  /* int8 sub-block scales                  */
 #define Q6_K_D_BYTES           2  /* float16 super-block scale              */
 #define Q6_K_BYTES_PER_BLOCK 210  /* 128+64+16+2                            */
 
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
 #define GGUF_MTYPE_UINT64  10
 #define GGUF_MTYPE_INT64   11
 #define GGUF_MTYPE_FLOAT64 12
 
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
     float  *data;
     size_t  capacity;
     size_t  used;
 } Arena;
 
 static Arena g_arena;
 
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
  * ============================================================ */
 typedef struct {
     float *ln1_weight;        /* [EMBED_DIM]           */
     float *ln1_bias;          /* [EMBED_DIM]           */
     float *qkv_weight;        /* [3*EMBED_DIM*EMBED_DIM] */
     float *qkv_bias;          /* [3*EMBED_DIM]         */
     float *attn_proj_weight;  /* [EMBED_DIM*EMBED_DIM] */
     float *attn_proj_bias;    /* [EMBED_DIM]           */
     float *ln2_weight;        /* [EMBED_DIM]           */
     float *ln2_bias;          /* [EMBED_DIM]           */
     float *ffn_fc_weight;     /* [FFN_DIM*EMBED_DIM]   */
     float *ffn_fc_bias;       /* [FFN_DIM]             */
     float *ffn_proj_weight;   /* [EMBED_DIM*FFN_DIM]   */
     float *ffn_proj_bias;     /* [EMBED_DIM]           */
 } LayerWeights;
 
 typedef struct {
     float       *wte;                    /* [VOCAB_SIZE, EMBED_DIM] token embeddings  */
     float       *wpe;                    /* [SEQ_LEN, EMBED_DIM]    position embeddings */
     LayerWeights layers[GPT2_N_LAYERS];
     float       *ln_f_weight;            /* [EMBED_DIM]             */
     float       *ln_f_bias;             /* [EMBED_DIM]             */
     float       *lm_head;               /* [VOCAB_SIZE, EMBED_DIM] LM head weights.
                                           * For .bin (tied weights): same pointer as wte.
                                           * For GGUF: points to output.weight if present,
                                           * otherwise falls back to wte (tied). */
 } ModelWeights;
 
 /* ============================================================
  * KV CACHE
  * ============================================================ */
 typedef struct {
     float *k_cache;  /* [N_LAYERS * SEQ_LEN * N_HEADS * HEAD_DIM] */
     float *v_cache;
     int    seq_len;
 } KVCache;
 
 /* ============================================================
  * ACTIVATION BUFFERS
  * ============================================================ */
 typedef struct {
     float *x;            /* [EMBED_DIM]              */
     float *x_norm;       /* [EMBED_DIM]              */
     float *qkv;          /* [3*EMBED_DIM]            */
     float *attn_out;     /* [EMBED_DIM]              */
     float *proj_out;     /* [EMBED_DIM] (FIX #3)     */
     float *ffn_hidden;   /* [FFN_DIM]                */
     float *ffn_out;      /* [EMBED_DIM] (FIX #3)     */
     float *logits;       /* [VOCAB_SIZE]             */
     float *attn_scores;  /* [N_HEADS * SEQ_LEN]      */
     void  *sorted_buf;   /* for top-p (FIX #5)       */
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
 
 typedef struct {
     VocabEntry vocab[BPE_MAX_VOCAB];
     int        vocab_size;
     BPEMerge   merges[BPE_MAX_MERGES];
     int        n_merges;
     char       byte_encoder[256][8];
     int        byte_decoder[0x400];
 #define VOCAB_HASH_SIZE 131072
     int vocab_hash[VOCAB_HASH_SIZE];
     int vocab_hash_next[BPE_MAX_VOCAB];
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
  * Decode one (scale, min) pair from the 12-byte packed scales field.
  * Encodes 8 scale values and 8 min values in 6 bits each = 96 bits = 12 bytes.
  *
  * For j < 4:
  *   scale = scales[j]   & 0x3F
  *   min   = scales[j+4] & 0x3F
  * For j >= 4:
  *   scale = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
  *   min   = (scales[j+4] >>  4)  | ((scales[j  ] >> 6) << 4)
  */
 static void get_scale_min_k4(int j, const uint8_t *scales,
                               uint8_t *out_sc, uint8_t *out_m) {
     if (j < 4) {
         *out_sc = scales[j]   & 0x3F;
         *out_m  = scales[j+4] & 0x3F;
     } else {
         *out_sc = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4);
         *out_m  = (scales[j+4] >>  4)  | ((scales[j  ] >> 6) << 4);
     }
 }
 
 /*
  * Dequantize a Q5_K super-block tensor into float32.
  * Handles both Q5_K_S and Q5_K_M — they share the same binary format (type 13).
  *
  * Block memory layout (176 bytes):
  *   offset  0: d     [2]  float16 quant scale
  *   offset  2: dmin  [2]  float16 min scale
  *   offset  4: scales[12] 6-bit packed scales+mins (8 of each)
  *   offset 16: qh    [32] high bit of each 5-bit quant, 8 packed per byte
  *   offset 48: qs    [128] low 4 bits of each 5-bit quant
  *
  * qs packing (critical — NOT a simple sequential layout):
  *   qs[k] lo nibble = lo4 of element k        (k = 0..127)
  *   qs[k] hi nibble = lo4 of element k+128    (k = 0..127)
  *
  * qh packing:
  *   qh[l] bit b = high bit of the element accessed in the outer loop
  *   at shift = j>>5 (0,2,4,6 for j=0,64,128,192)
  *
  * Loop structure (4 outer iterations, j = 0, 64, 128, 192):
  *   ql_base  = j % 128        — wraps: 0,64,0,64
  *   nibble   = (j >= 128)?4:0 — 0 for j<128 (lo nibble), 4 for j>=128 (hi nibble)
  *   shift    = j >> 5         — 0,2,4,6
  *   Two inner loops of 32 elements each, using two consecutive scale groups.
  *
  * Verified with full round-trip tests against quantizer output.
  */
 static void dequant_q5k(const uint8_t *src, float *dst, size_t n_elements) {
     size_t n_blocks = n_elements / Q5_K_BLOCK_SIZE;
 
     for (size_t b = 0; b < n_blocks; b++) {
         uint16_t d_raw, dmin_raw;
         memcpy(&d_raw,    src + 0, sizeof(uint16_t));
         memcpy(&dmin_raw, src + 2, sizeof(uint16_t));
         float d    = f16_to_f32(d_raw);
         float dmin = f16_to_f32(dmin_raw);
 
         const uint8_t *scales = src + 4;   /* [12] packed 6-bit scales+mins */
         const uint8_t *qh     = src + 16;  /* [32] high bits                */
         const uint8_t *qs     = src + 48;  /* [128] low nibbles             */
         float         *y      = dst + b * Q5_K_BLOCK_SIZE;
 
         int is_ = 0;       /* scale group index, 0..7 */
         int y_off = 0;     /* output element index    */
 
         for (int j = 0; j < 256; j += 64) {
             uint8_t sc, m;
             get_scale_min_k4(is_++, scales, &sc, &m);
             float d1 = d * (float)sc,  m1 = dmin * (float)m;
             get_scale_min_k4(is_++, scales, &sc, &m);
             float d2 = d * (float)sc,  m2 = dmin * (float)m;
 
             int ql_base = j & 127;         /* j%128: 0,64,0,64 for j=0,64,128,192  */
             int nibble  = (j >= 128) ? 4 : 0; /* lo nibble for j<128, hi for j>=128 */
             int shift   = j >> 5;          /* 0,2,4,6 for j=0,64,128,192           */
 
             /* First group of 32: elements y_off .. y_off+31 */
             for (int l = 0; l < 32; l++) {
                 int hi = (qh[l] >> shift) & 1;
                 int lo = (qs[ql_base + l] >> nibble) & 0x0F;
                 y[y_off + l] = d1 * (float)(lo | (hi << 4)) - m1;
             }
             y_off += 32;
 
             /* Second group of 32: elements y_off .. y_off+31 */
             for (int l = 0; l < 32; l++) {
                 int hi = (qh[l] >> (shift + 1)) & 1;
                 int lo = (qs[ql_base + l + 32] >> nibble) & 0x0F;
                 y[y_off + l] = d2 * (float)(lo | (hi << 4)) - m2;
             }
             y_off += 32;
         }
 
         src += Q5_K_BYTES_PER_BLOCK;
     }
 }
 
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
     float *k_cache,
     float *v_cache,
     int pos,
     float *qkv_buf,
     float *scores_buf)
 {
     const int D  = GPT2_EMBED_DIM;
     const int H  = GPT2_N_HEADS;
     const int Dh = GPT2_HEAD_DIM;
     const float scale = 1.0f / sqrtf((float)Dh);
 
     matmul_vec(qkv_buf, lw->qkv_weight, lw->qkv_bias, x_norm, 3 * D, D);
 
     float *q_vec = qkv_buf;
     float *k_vec = qkv_buf + D;
     float *v_vec = qkv_buf + 2 * D;
 
     float *k_dest = k_cache + (size_t)pos * H * Dh;
     float *v_dest = v_cache + (size_t)pos * H * Dh;
     memcpy(k_dest, k_vec, (size_t)H * Dh * sizeof(float));
     memcpy(v_dest, v_vec, (size_t)H * Dh * sizeof(float));
 
     int ctx_len = pos + 1;
 
 #ifdef _OPENMP
     #pragma omp parallel for schedule(static)
 #endif
     for (int h = 0; h < H; h++) {
         float *q_h    = q_vec + h * Dh;
         float *scores = scores_buf + h * GPT2_SEQ_LEN;
 
         for (int t = 0; t < ctx_len; t++) {
             float *k_t = k_cache + (size_t)t * H * Dh + h * Dh;
             float dot = 0.0f;
             int d = 0;
             for (; d <= Dh - 8; d += 8) {
                 dot += q_h[d+0]*k_t[d+0] + q_h[d+1]*k_t[d+1]
                      + q_h[d+2]*k_t[d+2] + q_h[d+3]*k_t[d+3]
                      + q_h[d+4]*k_t[d+4] + q_h[d+5]*k_t[d+5]
                      + q_h[d+6]*k_t[d+6] + q_h[d+7]*k_t[d+7];
             }
             for (; d < Dh; d++) dot += q_h[d] * k_t[d];
             scores[t] = dot * scale;
         }
 
         softmax(scores, ctx_len);
 
         float *out_h = out + h * Dh;
         memset(out_h, 0, Dh * sizeof(float));
         for (int t = 0; t < ctx_len; t++) {
             float *v_t = v_cache + (size_t)t * H * Dh + h * Dh;
             float s = scores[t];
             for (int d = 0; d < Dh; d++) out_h[d] += s * v_t[d];
         }
     }
 }
 
 /* ============================================================
  * TRANSFORMER BLOCK — FIX #3: no more stack VLAs
  * ============================================================ */
 static void transformer_block_forward(
     float *x,
     const LayerWeights *lw,
     float *k_cache,
     float *v_cache,
     int pos,
     float *scratch_norm,
     float *scratch_qkv,
     float *scratch_attn,
     float *scratch_scores,
     float *scratch_ffn,
     float *scratch_proj,   /* [EMBED_DIM] — replaces stack VLA */
     float *scratch_ffnout  /* [EMBED_DIM] — replaces stack VLA */
 ) {
     const int D = GPT2_EMBED_DIM;
     const int F = GPT2_FFN_DIM;
 
     /* Sub-layer 1: LN + MHA + residual */
     layer_norm(scratch_norm, x, lw->ln1_weight, lw->ln1_bias, D);
     attention_forward(scratch_attn, scratch_norm, lw,
                       k_cache, v_cache, pos,
                       scratch_qkv, scratch_scores);
     matmul_vec(scratch_proj, lw->attn_proj_weight, lw->attn_proj_bias,
                scratch_attn, D, D);
     for (int i = 0; i < D; i++) x[i] += scratch_proj[i];
 
     /* Sub-layer 2: LN + FFN + residual */
     layer_norm(scratch_norm, x, lw->ln2_weight, lw->ln2_bias, D);
     matmul_vec(scratch_ffn, lw->ffn_fc_weight, lw->ffn_fc_bias,
                scratch_norm, F, D);
     for (int i = 0; i < F; i++) scratch_ffn[i] = gelu(scratch_ffn[i]);
     matmul_vec(scratch_ffnout, lw->ffn_proj_weight, lw->ffn_proj_bias,
                scratch_ffn, D, F);
     for (int i = 0; i < D; i++) x[i] += scratch_ffnout[i];
 }
 
 /* ============================================================
  * MODEL FORWARD — FIX #4: no stack VLA for x_final
  * ============================================================ */
 static float* model_forward(int token_id, int pos) {
     const int D = GPT2_EMBED_DIM;
     const int V = GPT2_VOCAB_SIZE;
 
     float *x = g_act.x;
     float *tok_emb = g_weights.wte + (size_t)token_id * D;
     float *pos_emb = g_weights.wpe + (size_t)pos * D;
     for (int i = 0; i < D; i++) x[i] = tok_emb[i] + pos_emb[i];
 
     for (int l = 0; l < GPT2_N_LAYERS; l++) {
         size_t layer_offset = (size_t)l * GPT2_SEQ_LEN * GPT2_N_HEADS * GPT2_HEAD_DIM;
         float *k_cache_l = g_kv_cache.k_cache + layer_offset;
         float *v_cache_l = g_kv_cache.v_cache + layer_offset;
 
         transformer_block_forward(
             x, &g_weights.layers[l],
             k_cache_l, v_cache_l, pos,
             g_act.x_norm,
             g_act.qkv,
             g_act.attn_out,
             g_act.attn_scores,
             g_act.ffn_hidden,
             g_act.proj_out,     /* FIX #3 */
             g_act.ffn_out       /* FIX #3 */
         );
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
     if (g_rng_state == 0) g_rng_state = 1; /* avoid degenerate 0 state */
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
     const ProbIdx *pa = (const ProbIdx *)a;
     const ProbIdx *pb = (const ProbIdx *)b;
     return (pb->prob > pa->prob) ? 1 : (pb->prob < pa->prob) ? -1 : 0;
 }
 
 /* FIX #5: sorted buffer is pre-allocated, not static */
 static int sample_top_p(float *logits, float temperature, float top_p) {
     const int V = GPT2_VOCAB_SIZE;
     ProbIdx *sorted = (ProbIdx*)g_act.sorted_buf;
 
     if (temperature < 1e-6f) {
         int best = 0;
         for (int i = 1; i < V; i++)
             if (logits[i] > logits[best]) best = i;
         return best;
     }
 
     float inv_temp = 1.0f / temperature;
     for (int i = 0; i < V; i++) logits[i] *= inv_temp;
     softmax(logits, V);
 
     for (int i = 0; i < V; i++) { sorted[i].prob = logits[i]; sorted[i].idx = i; }
     qsort(sorted, V, sizeof(ProbIdx), cmp_prob_desc);
 
     float cumsum = 0.0f;
     int nucleus_size = 0;
     for (int i = 0; i < V; i++) {
         cumsum += sorted[i].prob;
         nucleus_size = i + 1;
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
     return sorted[nucleus_size - 1].idx;
 }
 
 /* ============================================================
  * ACTIVATION / KV CACHE INIT
  * ============================================================ */
 static void init_activations(void) {
     const int D = GPT2_EMBED_DIM;
     const int V = GPT2_VOCAB_SIZE;
     const int F = GPT2_FFN_DIM;
     const int H = GPT2_N_HEADS;
     const int S = GPT2_SEQ_LEN;
 
 #define ACT_ALLOC(field, n) \
     do { \
         g_act.field = (float*)malloc((n) * sizeof(float)); \
         if (!g_act.field) { \
             fprintf(stderr, "[FATAL] Cannot allocate activation: " #field "\n"); \
             exit(1); \
         } \
     } while(0)
 
     ACT_ALLOC(x,           D);
     ACT_ALLOC(x_norm,      D);
     ACT_ALLOC(qkv,      3 * D);
     ACT_ALLOC(attn_out,    D);
     ACT_ALLOC(proj_out,    D);
     ACT_ALLOC(ffn_hidden,  F);
     ACT_ALLOC(ffn_out,     D);
     ACT_ALLOC(logits,      V);
     ACT_ALLOC(attn_scores, H * S);
 
     /* FIX #5: sorted_buf for top-p sampling */
     g_act.sorted_buf = malloc((size_t)V * sizeof(ProbIdx));
     if (!g_act.sorted_buf) {
         fprintf(stderr, "[FATAL] Cannot allocate sorted_buf\n");
         exit(1);
     }
 #undef ACT_ALLOC
 }
 
 static void free_activations(void) {
     free(g_act.x);
     free(g_act.x_norm);
     free(g_act.qkv);
     free(g_act.attn_out);
     free(g_act.proj_out);
     free(g_act.ffn_hidden);
     free(g_act.ffn_out);
     free(g_act.logits);
     free(g_act.attn_scores);
     free(g_act.sorted_buf);
 }
 
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
  * ARENA ALLOCATION HELPER
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
     const int D = GPT2_EMBED_DIM, V = GPT2_VOCAB_SIZE;
     const int S = GPT2_SEQ_LEN,   L = GPT2_N_LAYERS;
     const int F = GPT2_FFN_DIM;
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
 
 /* Assign weight pointers from arena (same order as binary file) */
 static void assign_weight_pointers(void) {
     const int D = GPT2_EMBED_DIM, V = GPT2_VOCAB_SIZE;
     const int S = GPT2_SEQ_LEN,   L = GPT2_N_LAYERS;
     const int F = GPT2_FFN_DIM;
 
     g_weights.wte = arena_alloc((size_t)V * D);
     g_weights.wpe = arena_alloc((size_t)S * D);
 
     for (int l = 0; l < L; l++) {
         LayerWeights *lw = &g_weights.layers[l];
         lw->ln1_weight      = arena_alloc(D);
         lw->ln1_bias        = arena_alloc(D);
         lw->qkv_weight      = arena_alloc((size_t)3*D*D);
         lw->qkv_bias        = arena_alloc(3*D);
         lw->attn_proj_weight = arena_alloc((size_t)D*D);
         lw->attn_proj_bias  = arena_alloc(D);
         lw->ln2_weight      = arena_alloc(D);
         lw->ln2_bias        = arena_alloc(D);
         lw->ffn_fc_weight   = arena_alloc((size_t)F*D);
         lw->ffn_fc_bias     = arena_alloc(F);
         lw->ffn_proj_weight = arena_alloc((size_t)D*F);
         lw->ffn_proj_bias   = arena_alloc(D);
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
     do { if (fread(&(var), sizeof(var), 1, f) != 1) { \
         fprintf(stderr, "[ERROR] Truncated header in %s\n", path); \
         fclose(f); exit(1); } } while(0)
 
     FREAD1(magic); FREAD1(version); FREAD1(vocab_size); FREAD1(seq_len);
     FREAD1(n_layers); FREAD1(n_heads); FREAD1(embed_dim);
 #undef FREAD1
 
     if (magic != MODEL_MAGIC) {
         fprintf(stderr, "[ERROR] Bad magic 0x%08X (expected 0x%08X)\n",
                 magic, MODEL_MAGIC);
         fclose(f); exit(1);
     }
     if (version != MODEL_VERSION) {
         fprintf(stderr, "[ERROR] Version mismatch: got %u, expected %u\n",
                 version, MODEL_VERSION);
         fclose(f); exit(1);
     }
     if (vocab_size != GPT2_VOCAB_SIZE || seq_len != GPT2_SEQ_LEN ||
         n_layers   != GPT2_N_LAYERS   || n_heads != GPT2_N_HEADS ||
         embed_dim  != GPT2_EMBED_DIM) {
         fprintf(stderr, "[ERROR] Architecture mismatch:\n");
         fprintf(stderr, "  Expected: V=%d S=%d L=%d H=%d D=%d\n",
                 GPT2_VOCAB_SIZE, GPT2_SEQ_LEN, GPT2_N_LAYERS,
                 GPT2_N_HEADS,    GPT2_EMBED_DIM);
         fprintf(stderr, "  Got:      V=%u S=%u L=%u H=%u D=%u\n",
                 vocab_size, seq_len, n_layers, n_heads, embed_dim);
         fclose(f); exit(1);
     }
 
     size_t total = gpt2_total_params();
     printf("[INFO] Parameters: %zu  (%.1f MB)\n",
            total, total * 4.0 / (1024*1024));
 
     arena_init(total);
     assign_weight_pointers();
 
     /* Read all floats directly into the arena */
     if (fread(g_arena.data, sizeof(float), total, f) != total) {
         fprintf(stderr, "[ERROR] Truncated weight data in %s\n", path);
         fclose(f); exit(1);
     }
 
     /* GPT-2 uses tied weights: LM head == token embedding table */
     g_weights.lm_head = g_weights.wte;
 
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
 
 static uint8_t  gguf_u8(FILE *f)  { uint8_t  v; gguf_read_bytes(f, &v, 1); return v; }
 static uint16_t gguf_u16(FILE *f) { uint16_t v; gguf_read_bytes(f, &v, 2); return v; }
 static uint32_t gguf_u32(FILE *f) { uint32_t v; gguf_read_bytes(f, &v, 4); return v; }
 static uint64_t gguf_u64(FILE *f) { uint64_t v; gguf_read_bytes(f, &v, 8); return v; }
 static int32_t  gguf_i32(FILE *f) { int32_t  v; gguf_read_bytes(f, &v, 4); return v; }
 static int64_t  gguf_i64(FILE *f) { int64_t  v; gguf_read_bytes(f, &v, 8); return v; }
 static float    gguf_f32(FILE *f) { float    v; gguf_read_bytes(f, &v, 4); return v; }
 static double   gguf_f64(FILE *f) { double   v; gguf_read_bytes(f, &v, 8); return v; }
 
 /* Read a GGUF string (uint64 length + bytes, NOT null-terminated in file) */
 static char* gguf_read_string(FILE *f, uint64_t *out_len) {
     uint64_t len = gguf_u64(f);
     char *s = (char*)malloc(len + 1);
     if (!s) { fprintf(stderr, "[FATAL] OOM in gguf_read_string\n"); exit(1); }
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
     for (uint64_t i = 0; i < count; i++) {
         gguf_skip_value(f, elem_type);
     }
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
         case GGUF_MTYPE_STRING: {
             char *s = gguf_read_string(f, NULL);
             free(s);
             break;
         }
         case GGUF_MTYPE_ARRAY:   gguf_skip_array(f); break;
         case GGUF_MTYPE_UINT64:  gguf_u64(f); break;
         case GGUF_MTYPE_INT64:   gguf_i64(f); break;
         case GGUF_MTYPE_FLOAT64: gguf_f64(f); break;
         default:
             fprintf(stderr, "[ERROR] Unknown GGUF metadata type %u\n", vtype);
             exit(1);
     }
 }
 
 /* Descriptor for one tensor in the GGUF file */
 #define GGUF_MAX_TENSORS 512
 #define GGUF_MAX_DIMS      4
 typedef struct {
     char     name[256];
     uint32_t type;          /* GGUF_TYPE_F32 or GGUF_TYPE_F16 */
     uint32_t n_dims;
     uint64_t dims[GGUF_MAX_DIMS];
     uint64_t offset;        /* byte offset from data section start */
     size_t   n_elements;
 } GGUFTensor;
 
 /* Map a GGUF tensor name to the corresponding float* in our weight struct.
  * Returns NULL if the tensor is not needed. */
 static float** gguf_name_to_ptr(const char *name) {
     /* global tensors */
     if (strcmp(name, "token_embd.weight") == 0) return &g_weights.wte;
     if (strcmp(name, "position_embd.weight") == 0) return &g_weights.wpe;
     if (strcmp(name, "output_norm.weight") == 0) return &g_weights.ln_f_weight;
     if (strcmp(name, "output_norm.bias")   == 0) return &g_weights.ln_f_bias;
     if (strcmp(name, "output.weight")      == 0) return &g_weights.lm_head;
 
     /* per-layer tensors: blk.N.xxx */
     if (strncmp(name, "blk.", 4) == 0) {
         int layer = atoi(name + 4);
         if (layer < 0 || layer >= GPT2_N_LAYERS) return NULL;
         const char *rest = strchr(name + 4, '.');
         if (!rest) return NULL;
         rest++;  /* skip the '.' */
         LayerWeights *lw = &g_weights.layers[layer];
         if (strcmp(rest, "attn_norm.weight")   == 0) return &lw->ln1_weight;
         if (strcmp(rest, "attn_norm.bias")     == 0) return &lw->ln1_bias;
         if (strcmp(rest, "attn_qkv.weight")    == 0) return &lw->qkv_weight;
         if (strcmp(rest, "attn_qkv.bias")      == 0) return &lw->qkv_bias;
         if (strcmp(rest, "attn_output.weight") == 0) return &lw->attn_proj_weight;
         if (strcmp(rest, "attn_output.bias")   == 0) return &lw->attn_proj_bias;
         if (strcmp(rest, "ffn_norm.weight")    == 0) return &lw->ln2_weight;
         if (strcmp(rest, "ffn_norm.bias")      == 0) return &lw->ln2_bias;
         if (strcmp(rest, "ffn_up.weight")      == 0) return &lw->ffn_fc_weight;
         if (strcmp(rest, "ffn_up.bias")        == 0) return &lw->ffn_fc_bias;
         if (strcmp(rest, "ffn_down.weight")    == 0) return &lw->ffn_proj_weight;
         if (strcmp(rest, "ffn_down.bias")      == 0) return &lw->ffn_proj_bias;
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
     if (!f) {
         fprintf(stderr, "[ERROR] Cannot open GGUF file: %s\n", path);
         exit(1);
     }
 
     /* ---- Header ---- */
     uint32_t magic   = gguf_u32(f);
     uint32_t version = gguf_u32(f);
 
     if (magic != GGUF_MAGIC) {
         fprintf(stderr, "[ERROR] Not a GGUF file (magic=0x%08X)\n", magic);
         fclose(f); exit(1);
     }
     if (version < GGUF_VERSION_MIN || version > GGUF_VERSION_MAX) {
         fprintf(stderr, "[ERROR] Unsupported GGUF version %u (supported: %u-%u)\n",
                 version, GGUF_VERSION_MIN, GGUF_VERSION_MAX);
         fclose(f); exit(1);
     }
     printf("[INFO] GGUF version %u\n", version);
 
     uint64_t n_tensors = gguf_u64(f);
     uint64_t n_kv      = gguf_u64(f);
 
     printf("[INFO] Tensors: %llu  Metadata KV pairs: %llu\n",
            (unsigned long long)n_tensors, (unsigned long long)n_kv);
 
     /* ---- Skip metadata key-value pairs ---- */
     for (uint64_t i = 0; i < n_kv; i++) {
         char *key = gguf_read_string(f, NULL);
         uint32_t vtype = gguf_u32(f);
         gguf_skip_value(f, vtype);
         free(key);
     }
 
     /* ---- Read tensor info ---- */
     if (n_tensors > GGUF_MAX_TENSORS) {
         fprintf(stderr, "[ERROR] Too many tensors (%llu > %d)\n",
                 (unsigned long long)n_tensors, GGUF_MAX_TENSORS);
         fclose(f); exit(1);
     }
 
     GGUFTensor *tensors = (GGUFTensor*)calloc(n_tensors, sizeof(GGUFTensor));
     if (!tensors) { fprintf(stderr, "[FATAL] OOM\n"); exit(1); }
 
     for (uint64_t i = 0; i < n_tensors; i++) {
         GGUFTensor *t = &tensors[i];
         uint64_t name_len;
         char *name = gguf_read_string(f, &name_len);
         strncpy(t->name, name, sizeof(t->name) - 1);
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
     printf("[INFO] Parameters: %zu  (%.1f MB float32)\n",
            total, total * 4.0 / (1024*1024));
 
     /* Extra V*D floats reserved for lm_head (output.weight tensor in GGUF).
      * GPT-2 uses tied weights, but llama.cpp GGUF stores output.weight as a
      * separate tensor. We allocate space for it and default it to a copy of
      * wte; if output.weight is present it will overwrite this copy. */
     const size_t lm_head_size = (size_t)GPT2_VOCAB_SIZE * GPT2_EMBED_DIM;
     arena_init(total + lm_head_size);
     assign_weight_pointers();
     /* Allocate and default-populate lm_head from arena */
     g_weights.lm_head = arena_alloc(lm_head_size);
     memcpy(g_weights.lm_head, g_weights.wte, lm_head_size * sizeof(float));
 
     /* ---- Load each tensor ---- */
     int tensors_loaded = 0;
     for (uint64_t i = 0; i < n_tensors; i++) {
         GGUFTensor *t = &tensors[i];
 
         float **dst_ptr = gguf_name_to_ptr(t->name);
         if (!dst_ptr) {
             /* Skip tensors we don't need (e.g. lm_head if separate) */
             continue;
         }
         float *dst = *dst_ptr;
 
         long tensor_offset = data_start + (long)t->offset;
         fseek(f, tensor_offset, SEEK_SET);
 
         if (t->type == GGUF_TYPE_F32) {
             if (fread(dst, sizeof(float), t->n_elements, f) != t->n_elements) {
                 fprintf(stderr, "[ERROR] Short read on tensor %s\n", t->name);
                 fclose(f); free(tensors); exit(1);
             }
         } else if (t->type == GGUF_TYPE_F16) {
             /* Dequantize F16 → F32 */
             uint16_t *tmp = (uint16_t*)malloc(t->n_elements * sizeof(uint16_t));
             if (!tmp) { fprintf(stderr, "[FATAL] OOM F16 buf\n"); exit(1); }
             if (fread(tmp, sizeof(uint16_t), t->n_elements, f) != t->n_elements) {
                 fprintf(stderr, "[ERROR] Short read F16 tensor %s\n", t->name);
                 free(tmp); fclose(f); free(tensors); exit(1);
             }
             for (size_t e = 0; e < t->n_elements; e++) {
                 dst[e] = f16_to_f32(tmp[e]);
             }
             free(tmp);
         } else if (t->type == GGUF_TYPE_Q8_0) {
             /* Dequantize Q8_0 → F32 */
             if (t->n_elements % Q8_0_BLOCK_SIZE != 0) {
                 fprintf(stderr, "[ERROR] Q8_0 tensor %s has %zu elements "
                                 "(not a multiple of %d)\n",
                         t->name, t->n_elements, Q8_0_BLOCK_SIZE);
                 fclose(f); free(tensors); exit(1);
             }
             size_t raw_bytes = (t->n_elements / Q8_0_BLOCK_SIZE) * Q8_0_BYTES_PER_BLOCK;
             uint8_t *tmp = (uint8_t *)malloc(raw_bytes);
             if (!tmp) {
                 fprintf(stderr, "[FATAL] OOM for Q8_0 buffer (%s)\n", t->name);
                 exit(1);
             }
             if (fread(tmp, 1, raw_bytes, f) != raw_bytes) {
                 fprintf(stderr, "[ERROR] Short read on Q8_0 tensor %s\n", t->name);
                 free(tmp); fclose(f); free(tensors); exit(1);
             }
             dequant_q8_0(tmp, dst, t->n_elements);
             free(tmp);
         } else if (t->type == GGUF_TYPE_Q6_K) {
             /* Dequantize Q6_K → F32 */
             if (t->n_elements % Q6_K_BLOCK_SIZE != 0) {
                 fprintf(stderr, "[ERROR] Q6_K tensor %s has %zu elements "
                                 "(not a multiple of %d)\n",
                         t->name, t->n_elements, Q6_K_BLOCK_SIZE);
                 fclose(f); free(tensors); exit(1);
             }
             size_t raw_bytes = (t->n_elements / Q6_K_BLOCK_SIZE) * Q6_K_BYTES_PER_BLOCK;
             uint8_t *tmp = (uint8_t *)malloc(raw_bytes);
             if (!tmp) {
                 fprintf(stderr, "[FATAL] OOM for Q6_K buffer (%s)\n", t->name);
                 exit(1);
             }
             if (fread(tmp, 1, raw_bytes, f) != raw_bytes) {
                 fprintf(stderr, "[ERROR] Short read on Q6_K tensor %s\n", t->name);
                 free(tmp); fclose(f); free(tensors); exit(1);
             }
             dequant_q6k(tmp, dst, t->n_elements);
             free(tmp);
         } else if (t->type == GGUF_TYPE_Q5_K) {
             /* Dequantize Q5_K (S or M variant) → F32. Both variants share type 13. */
             if (t->n_elements % Q5_K_BLOCK_SIZE != 0) {
                 fprintf(stderr, "[ERROR] Q5_K tensor %s has %zu elements "
                                 "(not a multiple of %d)\n",
                         t->name, t->n_elements, Q5_K_BLOCK_SIZE);
                 fclose(f); free(tensors); exit(1);
             }
             size_t raw_bytes = (t->n_elements / Q5_K_BLOCK_SIZE) * Q5_K_BYTES_PER_BLOCK;
             uint8_t *tmp = (uint8_t *)malloc(raw_bytes);
             if (!tmp) {
                 fprintf(stderr, "[FATAL] OOM for Q5_K buffer (%s)\n", t->name);
                 exit(1);
             }
             if (fread(tmp, 1, raw_bytes, f) != raw_bytes) {
                 fprintf(stderr, "[ERROR] Short read on Q5_K tensor %s\n", t->name);
                 free(tmp); fclose(f); free(tensors); exit(1);
             }
             dequant_q5k(tmp, dst, t->n_elements);
             free(tmp);
         } else {
             fprintf(stderr, "[ERROR] Unsupported tensor type %u for %s\n",
                     t->type, t->name);
             fprintf(stderr, "  Supported: F32 (0), F16 (1), Q8_0 (8), Q5_K (13), Q6_K (14)\n");
             fprintf(stderr, "  To convert: ./quantize model.gguf out.gguf Q5_K_M\n");
             fclose(f); free(tensors); exit(1);
         }
 
         /* No transpose needed: llama.cpp GGUF already stores weights in [out, in] row-major. */
 
         tensors_loaded++;
     }
 
     free(tensors);
     fclose(f);
 
     /* If output.weight was not present (weight-tied GGUF), lm_head already
      * points to the wte copy made above. Either way we're correct. */
     printf("[INFO] Tensors loaded: %d\n", tensors_loaded);
     printf("[INFO] Loaded (GGUF F16): %s\n", path);
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
 typedef enum {
     FORMAT_UNKNOWN,
     FORMAT_BIN,
     FORMAT_GGUF
 } ModelFormat;
 
 static ModelFormat detect_format(const char *path) {
     /* Check file extension first (fast path) */
     size_t len = strlen(path);
     if (len >= 5 && strcmp(path + len - 5, ".gguf") == 0) return FORMAT_GGUF;
     if (len >= 4 && strcmp(path + len - 4, ".bin")  == 0) return FORMAT_BIN;
 
     /* Peek at magic bytes */
     FILE *f = fopen(path, "rb");
     if (!f) return FORMAT_UNKNOWN;
     uint32_t magic = 0;
     fread(&magic, sizeof(uint32_t), 1, f);
     fclose(f);
 
     if (magic == MODEL_MAGIC) return FORMAT_BIN;
     if (magic == GGUF_MAGIC)  return FORMAT_GGUF;
     return FORMAT_UNKNOWN;
 }
 
 /* Probe well-known default paths in order of preference */
 static const char* find_default_model(void) {
     static const char *candidates[] = {
         "gpt2_124m.bin",
         "gpt2.f16.gguf",
         "gpt2.gguf",
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
             printf("[INFO] Format: GGUF (.gguf) — F16/Q8_0/Q5_K/Q6_K dequantization\n");
             load_model_gguf(path);
             break;
         default:
             fprintf(stderr, "[ERROR] Cannot determine format of: %s\n", path);
             fprintf(stderr, "  Supported: *.bin (custom float32) or *.gguf (GGUF F16/F32)\n");
             exit(1);
     }
 }
 
 /* ============================================================
  * TOKENIZER
  * ============================================================ */
 static void init_byte_encoder(Tokenizer *tok) {
     int bs[256], cs[256];
     int n_bs = 0;
 
     for (int b = 33;  b <= 126; b++) { bs[n_bs] = b; cs[n_bs] = b; n_bs++; }
     for (int b = 161; b <= 172; b++) { bs[n_bs] = b; cs[n_bs] = b; n_bs++; }
     for (int b = 174; b <= 255; b++) { bs[n_bs] = b; cs[n_bs] = b; n_bs++; }
 
     int extra = 256;
     for (int b = 0; b < 256; b++) {
         int found = 0;
         for (int i = 0; i < n_bs; i++) { if (bs[i] == b) { found = 1; break; } }
         if (!found) { bs[n_bs] = b; cs[n_bs] = extra++; n_bs++; }
     }
 
     for (int i = 0; i < 256; i++) tok->byte_decoder[cs[i]] = bs[i];
 
     for (int i = 0; i < 256; i++) {
         int cp = cs[i];
         char *out = tok->byte_encoder[bs[i]];
         if (cp < 0x80) {
             out[0] = (char)cp; out[1] = '\0';
         } else if (cp < 0x800) {
             out[0] = (char)(0xC0 | (cp >> 6));
             out[1] = (char)(0x80 | (cp & 0x3F));
             out[2] = '\0';
         } else {
             out[0] = (char)(0xE0 | (cp >> 12));
             out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
             out[2] = (char)(0x80 | (cp & 0x3F));
             out[3] = '\0';
         }
     }
 }
 
 static int utf8_decode(const char **s) {
     unsigned char c = (unsigned char)**s;
     int cp;
     if (c < 0x80) {
         cp = c; (*s)++;
     } else if ((c & 0xE0) == 0xC0) {
         cp  = (c & 0x1F) << 6; (*s)++;
         cp |= ((unsigned char)**s & 0x3F); (*s)++;
     } else if ((c & 0xF0) == 0xE0) {
         cp  = (c & 0x0F) << 12; (*s)++;
         cp |= ((unsigned char)**s & 0x3F) << 6; (*s)++;
         cp |= ((unsigned char)**s & 0x3F); (*s)++;
     } else {
         cp = '?'; (*s)++;
     }
     return cp;
 }
 
 static uint32_t str_hash(const uint8_t *s, int len) {
     uint32_t h = 2166136261u;
     for (int i = 0; i < len; i++) { h ^= s[i]; h *= 16777619u; }
     return h;
 }
 
 static void vocab_hash_insert(Tokenizer *tok, int token_id) {
     uint32_t h    = str_hash(tok->vocab[token_id].bytes, tok->vocab[token_id].len);
     uint32_t slot = h % VOCAB_HASH_SIZE;
     tok->vocab_hash_next[token_id] = tok->vocab_hash[slot];
     tok->vocab_hash[slot] = token_id;
 }
 
 static int vocab_lookup(const Tokenizer *tok, const uint8_t *s, int len) {
     uint32_t slot = str_hash(s, len) % VOCAB_HASH_SIZE;
     int id = tok->vocab_hash[slot];
     while (id != -1) {
         if (tok->vocab[id].len == len &&
             memcmp(tok->vocab[id].bytes, s, len) == 0) return id;
         id = tok->vocab_hash_next[id];
     }
     return -1;
 }
 
 static void load_encoder_json(Tokenizer *tok, const char *path) {
     FILE *f = fopen(path, "r");
     if (!f) {
         fprintf(stderr, "[ERROR] Cannot open encoder.json: %s\n", path);
         exit(1);
     }
 
     memset(tok->vocab_hash,      -1, sizeof(tok->vocab_hash));
     memset(tok->vocab_hash_next, -1, sizeof(tok->vocab_hash_next));
     tok->vocab_size = 0;
 
     fseek(f, 0, SEEK_END);
     long fsize = ftell(f);
     fseek(f, 0, SEEK_SET);
 
     char *buf = (char*)malloc((size_t)fsize + 1);
     if (!buf) { fprintf(stderr, "[FATAL] OOM loading encoder.json\n"); fclose(f); exit(1); }
 
     if (fread(buf, 1, (size_t)fsize, f) != (size_t)fsize) {
         fprintf(stderr, "[ERROR] Short read on encoder.json\n");
         free(buf); fclose(f); exit(1);
     }
     buf[fsize] = '\0';
     fclose(f);
 
     char *p = buf;
     while (*p && *p != '{') p++;
     if (*p) p++;
 
     while (*p) {
         while (*p && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t' || *p == ',')) p++;
         if (*p == '}') break;
         if (*p != '"') { p++; continue; }
 
         p++;  /* skip '"' */
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
                         p++;
                         char hex[5] = {0};
                         for (int hi = 0; hi < 4 && *p; hi++) hex[hi] = *p++;
                         int cp = (int)strtol(hex, NULL, 16);
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
                     default: key[key_len++] = (uint8_t)*p++; break;
                 }
             } else {
                 key[key_len++] = (uint8_t)*p++;
             }
         }
         if (*p == '"') p++;
 
         while (*p && (*p == ' ' || *p == ':' || *p == '\t')) p++;
         if (*p < '0' || *p > '9') continue;
 
         int token_id = 0;
         while (*p >= '0' && *p <= '9') { token_id = token_id * 10 + (*p - '0'); p++; }
 
         if (token_id < BPE_MAX_VOCAB) {
             memcpy(tok->vocab[token_id].bytes, key, (size_t)key_len);
             tok->vocab[token_id].len = key_len;
             vocab_hash_insert(tok, token_id);
             if (token_id + 1 > tok->vocab_size) tok->vocab_size = token_id + 1;
         }
     }
 
     free(buf);
     printf("[INFO] Vocabulary loaded: %d tokens\n", tok->vocab_size);
 }
 
 /* FIX #2: unified skip logic (the original had a double-skip dead code path) */
 static void load_vocab_bpe(Tokenizer *tok, const char *path) {
     FILE *f = fopen(path, "r");
     if (!f) {
         fprintf(stderr, "[ERROR] Cannot open vocab.bpe: %s\n", path);
         exit(1);
     }
 
     tok->n_merges = 0;
     char line[1024];
 
     /* Skip version header line */
     if (!fgets(line, sizeof(line), f)) { fclose(f); return; }
 
     while (fgets(line, sizeof(line), f) && tok->n_merges < BPE_MAX_MERGES) {
         int len = (int)strlen(line);
         while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
         if (len == 0) continue;
 
         char *space = strchr(line, ' ');
         if (!space) continue;
         *space = '\0';
         char *left_str  = line;
         char *right_str = space + 1;
 
         int left_id  = vocab_lookup(tok, (const uint8_t*)left_str,  (int)strlen(left_str));
         int right_id = vocab_lookup(tok, (const uint8_t*)right_str, (int)strlen(right_str));
 
         /* FIX #2: single unified skip; original had unreachable dead code */
         if (left_id == -1 || right_id == -1) continue;
 
         int ll = (int)strlen(left_str), rl = (int)strlen(right_str);
         if (ll + rl >= BPE_TOKEN_MAX_LEN) continue;
 
         uint8_t merged[BPE_TOKEN_MAX_LEN];
         memcpy(merged, left_str, (size_t)ll);
         memcpy(merged + ll, right_str, (size_t)rl);
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
 
 static void load_tokenizer(const char *encoder_path, const char *bpe_path) {
     init_byte_encoder(&g_tokenizer);
     load_encoder_json(&g_tokenizer, encoder_path);
     load_vocab_bpe(&g_tokenizer, bpe_path);
 }
 
 /* ============================================================
  * BPE ENCODING
  * ============================================================ */
 #define MAX_WORD_LEN    128
 #define MAX_WORD_TOKENS (MAX_WORD_LEN * 4)
 
 typedef struct { int ids[MAX_WORD_TOKENS]; int len; } TokenSeq;
 
 static void bpe_apply_merges(TokenSeq *seq, const Tokenizer *tok) {
     while (seq->len >= 2) {
         int best_merge = tok->n_merges, best_pos = -1;
         for (int i = 0; i < seq->len - 1; i++) {
             int a = seq->ids[i], b = seq->ids[i+1];
             for (int m = 0; m < tok->n_merges; m++) {
                 if (tok->merges[m].left == a && tok->merges[m].right == b) {
                     if (m < best_merge) { best_merge = m; best_pos = i; }
                     break;
                 }
             }
         }
         if (best_pos == -1) break;
         seq->ids[best_pos] = tok->merges[best_merge].result;
         for (int i = best_pos + 1; i < seq->len - 1; i++) seq->ids[i] = seq->ids[i+1];
         seq->len--;
     }
 }
 
 static int encode_word(const Tokenizer *tok,
                        const uint8_t *word_bytes, int word_len,
                        int *out_ids) {
     TokenSeq seq = {.len = 0};
     for (int i = 0; i < word_len && seq.len < MAX_WORD_TOKENS; i++) {
         uint8_t b = word_bytes[i];
         const char *encoded = tok->byte_encoder[b];
         int tid = vocab_lookup(tok, (const uint8_t*)encoded, (int)strlen(encoded));
         seq.ids[seq.len++] = (tid == -1) ? (int)b : tid;
     }
     bpe_apply_merges(&seq, tok);
     for (int i = 0; i < seq.len; i++) out_ids[i] = seq.ids[i];
     return seq.len;
 }
 
 static int tokenize(const Tokenizer *tok, const char *text,
                     int *out_ids, int max_tokens) {
     int n_tokens = 0;
     const uint8_t *p = (const uint8_t*)text;
     int text_len = (int)strlen(text);
     int i = 0;
 
     while (i < text_len && n_tokens < max_tokens) {
         uint8_t word[MAX_WORD_LEN];
         int wlen = 0;
 
         if (p[i] == ' ' && i + 1 < text_len) word[wlen++] = p[i++];
 
         if (i >= text_len) {
             if (wlen > 0) {
                 int word_ids[MAX_WORD_TOKENS];
                 int n = encode_word(tok, word, wlen, word_ids);
                 for (int j = 0; j < n && n_tokens < max_tokens; j++) out_ids[n_tokens++] = word_ids[j];
             }
             break;
         }
 
         uint8_t c = p[i];
         if ((c>='A'&&c<='Z')||(c>='a'&&c<='z')||(c>='0'&&c<='9')||c>=0x80) {
             while (i < text_len && wlen < MAX_WORD_LEN - 1) {
                 uint8_t cc = p[i];
                 if ((cc>='A'&&cc<='Z')||(cc>='a'&&cc<='z')||(cc>='0'&&cc<='9')||cc>=0x80)
                     word[wlen++] = p[i++];
                 else break;
             }
         } else {
             word[wlen++] = p[i++];
         }
 
         if (wlen > 0) {
             int word_ids[MAX_WORD_TOKENS];
             int n = encode_word(tok, word, wlen, word_ids);
             for (int j = 0; j < n && n_tokens < max_tokens; j++) out_ids[n_tokens++] = word_ids[j];
         }
     }
     return n_tokens;
 }
 
 static int detokenize_token(const Tokenizer *tok, int token_id,
                             char *out_buf, int buf_size) {
     if (token_id < 0 || token_id >= tok->vocab_size) return 0;
     const VocabEntry *ve = &tok->vocab[token_id];
     const char *s = (const char*)ve->bytes;
     const char *end = s + ve->len;
     int out_len = 0;
     while (s < end && out_len < buf_size - 1) {
         int cp = utf8_decode(&s);
         if (cp >= 0 && cp < 0x400) out_buf[out_len++] = (char)tok->byte_decoder[cp];
     }
     out_buf[out_len] = '\0';
     return out_len;
 }
 
 /* ============================================================
  * GENERATION LOOP
  * ============================================================ */
 static void generate(const char *prompt, int max_new_tokens,
                      float temperature, float top_p) {
     int prompt_tokens[GPT2_SEQ_LEN];
     int n_prompt = tokenize(&g_tokenizer, prompt, prompt_tokens, GPT2_SEQ_LEN);
 
     if (n_prompt == 0) {
         fprintf(stderr, "[ERROR] Empty prompt after tokenization.\n");
         return;
     }
     printf("[INFO] Prompt tokens: %d\n", n_prompt);
     printf("\n--- Generated Text ---\n%s", prompt);
     fflush(stdout);
 
     g_kv_cache.seq_len = 0;
     float *logits = NULL;
     for (int i = 0; i < n_prompt; i++) {
         logits = model_forward(prompt_tokens[i], i);
         g_kv_cache.seq_len = i + 1;
     }
 
     int pos = n_prompt;
     char decode_buf[64];
 
     for (int step = 0; step < max_new_tokens; step++) {
         if (pos >= GPT2_SEQ_LEN) { printf("\n[Context window full]\n"); break; }
 
         int next_token = sample_top_p(logits, temperature, top_p);
         if (next_token == 50256) { printf("\n[EOS]\n"); break; }
 
         int dec_len = detokenize_token(&g_tokenizer, next_token, decode_buf, sizeof(decode_buf));
         if (dec_len > 0) { fwrite(decode_buf, 1, (size_t)dec_len, stdout); fflush(stdout); }
 
         logits = model_forward(next_token, pos);
         g_kv_cache.seq_len = pos + 1;
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
     printf("  lm.c — Unified GPT-2 Inference (C99)\n");
     printf("  Backends: custom .bin  |  GGUF .gguf (F32/F16/Q8_0/Q5_K/Q6_K)\n");
     printf("==============================================\n\n");
 
     const char *prompt      = "Hello, world!";
     int         max_tokens  = 64;
     float       temperature = 0.7f;
     float       top_p       = 0.9f;
     const char *model_path  = NULL;
     const char *encoder_path = "encoder.json";
     const char *bpe_path     = "vocab.bpe";
 
     /* Parse args */
     for (int i = 1; i < argc; i++) {
         if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
             model_path = argv[++i];
         } else if (strcmp(argv[i], "--encoder") == 0 && i + 1 < argc) {
             encoder_path = argv[++i];
         } else if (strcmp(argv[i], "--bpe") == 0 && i + 1 < argc) {
             bpe_path = argv[++i];
         } else if (i == 1 && argv[i][0] != '-') {
             prompt = argv[i];
         } else if (i == 2 && argv[i][0] != '-') {
             max_tokens = atoi(argv[i]);
         } else if (i == 3 && argv[i][0] != '-') {
             temperature = (float)atof(argv[i]);
         } else if (i == 4 && argv[i][0] != '-') {
             top_p = (float)atof(argv[i]);
         }
     }
 
     /* Validate */
     if (max_tokens <= 0) max_tokens = 64;
     if (max_tokens > GPT2_SEQ_LEN - 10) max_tokens = GPT2_SEQ_LEN - 10;
     if (temperature < 0.0f) temperature = 0.7f;
     if (top_p <= 0.0f || top_p > 1.0f) top_p = 0.9f;
 
     /* Auto-detect model path */
     if (!model_path) {
         model_path = find_default_model();
         if (!model_path) {
             fprintf(stderr,
                 "[ERROR] No model file found. Tried: gpt2_124m.bin, gpt2.f16.gguf\n"
                 "  Run:  python3 converter.py          → generates gpt2_124m.bin\n"
                 "  Or:   ollama pull gpt2              → then locate the .gguf file\n"
                 "  Or pass: --model <path>\n");
             return 1;
         }
     }
 
     printf("[CONFIG] model:       %s\n", model_path);
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
 
     /* Cleanup */
     free(g_arena.data);
     free(g_kv_cache.k_cache);
     free(g_kv_cache.v_cache);
     free_activations();
 
     return 0;
 }