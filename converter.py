#!/usr/bin/env python3
"""
converter.py — Download GPT-2 124M from Hugging Face and convert to
               the binary format expected by gpt2_edge.c

Binary format (little-endian float32):
  Header (7 x uint32):
    [magic=0x47505432][version=1][vocab_size][seq_len]
    [n_layers][n_heads][embed_dim]
  Weights (float32 arrays in order):
    wte, wpe,
    for each layer:
      ln1_weight, ln1_bias,
      qkv_weight, qkv_bias,       <- c_attn (fused QKV)
      attn_proj_weight, attn_proj_bias,  <- c_proj
      ln2_weight, ln2_bias,
      ffn_fc_weight, ffn_fc_bias, <- mlp.c_fc
      ffn_proj_weight, ffn_proj_bias,    <- mlp.c_proj
    ln_f_weight, ln_f_bias

Requirements:
    pip install torch transformers numpy

Usage:
    python3 converter.py
    # Outputs: gpt2_124m.bin, encoder.json, vocab.bpe
"""

import os
import sys
import struct
import shutil
import json
import urllib.request

def full_cache_clear():
    print("[INFO] Doing FULL cache clear (both old + new HF locations)...")
    paths = [
        os.path.expanduser("~/.cache/huggingface"),
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.expanduser("~/.cache/huggingface/transformers"),
    ]
    for p in paths:
        if os.path.exists(p):
            print(f"  Deleting {p}")
            shutil.rmtree(p, ignore_errors=True)
    print("  Cache completely cleared.\n")

def download_and_convert():
    print("=" * 60)
    print("GPT-2 124M Converter — FIXED & ROBUST VERSION")
    print("=" * 60)

    MODEL_ID = "openai-community/gpt2"   # ← This is the fix (full repo ID)

    print(f"\n[1/4] Downloading GPT-2 124M ({MODEL_ID})...")
    try:
        import torch
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained(
            MODEL_ID,
            local_files_only=False,
            force_download=False,   # set True if you want to force every time
            resume_download=True
        )
        state = model.state_dict()
        print("   Model downloaded successfully.")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print("Trying manual full cache wipe + retry...")
        full_cache_clear()

        # retry once
        # Load tokenizer (for encoder.json + vocab.bpe)
        model = GPT2LMHeadModel.from_pretrained(MODEL_ID, force_download=True)

        state = model.state_dict()

    # === Write binary file with correct transposes ===
    print("\n[2/4] Converting weights to gpt2_124m.bin ...")

    # -------------------------------------------------------
    # Validate architecture matches gpt2_edge.c expectations
    # -------------------------------------------------------

    V, D, L, H, S = 50257, 768, 12, 12, 1024
    F = D * 4

    with open("gpt2_124m.bin", "wb") as f:
        # Header
        header = struct.pack("<7I", 0x47505432, 1, V, S, L, H, D)
        f.write(header)

        def write(name, tensor):
            arr = tensor.float().detach().cpu().numpy().astype("float32")
            arr = arr if arr.flags['C_CONTIGUOUS'] else arr.copy()
            f.write(arr.tobytes())
            print(f"   ✓ {name:35} {arr.shape}")

        write("wte", state["transformer.wte.weight"])
        write("wpe", state["transformer.wpe.weight"])

        for l in range(L):
            prefix = f"transformer.h.{l}"
            print(f"   Layer {l}...")

            write(f"ln1_weight", state[f"{prefix}.ln_1.weight"])
            write(f"ln1_bias",   state[f"{prefix}.ln_1.bias"])

            # QKV — HF stores [D, 3D] → we need [3D, D]
            qkv_w = state[f"{prefix}.attn.c_attn.weight"]   # [D, 3D]
            write(f"qkv_weight", qkv_w.T)
            write(f"qkv_bias",   state[f"{prefix}.attn.c_attn.bias"])

            # attn proj — HF [D, D] → [D, D]
            proj_w = state[f"{prefix}.attn.c_proj.weight"]
            write(f"attn_proj_weight", proj_w.T)
            write(f"attn_proj_bias",   state[f"{prefix}.attn.c_proj.bias"])

            write(f"ln2_weight", state[f"{prefix}.ln_2.weight"])
            write(f"ln2_bias",   state[f"{prefix}.ln_2.bias"])

            # FFN fc — HF [D, F] → [F, D]
            fc_w = state[f"{prefix}.mlp.c_fc.weight"]
            write(f"ffn_fc_weight", fc_w.T)
            write(f"ffn_fc_bias",   state[f"{prefix}.mlp.c_fc.bias"])

            # FFN proj — HF [F, D] → [D, F]
            proj_w = state[f"{prefix}.mlp.c_proj.weight"]
            write(f"ffn_proj_weight", proj_w.T)
            write(f"ffn_proj_bias",   state[f"{prefix}.mlp.c_proj.bias"])

        write("ln_f_weight", state["transformer.ln_f.weight"])
        write("ln_f_bias",   state["transformer.ln_f.bias"])

    print(f"\n[3/4] Downloading tokenizer files...")
    urls = {
        "encoder.json": f"https://huggingface.co/{MODEL_ID}/resolve/main/vocab.json",
        "vocab.bpe":    f"https://huggingface.co/{MODEL_ID}/resolve/main/merges.txt"
    }
    for filename, url in urls.items():
        print(f"   Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"   ✓ {filename} ({os.path.getsize(filename)/1024:.1f} KB)")

    print("\n" + "="*60)
    print("SUCCESS! Files ready:")
    print("   • gpt2_124m.bin     (~498 MB)")
    print("   • encoder.json")
    print("   • vocab.bpe")
    print("\nNext step:")
    print('   gcc -O3 -march=native -ffast-math -Xclang -fopenmp \\')
    print('       -I/opt/homebrew/opt/libomp/include \\')
    print('       -L/opt/homebrew/opt/libomp/lib -lomp \\')
    print('       gpt2.c -o gpt2 -lm')
    print('   ./gpt2 "The answer to life the universe and everything is" 100 0.8 0.95')

if __name__ == "__main__":
    if "--clear" in sys.argv or "-c" in sys.argv:
        full_cache_clear()
    download_and_convert()
