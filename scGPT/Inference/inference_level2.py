#!/usr/bin/env python3
"""
inference_level2.py
===================
Level 2: Cell type classification (8 classes)

Mirrors level_2.py exactly:
  - vocab:       scGPT_brain/vocab.json          ← brain pretrained (NOT human)
  - HVGs:        scGPT_level2_celltype/hvg_genes.pkl
  - gene_ids:    built from adata.var["gene_name"] via vocab()
  - Preprocess:  normalize_total(1e4) → log1p → bin(n_bins) via Preprocessor
                 with filter_gene_by_counts=3, subset_hvg=n_hvg
                 (on inference: subset_hvg=False, already filtered)
  - Tokenize:    tokenize_and_pad_batch, max_len=3001, include_zero_gene=False
  - Forward:     model(gene_ids, values, src_key_padding_mask, batch_labels=None, CLS=True)
  - Output:      cls_output → softmax → hierarchical rejection

Rejection thresholds (thesis Section 3.4.2):
  τ_specific = 0.85  → accept specific cell type
  τ_family   = 0.65  → fall back to family group

Usage:
    # Raw counts (default - will preprocess)
    python inference_level2.py --input cells.h5ad --output results_l2.csv
    
    # Already preprocessed data (skip preprocessing)
    python inference_level2.py --input preprocessed.h5ad --output results_l2.csv --preprocessed
    
    # Specify device
    python inference_level2.py --input cells.h5ad --output results_l2.csv --device cpu
"""

import gc
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import scanpy as sc
from scipy.sparse import issparse
from torch.utils.data import Dataset, DataLoader

from scgpt.model import TransformerModel
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer import tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab

# ── PATHS ─────────────────────────────────────────────────────────────────────
SAVE_DIR    = Path("./save/scGPT_level2_celltype")
MODEL_PATH  = SAVE_DIR / "best_model.pt"
HVG_PATH    = SAVE_DIR / "hvg_genes.pkl"
LABEL_PATH  = SAVE_DIR / "id_to_label.json"
TAXONOMY_PATH = SAVE_DIR / "taxonomy.json"
VOCAB_DIR   = Path("./save/scGPT_brain")        # brain pretrained for L2
ARGS_PATH   = VOCAB_DIR / "args.json"
VOCAB_PATH  = VOCAB_DIR / "vocab.json"

# ── CONSTANTS (must match level_2.py exactly) ──────────────────────────────────
N_BINS      = 51       # config["n_bins"]
N_HVG       = 1500     # n_hvg in level_2.py
MAX_SEQ_LEN = 3001     # overrides n_hvg+1 (level_2.py line 647)
PAD_TOKEN   = "<pad>"
PAD_VALUE   = -2
MASK_VALUE  = -1
INCLUDE_ZERO_GENE = False
BATCH_SIZE  = 32

# ── THRESHOLDS (thesis Section 3.4.2) ─────────────────────────────────────────
TAU_SPECIFIC = 0.85
TAU_FAMILY   = 0.65


# ── PREPROCESSING ─────────────────────────────────────────────────────────────
def preprocess(adata, hvg_genes):
    """
    Filter to training HVGs then apply scGPT Preprocessor.
    level_2.py applies filter_gene_by_counts=3 and subset_hvg during training.
    At inference we skip those (already have the right genes) and just
    normalize → log1p → bin.
    """
    print(f"  Input:  {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    present = [g for g in hvg_genes if g in adata.var_names]
    missing = len(hvg_genes) - len(present)
    if missing > 0:
        print(f"  WARNING: {missing}/{len(hvg_genes)} HVGs missing")
    adata = adata[:, present].copy()

    # level_2.py sets gene_name = var_names (line 330)
    adata.var["gene_name"] = adata.var.index.tolist()

    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=False,   # already filtered to HVGs
        filter_cell_by_counts=False,
        normalize_total=1e4,
        result_normed_key="X_normed",
        log1p=True,
        result_log1p_key="X_logip",
        subset_hvg=False,              # already filtered
        binning=N_BINS,
        result_binned_key="X_binned",
    )
    preprocessor(adata, batch_key=None)
    print(f"  Ready:  {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    return adata


# ── TOKENISATION (mirrors level_2.py lines 1169-1200) ─────────────────────────
def tokenise(adata, vocab):
    """
    Build gene_ids from adata.var["gene_name"] via vocab(), then
    tokenize_and_pad_batch with the same parameters as training.
    """
    # gene_ids vector — same as level_2.py line 1090
    gene_ids = np.array(vocab(adata.var["gene_name"].tolist()), dtype=int)

    # Expression matrix from X_binned layer
    counts = (
        adata.layers["X_binned"].A
        if issparse(adata.layers["X_binned"])
        else adata.layers["X_binned"]
    )

    tokenized = tokenize_and_pad_batch(
        counts,
        gene_ids,
        max_len=MAX_SEQ_LEN,
        vocab=vocab,
        pad_token=PAD_TOKEN,
        pad_value=PAD_VALUE,
        append_cls=True,
        include_zero_gene=INCLUDE_ZERO_GENE,
    )
    return tokenized


# ── MODEL LOADING ──────────────────────────────────────────────────────────────
def load_model(vocab, n_cls, device):
    with open(ARGS_PATH) as f:
        cfg = json.load(f)

    # n_layers_cls from model config (level_2.py line 1076)
    n_layers_cls = cfg.get("n_layers_cls", 3)

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=cfg.get("embsize", 512),
        nhead=cfg.get("nheads", cfg.get("nhead", 8)),   # brain args.json uses "nheads"
        d_hid=cfg.get("d_hid", 512),
        nlayers=cfg.get("nlayers", 12),
        nlayers_cls=n_layers_cls,
        n_cls=n_cls,
        vocab=vocab,
        dropout=cfg.get("dropout", 0.1),
        pad_token=PAD_TOKEN,
        pad_value=PAD_VALUE,
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,        # INPUT_BATCH_LABELS=False in level_2.py
        num_batch_labels=1,
        domain_spec_batchnorm=False,   # DSBN=False
        input_emb_style="continuous",
        n_input_bins=N_BINS,
        cell_emb_style="cls",
        mvc_decoder_style="inner product",
        ecs_threshold=0.0,
        explicit_zero_prob=False,
        use_fast_transformer=True,
        fast_transformer_backend="flash",
        pre_norm=False,
    ).to(device)

    saved   = torch.load(MODEL_PATH, map_location=device)
    current = model.state_dict()
    matched = {k: v for k, v in saved.items()
               if k in current and v.shape == current[k].shape}
    current.update(matched)
    model.load_state_dict(current)
    model.eval()
    print(f"  Loaded {len(matched)}/{len(saved)} weight tensors")
    return model


# ── INFERENCE ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, tokenized, device):
    """
    Batch through the model exactly as in evaluate() in level_2.py.
    No masking at inference (mask_ratio=0.0).
    """
    gene_ids_all = tokenized["genes"]    # (n_cells, seq_len)
    values_all   = tokenized["values"]   # (n_cells, seq_len)
    n_cells      = gene_ids_all.shape[0]

    all_probs = []
    for start in range(0, n_cells, BATCH_SIZE):
        end      = min(start + BATCH_SIZE, n_cells)
        gene_ids = gene_ids_all[start:end].to(device)
        values   = values_all[start:end].to(device)
        src_mask = gene_ids.eq(vocab_global[PAD_TOKEN])

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(
                gene_ids,
                values,
                src_key_padding_mask=src_mask,
                batch_labels=None,
                CLS=True,
            )
        probs = F.softmax(out["cls_output"], dim=-1)
        all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)   # (n_cells, n_cls)


# ── HIERARCHICAL REJECTION ─────────────────────────────────────────────────────
def apply_hierarchical_rejection(probs, id_to_label, taxonomy):
    """
    Level 2 rejection (thesis Section 3.4.2):
      1. If max(P) >= τ_specific  → accept specific cell type
      2. Else aggregate by family:
         P(family) = Σ P(celltypes in that family)
         If max(P_family) >= τ_family → report family name
      3. Else → rejected
    """
    # Build family → list of class ids
    family_to_ids = {}
    for class_id, name in id_to_label.items():
        if name in taxonomy:
            fam = taxonomy[name]["family"]
            family_to_ids.setdefault(fam, []).append(class_id)

    records = []
    for i, p in enumerate(probs):
        specific_conf = float(p.max())
        pred_id       = int(p.argmax())
        pred_label    = id_to_label[pred_id]

        if specific_conf >= TAU_SPECIFIC:
            records.append({
                "cell_idx":      i,
                "l2_pred":       pred_label,
                "l2_confidence": round(specific_conf, 4),
                "l2_decision":   "specific",
            })
            continue

        # Family aggregation
        best_fam, best_fam_prob = None, 0.0
        for fam, ids in family_to_ids.items():
            fam_prob = float(sum(p[cid] for cid in ids if cid < len(p)))
            if fam_prob > best_fam_prob:
                best_fam, best_fam_prob = fam, fam_prob

        if best_fam_prob >= TAU_FAMILY:
            records.append({
                "cell_idx":      i,
                "l2_pred":       f"Family:{best_fam}",
                "l2_confidence": round(best_fam_prob, 4),
                "l2_decision":   "family",
            })
        else:
            records.append({
                "cell_idx":      i,
                "l2_pred":       "rejected",
                "l2_confidence": round(specific_conf, 4),
                "l2_decision":   "rejected",
            })

    return records


# ── MAIN ──────────────────────────────────────────────────────────────────────
vocab_global = None   # needed inside run_inference closure

def main():
    global vocab_global

    parser = argparse.ArgumentParser(description="Level 2 inference: Cell type classification")
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--preprocessed", action="store_true",
                        help="Skip preprocessing (data already normalized/binned)")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Load artifacts ────────────────────────────────────────────────────────
    print("\nLoading artifacts...")
    with open(HVG_PATH,      "rb") as f: hvg_genes   = pickle.load(f)
    with open(LABEL_PATH)          as f: id_to_label = {int(k): v for k, v in json.load(f).items()}
    with open(TAXONOMY_PATH)       as f: taxonomy    = json.load(f)

    vocab = GeneVocab.from_file(VOCAB_PATH)
    for tok in [PAD_TOKEN, "<cls>", "<eoc>"]:
        if tok not in vocab:
            vocab.append_token(tok)
    vocab.set_default_index(vocab[PAD_TOKEN])
    vocab_global = vocab

    print(f"  Vocab:    {len(vocab)} tokens")
    print(f"  HVGs:     {len(hvg_genes)} genes")
    print(f"  Labels:   {id_to_label}")
    print(f"  Families: {set(v['family'] for v in taxonomy.values())}")

    # ── Load & preprocess ─────────────────────────────────────────────────────
    print(f"\nLoading: {args.input}")
    adata = sc.read_h5ad(args.input)
    
    if args.preprocessed:
        print("Using pre-processed data (skipping normalization/binning)")
        if "X_binned" in adata.layers:
            print("  Found X_binned in layers")
        elif adata.X.max() <= 51 and adata.X.min() >= 0:
            print("  Data in .X appears binned (creating X_binned layer)")
            adata.layers["X_binned"] = adata.X.copy()
        else:
            raise ValueError(
                "--preprocessed flag used but cannot find binned data.\n"
                f"Expected X_binned layer or binned values in .X (range 0-51), "
                f"but got .X range: {adata.X.min():.2f} to {adata.X.max():.2f}"
            )
        if "gene_name" not in adata.var.columns:
            adata.var["gene_name"] = adata.var.index.tolist()
        # Make var_names unique if needed
        if not adata.var_names.is_unique:
            print("  Making var_names unique...")
            adata.var_names_make_unique()
        present = [g for g in hvg_genes if g in adata.var_names]
        if len(present) != len(hvg_genes):
            print(f"  WARNING: {len(hvg_genes) - len(present)} HVGs missing")
        adata = adata[:, present].copy()
        print(f"  Ready: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    else:
        print("Preprocessing raw counts...")
        adata = preprocess(adata, hvg_genes)

    # ── Tokenise ──────────────────────────────────────────────────────────────
    print("Tokenising...")
    tokenized = tokenise(adata, vocab)
    print(f"  Tokenized shape: {tokenized['genes'].shape}")

    # ── Build model & run ─────────────────────────────────────────────────────
    print("Building model...")
    model = load_model(vocab, n_cls=len(id_to_label), device=device)

    print(f"\nRunning inference on {adata.n_obs:,} cells...")
    probs = run_inference(model, tokenized, device)

    del model, tokenized
    torch.cuda.empty_cache()
    gc.collect()

    # ── Rejection & save ──────────────────────────────────────────────────────
    records = apply_hierarchical_rejection(probs, id_to_label, taxonomy)
    df = pd.DataFrame(records)

    print("\n── Results ──────────────────────────────────────")
    print(df["l2_decision"].value_counts().to_string())
    print("\nPrediction distribution:")
    print(df["l2_pred"].value_counts().to_string())

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()