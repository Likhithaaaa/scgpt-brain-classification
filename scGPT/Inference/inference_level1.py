#!/usr/bin/env python3
"""
inference_level1.py
===================
Level 1: Brain vs Non-Brain classification

Mirrors level_1.py exactly:
  - vocab:       scGPT_human/vocab.json
  - HVGs:        scGPT_level1_brain_vs_nonbrain/hvg_genes.pkl
  - Preprocess:  normalize_total(1e4) → log1p → bin(n_bins=51)
  - Dataset:     Level1Dataset (same gene_ids tensor, X_binned values)
  - Forward:     model(gene_ids, values, src_key_padding_mask, batch_labels=None, CLS=True)
  - Output:      cls_output → softmax → threshold τ=0.45

Usage:
    # Raw counts (default - will preprocess)
    python inference_level1.py --input cells.h5ad --output results_l1.csv
    
    # Already preprocessed data (skip preprocessing)
    python inference_level1.py --input preprocessed.h5ad --output results_l1.csv --preprocessed
    
    # Specify device
    python inference_level1.py --input cells.h5ad --output results_l1.csv --device cpu
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
from scgpt.tokenizer.gene_tokenizer import GeneVocab

# ── PATHS ─────────────────────────────────────────────────────────────────────
# Edit these to match your DeNBI directory layout
SAVE_DIR    = Path("./save/scGPT_level1_brain_vs_nonbrain")
MODEL_PATH  = SAVE_DIR / "best_model.pt"
HVG_PATH    = SAVE_DIR / "hvg_genes.pkl"
LABEL_PATH  = SAVE_DIR / "id_to_label.json"
VOCAB_DIR   = Path("./save/scGPT_human")        # human pretrained for L1
ARGS_PATH   = VOCAB_DIR / "args.json"
VOCAB_PATH  = VOCAB_DIR / "vocab.json"

# ── CONSTANTS (must match level_1.py exactly) ──────────────────────────────────
N_BINS      = 51
PAD_TOKEN   = "<pad>"
PAD_VALUE   = -2
BATCH_SIZE  = 32

# ── THRESHOLD (from thesis Section 3.4.1) ─────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.45


# ── DATASET (mirrors Level1Dataset in level_1.py) ─────────────────────────────
class Level1Dataset(Dataset):
    def __init__(self, adata, vocab):
        self.gene_ids = torch.from_numpy(np.array(
            [vocab[g] if g in vocab else vocab[PAD_TOKEN]
             for g in adata.var_names],
            dtype=np.int64
        )).long()
        # X_binned is set by Preprocessor
        self.values = adata.layers["X_binned"]

    def __len__(self):
        return self.values.shape[0]

    def __getitem__(self, idx):
        cell_values = self.values[idx]
        if issparse(cell_values):
            cell_values = cell_values.toarray().squeeze()
        return {
            "gene_ids": self.gene_ids,                                     # shared across all cells
            "values":   torch.from_numpy(cell_values.astype(np.float32)),
        }


# ── PREPROCESSING (mirrors level_1.py exactly) ────────────────────────────────
def preprocess(adata, hvg_genes):
    """Filter to training HVGs, then normalize→log1p→bin."""
    print(f"  Input:  {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # Keep only genes that were in training HVG list (in the same order)
    present = [g for g in hvg_genes if g in adata.var_names]
    missing = len(hvg_genes) - len(present)
    if missing > 0:
        print(f"  WARNING: {missing}/{len(hvg_genes)} HVGs missing in input — filling with zeros")

    # Subset to present HVGs first
    adata = adata[:, present].copy()

    # If some HVGs are missing, reindex to full HVG list (missing → zero column)
    if missing > 0:
        import anndata
        import scipy.sparse as sp
        n_cells = adata.n_obs
        # Build zero matrix for full HVG set
        full_X = sp.lil_matrix((n_cells, len(hvg_genes)), dtype=np.float32)
        present_set = set(present)
        present_idx = {g: i for i, g in enumerate(present)}
        for full_i, g in enumerate(hvg_genes):
            if g in present_idx:
                full_X[:, full_i] = adata[:, g].X
        full_X = full_X.tocsr()
        adata_full = anndata.AnnData(
            X=full_X,
            obs=adata.obs.copy(),
            var=pd.DataFrame(index=hvg_genes)
        )
        adata = adata_full

    # Preprocessor: normalize → log1p → bin (subset_hvg=False — already done)
    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=False,
        filter_cell_by_counts=False,
        normalize_total=1e4,
        log1p=True,
        subset_hvg=False,
        binning=N_BINS,
    )
    preprocessor(adata, batch_key=None)
    print(f"  Ready: {adata.n_obs:,} cells × {adata.n_vars:,} genes (X_binned created)")
    return adata


# ── MODEL LOADING ──────────────────────────────────────────────────────────────
def load_model(vocab, n_cls, device):
    with open(ARGS_PATH) as f:
        cfg = json.load(f)

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=cfg.get("embsize", 512),
        nhead=cfg.get("nhead", 8),
        d_hid=cfg.get("d_hid", 512),
        nlayers=cfg.get("nlayers", 12),
        nlayers_cls=3,
        n_cls=n_cls,
        vocab=vocab,
        dropout=cfg.get("dropout", 0.1),
        pad_token=PAD_TOKEN,
        pad_value=PAD_VALUE,
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,
        num_batch_labels=1,
        domain_spec_batchnorm=False,
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

    # Shape-safe weight loading (classifier head may differ from pretrained)
    saved   = torch.load(MODEL_PATH, map_location=device)
    current = model.state_dict()
    matched = {k: v for k, v in saved.items()
               if k in current and v.shape == current[k].shape}
    current.update(matched)
    model.load_state_dict(current)
    model.eval()
    print(f"  Loaded {len(matched)}/{len(saved)} weight tensors from {MODEL_PATH}")
    return model


# ── INFERENCE ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, adata, vocab, device):
    dataset = Level1Dataset(adata, vocab)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_probs = []
    for batch in loader:
        gene_ids = batch["gene_ids"].to(device)
        values   = batch["values"].to(device)
        src_mask = gene_ids.eq(vocab[PAD_TOKEN])

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

    return np.concatenate(all_probs, axis=0)   # (n_cells, 2)


# ── REJECTION LOGIC ───────────────────────────────────────────────────────────
def apply_rejection(probs, id_to_label):
    """
    L1 is binary — simple confidence threshold.
    τ = 0.45: permissive to preserve sensitivity (don't discard real brain cells).
    """
    records = []
    for i, p in enumerate(probs):
        confidence = float(p.max())
        pred_id    = int(p.argmax())
        pred_label = id_to_label[pred_id]

        if confidence >= CONFIDENCE_THRESHOLD:
            decision   = "accepted"
            final_pred = pred_label
        else:
            decision   = "ambiguous"
            final_pred = "ambiguous"

        records.append({
            "cell_idx":      i,
            "l1_pred":       final_pred,
            "l1_raw_pred":   pred_label,
            "l1_confidence": round(confidence, 4),
            "l1_brain_prob": round(float(p[1]), 4),   # prob of class 1 = brain
            "l1_decision":   decision,
            "is_brain":      final_pred == "brain",
        })
    return records


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Level 1 inference: Brain vs Non-Brain")
    parser.add_argument("--input",  required=True, help="Input .h5ad file")
    parser.add_argument("--output", required=True, help="Output .csv path")
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto")
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
    with open(HVG_PATH,   "rb") as f: hvg_genes  = pickle.load(f)
    with open(LABEL_PATH)       as f: id_to_label = {int(k): v for k, v in json.load(f).items()}

    vocab = GeneVocab.from_file(VOCAB_PATH)
    for tok in [PAD_TOKEN, "<cls>", "<eoc>"]:
        if tok not in vocab:
            vocab.append_token(tok)
    vocab.set_default_index(vocab[PAD_TOKEN])

    print(f"  Vocab:  {len(vocab)} tokens")
    print(f"  HVGs:   {len(hvg_genes)} genes")
    print(f"  Labels: {id_to_label}")

    # ── Load & preprocess input ───────────────────────────────────────────────
    print(f"\nLoading input: {args.input}")
    adata = sc.read_h5ad(args.input)
    
    if args.preprocessed:
        print("Using pre-processed data (skipping normalization/binning)")
        # Verify binned data exists
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
        # Make var_names unique if needed
        if not adata.var_names.is_unique:
            print("  Making var_names unique...")
            adata.var_names_make_unique()
        # Verify genes match HVGs
        present = [g for g in hvg_genes if g in adata.var_names]
        if len(present) != len(hvg_genes):
            print(f"  WARNING: {len(hvg_genes) - len(present)} HVGs missing")
        adata = adata[:, present].copy()
        print(f"  Ready: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    else:
        print("Preprocessing raw counts...")
        adata = preprocess(adata, hvg_genes)

    # ── Build model & run ─────────────────────────────────────────────────────
    print("\nBuilding model...")
    model = load_model(vocab, n_cls=len(id_to_label), device=device)

    print(f"\nRunning inference on {adata.n_obs:,} cells...")
    probs = run_inference(model, adata, vocab, device)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    # ── Apply rejection & save ────────────────────────────────────────────────
    records = apply_rejection(probs, id_to_label)
    df = pd.DataFrame(records)

    print("\n── Results ──────────────────────────────────────")
    print(df["l1_decision"].value_counts().to_string())
    print(df["l1_pred"].value_counts().to_string())
    print(f"\nBrain cells accepted: {df['is_brain'].sum():,} / {len(df):,}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()