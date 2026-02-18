#!/usr/bin/env python
# coding: utf-8

# LEVEL 1: BRAIN VS NON-BRAIN CLASSIFICATION
# Binary classification using scGPT (Human Pretrained Model)
# ================================================================================

import os
import gc
import json
import copy
import time
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import issparse
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

import scgpt as scg
from scgpt.model import TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch
from scgpt.loss import masked_mse_loss
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int

# WandB import
import wandb

# CONFIGURATION - LEVEL 1
# ================================================================================

class Config:
    # Data paths
    uniklinik_path = "./data/Reference.h5ad" 
    
    # Model - CHANGED TO HUMAN PRETRAINED
    load_model = "./save/scGPT_human/best_model.pt"  
    model_dir = "./save/scGPT_human"  # Directory with vocab.json and args.json
    
    # Training
    epochs = 5
    batch_size = 8
    gradient_accumulation_steps = 16
    lr = 5e-5
    schedule_ratio = 0.9
    schedule_interval = 1
    
    # Architecture (will load from human model)
    layer_size = 512
    nlayers = 12
    nhead = 8
    dropout = 0.1
    n_bins = 51
    
    # Data - REDUCED HVG
    n_hvg = 1200
    include_zero_gene = False
    mask_ratio = 0.0
    
    # Training settings
    amp = True
    freeze = True 
    DSBN = False
    pre_norm = False
    
    # Rejection
    confidence_threshold = 0.75
    
    # Paths
    save_dir = Path("./save/scGPT_level1_brain_vs_nonbrain/")
    save_eval_interval = 2
    
    # WandB
    use_wandb = True
    wandb_project = "scGPT-Level1"
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()
config.save_dir.mkdir(parents=True, exist_ok=True)

set_seed(42)

print("="*80)
print("LEVEL 1: BRAIN vs NON-BRAIN CLASSIFICATION")
print("="*80)
print(f"Device: {config.device}")
print(f"Pretrained model: {config.load_model} (HUMAN)")
print(f"HVG count: {config.n_hvg}")
print(f"Freeze encoder: {config.freeze}")
print(f"WandB logging: {config.use_wandb}")
print(f"Save directory: {config.save_dir}")
print("="*80)


# WANDB INITIALIZATION
# ================================================================================

if config.use_wandb:
    wandb.init(
        project=config.wandb_project,
        config={
            "model": "scGPT_human_pretrained",
            "task": "level_1_binary_classification",
            "n_hvg": config.n_hvg,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "accumulation_steps": config.gradient_accumulation_steps,
            "learning_rate": config.lr,
            "freeze_encoder": config.freeze,
            "architecture": {
                "layer_size": config.layer_size,
                "nlayers": config.nlayers,
                "nhead": config.nhead,
                "dropout": config.dropout,
            },
        },
        tags=["level1", "binary", "human_pretrained", "frozen_encoder"],
    )
    print("\n✓ WandB initialized")
    print(f"  Project: {config.wandb_project}")
    print(f"  Dashboard: {wandb.run.get_url()}")


# ================================================================================
# STEP 1: LOAD AND COMBINE DATASETS (PBMC 10k + Brain 10k)
# ================================================================================

print("\n" + "="*80)
print("LOADING DATASETS")
print("="*80)
# Load PBMC 10k (non-brain) - FULL VERSION WITH ALL GENES
print("\n1. Loading PBMC 10k (non-brain cells)...")

pbmc_10k_path = "pbmc_10k_protein_v3.h5ad"

if not Path(pbmc_10k_path).exists():
    print("   Downloading PBMC 10k from ScanPy datasets...")
    
    try:
        # Use full 3k dataset (less filtered than 68k_reduced)
        print("   Using PBMC 3k as base (better gene coverage)...")
        adata_pbmc = sc.datasets.pbmc3k_processed()  # Has ~2000 genes
        
        # Upsample to 10k cells
        from sklearn.utils import resample
        target_size = 10000
        
        print(f"   Upsampling from {len(adata_pbmc):,} to {target_size:,} cells...")
        upsampled_idx = resample(
            np.arange(len(adata_pbmc)),
            n_samples=target_size,
            replace=True,  # With replacement to get 10k
            random_state=42
        )
        adata_pbmc = adata_pbmc[upsampled_idx].copy()
        
        # Save for reuse
        adata_pbmc.write_h5ad(pbmc_10k_path)
        print(f"   ✓ Created and saved to {pbmc_10k_path}")
        
    except Exception as e:
        print(f"   Error: {e}")
        print("   Using standard PBMC 3k (will upsample)...")
        adata_pbmc = sc.datasets.pbmc3k()
        
        # Upsample to 10k
        target_size = 10000
        upsampled_idx = resample(
            np.arange(len(adata_pbmc)),
            n_samples=target_size,
            replace=True,
            random_state=42
        )
        adata_pbmc = adata_pbmc[upsampled_idx].copy()
else:
    adata_pbmc = sc.read_h5ad(pbmc_10k_path)

adata_pbmc.var_names_make_unique()   # For genes
adata_pbmc.obs_names_make_unique()

# Load Uniklinik Reference (brain) - MATCH SIZE AT 10K
print("\n2. Loading Uniklinik Reference data (brain cells)...")

if not Path(config.uniklinik_path).exists():
    print(f"  ERROR: {config.uniklinik_path} not found!")
    raise FileNotFoundError(config.uniklinik_path)

adata_brain_full = sc.read_h5ad(config.uniklinik_path)
print(f"   ✓ Loaded: {adata_brain_full.shape[0]:,} cells, {adata_brain_full.shape[1]:,} genes")

adata_brain_full.var_names_make_unique()   # For genes
adata_brain_full.obs_names_make_unique() 

# Sample 10k brain cells (balanced with PBMC)
brain_subset_size = len(adata_pbmc)  #  Match PBMC size (10k)

print(f"\n3. Balancing datasets: sampling {brain_subset_size:,} brain cells...")

if adata_brain_full.n_obs > brain_subset_size:
    
    # Stratified sampling by cell type if available
    if 'celltype' in adata_brain_full.obs.columns:
        print("   Using stratified sampling by cell type...")
        
        cell_types = adata_brain_full.obs['celltype'].unique()
        samples_per_type = brain_subset_size // len(cell_types)
        
        sample_idx = []
        for ct in cell_types:
            ct_idx = np.where(adata_brain_full.obs['celltype'] == ct)[0]
            n_sample = min(len(ct_idx), samples_per_type)
            if n_sample > 0:
                selected = np.random.choice(ct_idx, size=n_sample, replace=False)
                sample_idx.extend(selected)
        
        # Fill remaining slots randomly if needed
        remaining = brain_subset_size - len(sample_idx)
        if remaining > 0:
            all_idx = np.arange(adata_brain_full.n_obs)
            available_idx = np.setdiff1d(all_idx, sample_idx)
            extra = np.random.choice(available_idx, size=remaining, replace=False)
            sample_idx.extend(extra)
        
        sample_idx = np.array(sample_idx)[:brain_subset_size]
        adata_brain = adata_brain_full[sample_idx].copy()
        
    else:
        # Random sampling
        print("   Using random sampling...")
        sample_idx = np.random.choice(
            adata_brain_full.n_obs, 
            size=brain_subset_size, 
            replace=False,
            random_state=42
        )
        adata_brain = adata_brain_full[sample_idx].copy()
    
    print(f"   ✓ Sampled: {adata_brain.shape[0]:,} cells")
else:
    adata_brain = adata_brain_full.copy()
    print(f"   ✓ Using all {adata_brain.shape[0]:,} cells")

del adata_brain_full
gc.collect()

adata_brain.obs['tissue'] = 'brain'
adata_brain.obs['level1_label'] = 'brain'

# Calculate gene detection in brain
if issparse(adata_brain.X):
    brain_genes_per_cell = np.array((adata_brain.X > 0).sum(axis=1)).flatten()
else:
    brain_genes_per_cell = (adata_brain.X > 0).sum(axis=1)

print(f"   ✓ Brain (Uniklinik): {adata_brain.shape[0]:,} cells, {adata_brain.shape[1]:,} genes")
print(f"   Brain gene detection: {brain_genes_per_cell.mean():.0f} ± {brain_genes_per_cell.std():.0f} genes/cell")

# Find common genes
print("\n4. Finding common genes...")
pbmc_genes = set(adata_pbmc.var_names)
brain_genes = set(adata_brain.var_names)
common_genes = list(pbmc_genes & brain_genes)

print(f"   PBMC genes: {len(pbmc_genes):,}")
print(f"   Brain genes: {len(brain_genes):,}")
print(f"   Common genes: {len(common_genes):,}")

if len(common_genes) < 1000:
    print(f"\n   WARNING: Only {len(common_genes)} common genes!")
    
    # Try conversion if needed
    if adata_brain.var_names[0].startswith('ENSG'):
        print("\n   Brain uses Ensembl IDs, converting to symbols...")
        if 'gene_name' in adata_brain.var.columns:
            adata_brain.var_names = adata_brain.var['gene_name']
            adata_brain.var_names_make_unique()
            common_genes = list(set(adata_pbmc.var_names) & set(adata_brain.var_names))
            print(f"   ✓ After conversion: {len(common_genes):,} common genes")

if len(common_genes) < 500:
    print("\n   ERROR: Too few common genes")
    raise ValueError("Insufficient gene overlap")

# Filter to common genes
print("\n5. Filtering to common genes...")
adata_pbmc = adata_pbmc[:, common_genes].copy()
adata_brain = adata_brain[:, common_genes].copy()
adata_brain = adata_brain[:, adata_pbmc.var_names].copy()  # Ensure same order

# Concatenate
print("\n6. Combining datasets...")
adata_combined = adata_pbmc.concatenate(
    adata_brain, 
    batch_key='dataset_origin',
    batch_categories=['pbmc', 'uniklinik']
)

adata_combined.obs['level1_label'] = adata_combined.obs['dataset_origin'].map({
    'pbmc': 'non_brain',
    'uniklinik': 'brain'
})

# Create numeric labels
adata_combined.obs['level1_id'] = (adata_combined.obs['level1_label'] == 'brain').astype(int)


# Summary
print(f"\n{'='*80}")
print("COMBINED DATASET SUMMARY")
print('='*80)
print(f"Total cells: {adata_combined.n_obs:,}")
print(f"Total genes: {adata_combined.n_vars:,}")
print(f"\nClass distribution:")
print(adata_combined.obs['level1_label'].value_counts())
print(f"\nDataset origin:")
print(adata_combined.obs['dataset_origin'].value_counts())

# Calculate gene detection for combined
if issparse(adata_combined.X):
    combined_genes_per_cell = np.array((adata_combined.X > 0).sum(axis=1)).flatten()
else:
    combined_genes_per_cell = (adata_combined.X > 0).sum(axis=1)

brain_mask = adata_combined.obs['level1_label'] == 'brain'
print(f"\nGene detection comparison:")
print(f"  Non-brain: {combined_genes_per_cell[~brain_mask].mean():.0f} ± {combined_genes_per_cell[~brain_mask].std():.0f}")
print(f"  Brain:     {combined_genes_per_cell[brain_mask].mean():.0f} ± {combined_genes_per_cell[brain_mask].std():.0f}")
print(f"  Ratio:     {combined_genes_per_cell[brain_mask].mean() / combined_genes_per_cell[~brain_mask].mean():.2f}×")

# Balance check
brain_count = (adata_combined.obs['level1_label'] == 'brain').sum()
nonbrain_count = (adata_combined.obs['level1_label'] == 'non_brain').sum()
imbalance_ratio = max(brain_count, nonbrain_count) / min(brain_count, nonbrain_count)

print(f"\nClass balance:")
print(f"  Brain: {brain_count:,}")
print(f"  Non-brain: {nonbrain_count:,}")
print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")

if imbalance_ratio > 1.5:
    print("   Classes not perfectly balanced")
else:
    print("   ✓ Well balanced")

# WandB logging
if config.use_wandb:
    wandb.log({
        "data/total_cells": adata_combined.n_obs,
        "data/total_genes": adata_combined.n_vars,
        "data/brain_cells": brain_count,
        "data/nonbrain_cells": nonbrain_count,
        "data/gene_detection_ratio": combined_genes_per_cell[brain_mask].mean() / combined_genes_per_cell[~brain_mask].mean(),
        "data/imbalance_ratio": imbalance_ratio,
    })

print('='*80)

gc.collect()



def analyze_level1_dataset_simplified(adata, save_dir, dataset_name="Level 1"):
    """
    Simplified visualization - remove redundant graphs
    """
    
    class_counts = adata.obs['level1_label'].value_counts()
    total_cells = len(adata)
    
    # Create ONLY 3 essential plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = ['#FF6B6B', '#4ECDC4']
    
    # 1. Class distribution (keep this one)
    ax1 = axes[0]
    bars = ax1.bar(class_counts.index, class_counts.values, 
                   color=colors, edgecolor='black', linewidth=2, width=0.6)
    ax1.set_ylabel('Number of Cells', fontsize=14, fontweight='bold')
    ax1.set_title('Class Distribution', fontweight='bold', fontsize=16, pad=20)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(class_counts.values) * 1.2)  # Add headroom
    
    # Add labels ABOVE bars (no overlap)
    for bar, count, label in zip(bars, class_counts.values, class_counts.index):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(class_counts.values)*0.03,
                f'{count:,}\n({100*count/total_cells:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # 2. UMI Distribution (important - shows data quality)
    ax2 = axes[1]
    
    if issparse(adata.X):
        cell_counts = np.array(adata.X.sum(axis=1)).flatten()
    else:
        cell_counts = adata.X.sum(axis=1)
    
    brain_counts = cell_counts[adata.obs['level1_label'] == 'brain']
    nonbrain_counts = cell_counts[adata.obs['level1_label'] == 'non_brain']
    
    ax2.hist(nonbrain_counts, bins=50, alpha=0.7, color=colors[0], 
             label='Non-brain', edgecolor='black', linewidth=1)
    ax2.hist(brain_counts, bins=50, alpha=0.7, color=colors[1], 
             label='Brain', edgecolor='black', linewidth=1)
    ax2.set_xlabel('Total UMI Counts', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Cells', fontsize=14, fontweight='bold')
    ax2.set_title('UMI Distribution by Class', fontweight='bold', fontsize=16, pad=20)
    ax2.legend(fontsize=12, loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Gene Detection (critical - shows the problem!)
    ax3 = axes[2]
    
    if issparse(adata.X):
        genes_per_cell = np.array((adata.X > 0).sum(axis=1)).flatten()
    else:
        genes_per_cell = (adata.X > 0).sum(axis=1)
    
    brain_genes = genes_per_cell[adata.obs['level1_label'] == 'brain']
    nonbrain_genes = genes_per_cell[adata.obs['level1_label'] == 'non_brain']
    
    # Box plot instead of violin (clearer)
    box_data = [nonbrain_genes, brain_genes]
    bp = ax3.boxplot(box_data, positions=[0, 1], widths=0.5,
                     patch_artist=True, showmeans=True,
                     medianprops=dict(color='black', linewidth=2),
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(['Non-brain', 'Brain'], fontsize=14, fontweight='bold')
    ax3.set_ylabel('Number of Detected Genes', fontsize=14, fontweight='bold')
    ax3.set_title('Gene Detection per Cell', fontweight='bold', fontsize=16, pad=20)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add statistics WITHOUT overlap
    stats_text = (f"Non-brain\nMean: {np.mean(nonbrain_genes):.0f}\n"
                  f"Median: {np.median(nonbrain_genes):.0f}\n\n"
                  f"Brain\nMean: {np.mean(brain_genes):.0f}\n"
                  f"Median: {np.median(brain_genes):.0f}")
    ax3.text(0.98, 0.97, stats_text, transform=ax3.transAxes, 
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Calculate imbalance
    imbalance_ratio = class_counts.max() / class_counts.min()
    if imbalance_ratio < 2:
        balance_status = "BALANCED"
        balance_color = "green"
    elif imbalance_ratio < 5:
        balance_status = "MODERATELY IMBALANCED"
        balance_color = "orange"
    else:
        balance_status = "HIGHLY IMBALANCED"
        balance_color = "red"
    
    # Calculate gene detection ratio
    gene_ratio = np.mean(brain_genes) / np.mean(nonbrain_genes)
    
    plt.suptitle(f'Level 1 Dataset Overview\n'
                 f'Status: {balance_status} | '
                 f'Gene Detection Ratio: {gene_ratio:.1f}×',
                 fontsize=18, fontweight='bold', color=balance_color, y=1.02)
    
    plt.tight_layout()
    plt.savefig(config.save_dir / "level1_data_overview_clean.png", dpi=300, bbox_inches='tight')
    
    # WandB
    if config.use_wandb:
        wandb.log({
            "preprocessing/data_overview": wandb.Image(str(save_dir / "level1_data_overview_clean.png")),
            "preprocessing/gene_detection_ratio": gene_ratio,
            "preprocessing/mean_genes_brain": float(np.mean(brain_genes)),
            "preprocessing/mean_genes_nonbrain": float(np.mean(nonbrain_genes)),
        })
    
    plt.show()
    
    # Print warning if gene detection is very different
    print("\n" + "="*60)
    print("DATA QUALITY ASSESSMENT")
    print("="*60)
    
    if gene_ratio > 2.5:
        print(" CRITICAL: Gene detection differs by >2.5×")
        print(f"   Non-brain: {np.mean(nonbrain_genes):.0f} genes")
        print(f"   Brain: {np.mean(brain_genes):.0f} genes")
        print("\n MODEL MAY LEARN DATA QUALITY INSTEAD OF CELL TYPE!")
        print("\nRECOMMENDATIONS:")
        print("  1. Use raw PBMC data (not scanpy's filtered version)")
        print("  2. Filter brain cells to match PBMC gene detection")
        print("  3. Use different non-brain dataset")
    elif gene_ratio > 1.5:
        print(" WARNING: Moderate gene detection difference")
        print(f"   Ratio: {gene_ratio:.2f}×")
        print("   Monitor if model learns biology vs data quality")
    else:
        print("✓ Gene detection is comparable between classes")
    
    print("="*60)


# Call simplified version
analyze_level1_dataset_simplified(adata_combined, config.save_dir, "Combined")


# TRAIN/VAL/TEST SPLIT
# ================================================================================

print("\n" + "="*80)
print("TRAIN/VAL/TEST SPLIT")
print("="*80)

train_idx, temp_idx = train_test_split(
    np.arange(adata_combined.n_obs),
    test_size=0.3,
    stratify=adata_combined.obs['level1_id'],
    random_state=42
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.4,
    stratify=adata_combined.obs['level1_id'].iloc[temp_idx],
    random_state=42
)

adata_train = adata_combined[train_idx].copy()
adata_val = adata_combined[val_idx].copy()
adata_test = adata_combined[test_idx].copy()

print(f"Train: {adata_train.shape[0]:,} cells")
print(f"Val:   {adata_val.shape[0]:,} cells")
print(f"Test:  {adata_test.shape[0]:,} cells")

if config.use_wandb:
    wandb.log({
        "data/train_cells": adata_train.shape[0],
        "data/val_cells": adata_val.shape[0],
        "data/test_cells": adata_test.shape[0],
    })

# ================================================================================
# STEP 3: PREPROCESSING WITH REDUCED HVG
# ================================================================================

print("\n" + "="*80)
print(f"PREPROCESSING (HVG={config.n_hvg})")
print("="*80)

# HVG selection on training set - USE RAW DATA
print(f"\n1. Selecting top {config.n_hvg} highly variable genes...")

# Create clean copy for HVG selection
adata_train_copy = adata_train.copy()

# Remove cells with zero counts
cell_counts = np.array(adata_train_copy.X.sum(axis=1)).flatten()
valid_cells = cell_counts > 0
print(f"   Removing {(~valid_cells).sum()} cells with zero counts...")
adata_train_copy = adata_train_copy[valid_cells].copy()

# Ensure we have raw counts (not logged/normalized)
# Check if data looks like raw counts
if adata_train_copy.X.max() < 50:  # Likely already normalized/logged
    print("  Data appears normalized, using cell_ranger method instead...")
    hvg_flavor = 'cell_ranger'
else:
    print("  Using seurat_v3 method...")
    hvg_flavor = 'seurat_v3'

# Normalize and compute HVG
sc.pp.normalize_total(adata_train_copy, target_sum=1e4)
sc.pp.log1p(adata_train_copy)

# Handle any remaining NaN/Inf
if issparse(adata_train_copy.X):
    adata_train_copy.X.data = np.nan_to_num(adata_train_copy.X.data, nan=0.0, posinf=0.0, neginf=0.0)
else:
    adata_train_copy.X = np.nan_to_num(adata_train_copy.X, nan=0.0, posinf=0.0, neginf=0.0)

# Select HVG
try:
    sc.pp.highly_variable_genes(
        adata_train_copy, 
        n_top_genes=config.n_hvg, 
        subset=False,
        flavor=hvg_flavor
    )
except Exception as e:
    print(f"   HVG selection failed with {hvg_flavor}, trying cell_ranger...")
    sc.pp.highly_variable_genes(
        adata_train_copy, 
        n_top_genes=config.n_hvg, 
        subset=False,
        flavor='cell_ranger'
    )

hvg_genes = adata_train_copy.var_names[adata_train_copy.var['highly_variable']].tolist()
print(f"   ✓ Selected {len(hvg_genes)} HVGs")

del adata_train_copy
gc.collect()

# ── SAVE INFERENCE ARTIFACTS ──────────────────────────────────────────────────
import pickle, json

# 1. HVG gene list
with open(config.save_dir / "hvg_genes.pkl", "wb") as f:
    pickle.dump(hvg_genes, f)
print(f"✓ Saved hvg_genes.pkl ({len(hvg_genes)} genes)")

# 2. Label mapping
# level1_id = (label == 'brain').astype(int) → 0=non_brain, 1=brain
id_to_label = {0: "non_brain", 1: "brain"}
with open(config.save_dir / "id_to_label.json", "w") as f:
    json.dump(id_to_label, f, indent=2)
print(f"✓ Saved id_to_label.json")
# ─────────────────────────────────────────────────────────────────────────────

# Filter all splits to HVG
print("\n2. Filtering to HVG...")
adata_train = adata_train[:, hvg_genes].copy()
adata_val = adata_val[:, hvg_genes].copy()
adata_test = adata_test[:, hvg_genes].copy()

print(f"   Train: {adata_train.shape}")
print(f"   Val:   {adata_val.shape}")
print(f"   Test:  {adata_test.shape}")

# Preprocess with scGPT preprocessor
print("\n3. Binning expression values...")
preprocessor = Preprocessor(
    use_key="X",
    filter_gene_by_counts=False,
    filter_cell_by_counts=False,
    normalize_total=1e4,
    log1p=True,
    subset_hvg=False,
    binning=config.n_bins,
)

for split_name, split_data in [("train", adata_train), ("val", adata_val), ("test", adata_test)]:
    print(f"   Processing {split_name}...")
    try:
        preprocessor(split_data, batch_key=None)
    except Exception as e:
        print(f"   Error in {split_name}: {e}")
        print(f"   Attempting recovery...")
        
        # Remove any problematic cells
        if issparse(split_data.X):
            cell_sums = np.array(split_data.X.sum(axis=1)).flatten()
        else:
            cell_sums = split_data.X.sum(axis=1)
        
        valid = (cell_sums > 0) & np.isfinite(cell_sums)
        split_data = split_data[valid].copy()
        
        # Try again
        preprocessor(split_data, batch_key=None)
    
    gc.collect()

print("\n✓ Preprocessing complete")

#  VOCABULARY (FROM HUMAN MODEL)
# ================================================================================

print("\n" + "="*80)
print("VOCABULARY (Human Pretrained)")
print("="*80)

from scgpt.tokenizer.gene_tokenizer import GeneVocab

vocab_file = Path(config.model_dir) / "vocab.json"
if not vocab_file.exists():
    print(f" ERROR: {vocab_file} not found!")
    raise FileNotFoundError(vocab_file)

vocab = GeneVocab.from_file(vocab_file)

pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
pad_value = -2

for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)

vocab.set_default_index(vocab[pad_token])

print(f"Vocabulary size: {len(vocab)}")

# Check gene coverage
genes_in_vocab = sum(1 for g in adata_train.var_names if g in vocab)
print(f"Genes in vocab: {genes_in_vocab}/{len(adata_train.var_names)} ({100*genes_in_vocab/len(adata_train.var_names):.1f}%)")

if genes_in_vocab < len(adata_train.var_names) * 0.8:
    print(" Warning: <80% gene coverage in vocabulary")

if config.use_wandb:
    wandb.log({
        "data/vocab_size": len(vocab),
        "data/genes_in_vocab": genes_in_vocab,
        "data/vocab_coverage": genes_in_vocab / len(adata_train.var_names),
    })

# DATASET AND DATALOADER
# ================================================================================

print("\n" + "="*80)
print("CREATING DATASETS")
print("="*80)

class Level1Dataset(Dataset):
    def __init__(self, adata, vocab):
        self.adata = adata
        self.gene_ids = torch.from_numpy(np.array(
            [vocab[g] if g in vocab else vocab[pad_token] for g in adata.var_names],
            dtype=np.int64
        )).long()
        
        self.values = adata.layers["X_binned"]
        self.labels = torch.from_numpy(adata.obs["level1_id"].values).long()
    
    def __len__(self):
        return self.adata.n_obs
    
    def __getitem__(self, idx):
        cell_values = self.values[idx]
        if issparse(cell_values):
            cell_values = cell_values.toarray().squeeze()
        
        return {
            "gene_ids": self.gene_ids,
            "values": torch.from_numpy(cell_values.astype(np.float32)),
            "labels": self.labels[idx],
        }

train_dataset = Level1Dataset(adata_train, vocab)
val_dataset = Level1Dataset(adata_val, vocab)
test_dataset = Level1Dataset(adata_test, vocab)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=False)

print(f"✓ Train: {len(train_loader)} batches")
print(f"✓ Val:   {len(val_loader)} batches")
print(f"✓ Test:  {len(test_loader)} batches")

# MODEL CREATION (HUMAN PRETRAINED)
# ================================================================================

print("\n" + "="*80)
print("MODEL CREATION (Human Pretrained)")
print("="*80)

device = torch.device(config.device)
ntokens = len(vocab)
num_classes = 2

# Load architecture config from human model
args_file = Path(config.model_dir) / "args.json"
if args_file.exists():
    with open(args_file, 'r') as f:
        model_configs = json.load(f)
    print(f"✓ Loaded model config from {args_file}")
    
    # Use config from human model
    embsize = model_configs.get("embsize", config.layer_size)
    d_hid = model_configs.get("d_hid", config.layer_size)
    nlayers = model_configs.get("nlayers", config.nlayers)
    nhead = model_configs.get("nhead", config.nhead)
    dropout = model_configs.get("dropout", config.dropout)
else:
    print("  args.json not found, using default config")
    embsize = config.layer_size
    d_hid = config.layer_size
    nlayers = config.nlayers
    nhead = config.nhead
    dropout = config.dropout

model = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    nlayers_cls=3,
    n_cls=num_classes,  # Binary: brain vs non-brain
    vocab=vocab,
    dropout=dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=False,
    do_dab=False,
    use_batch_labels=False,
    num_batch_labels=1,
    domain_spec_batchnorm=config.DSBN,
    input_emb_style="continuous",
    n_input_bins=config.n_bins,
    cell_emb_style="cls",
    mvc_decoder_style="inner product",
    ecs_threshold=0.0,
    explicit_zero_prob=False,
    use_fast_transformer=True,
    fast_transformer_backend="flash",
    pre_norm=config.pre_norm,
)

model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model created with {total_params:,} parameters")

# Load human pretrained weights
print(f"\nLoading human pretrained weights from {config.load_model}")
model_dict = model.state_dict()
pretrained_dict = torch.load(config.load_model, map_location=device)

# Load matching parameters (exclude classifier head)
pretrained_dict_filtered = {
    k: v for k, v in pretrained_dict.items()
    if k in model_dict and v.shape == model_dict[k].shape and "cls_decoder" not in k
}

model_dict.update(pretrained_dict_filtered)
model.load_state_dict(model_dict)
print(f"✓ Loaded {len(pretrained_dict_filtered)} pretrained parameters")

# Freeze encoder
if config.freeze:
    print("\nFreezing encoder parameters (training classifier head only)...")
    frozen_count = 0
    for name, param in model.named_parameters():
        if "cls_decoder" not in name:
            param.requires_grad = False
            frozen_count += 1

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params

print(f"\n✓ Freeze complete:")
print(f"  Total params:     {total_params:,}")
print(f"  Trainable params: {trainable_params:,}")
print(f"  Frozen params:    {frozen_params:,}")

if config.use_wandb:
    wandb.log({
        "model/total_params": total_params,
        "model/trainable_params": trainable_params,
        "model/frozen_params": frozen_params,
    })

# TRAINING SETUP
# ================================================================================

print("\n" + "="*80)
print("TRAINING SETUP")
print("="*80)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.schedule_interval, gamma=config.schedule_ratio)
scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

# Training history tracking
training_history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'learning_rates': [],
    'epoch_times': [],
}

print(f"Optimizer: Adam (lr={config.lr})")
print(f"Scheduler: StepLR (interval={config.schedule_interval}, gamma={config.schedule_ratio})")
print(f"Loss: CrossEntropyLoss")
print(f"Mixed precision: {config.amp}")
print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")

# TRAINING FUNCTIONS
# ================================================================================

def train_epoch(model, loader, optimizer, criterion, scaler, accumulation_steps):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(loader):
        gene_ids = batch["gene_ids"].to(device)
        values = batch["values"].to(device)
        labels = batch["labels"].to(device)
        
        src_key_padding_mask = gene_ids.eq(vocab[pad_token])
        
        with torch.cuda.amp.autocast(enabled=config.amp):
            output = model(
                gene_ids,
                values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
                CLS=True,
            )
            
            logits = output["cls_output"]
            loss = criterion(logits, labels) / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        
        # WandB batch logging
        if config.use_wandb and batch_idx % 50 == 0:
            wandb.log({
                "batch/loss": loss.item() * accumulation_steps,
                "batch/step": batch_idx,
            })
    
    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            gene_ids = batch["gene_ids"].to(device)
            values = batch["values"].to(device)
            labels = batch["labels"].to(device)
            
            src_key_padding_mask = gene_ids.eq(vocab[pad_token])
            
            with torch.cuda.amp.autocast(enabled=config.amp):
                output = model(
                    gene_ids,
                    values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                    CLS=True,
                )
            
            logits = output["cls_output"]
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs),
    }


# TRAINING LOOP
# ================================================================================

print("\n" + "="*80)
print("TRAINING")
print("="*80)

best_val_acc = 0.0
best_model = None
best_epoch = 0

for epoch in range(1, config.epochs + 1):
    print(f"\n{'='*80}")
    print(f"Epoch {epoch}/{config.epochs}")
    print('='*80)
    
    epoch_start = time.time()
    
    # Train
    train_loss, train_acc = train_epoch(
        model, train_loader, optimizer, criterion, scaler, 
        config.gradient_accumulation_steps
    )
    
    # Validate
    val_results = evaluate(model, val_loader, criterion)
    
    # Get current learning rate
    current_lr = scheduler.get_last_lr()[0]
    
    # Scheduler step
    scheduler.step()
    
    epoch_time = time.time() - epoch_start
    
    # Log metrics
    training_history['train_loss'].append(train_loss)
    training_history['train_acc'].append(train_acc)
    training_history['val_loss'].append(val_results['loss'])
    training_history['val_acc'].append(val_results['accuracy'])
    training_history['learning_rates'].append(current_lr)
    training_history['epoch_times'].append(epoch_time)
    
    print(f"\nEpoch {epoch} Results:")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}")
    print(f"  Val Loss:   {val_results['loss']:.4f}, Val Acc:   {val_results['accuracy']:.3f}")
    print(f"  LR: {current_lr:.2e}, Time: {epoch_time:.1f}s")
    
    # WandB epoch logging
    if config.use_wandb:
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_results['loss'],
            "val/accuracy": val_results['accuracy'],
            "train/learning_rate": current_lr,
            "train/epoch_time": epoch_time,
            "train/train_val_gap": train_acc - val_results['accuracy'],
        })
    
    # Save best model
    if val_results['accuracy'] > best_val_acc:
        best_val_acc = val_results['accuracy']
        best_model = copy.deepcopy(model)
        best_epoch = epoch
        torch.save(best_model.state_dict(), config.save_dir / "best_model.pt")
        print(f"  ✓ New best model saved (Acc: {best_val_acc:.3f})")
        
        if config.use_wandb:
            wandb.run.summary["best_val_accuracy"] = best_val_acc
            wandb.run.summary["best_epoch"] = best_epoch
    
    # Checkpoint
    if epoch % config.save_eval_interval == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'training_history': training_history,
        }, config.save_dir / f"checkpoint_epoch_{epoch}.pt")

print(f"\n{'='*80}")
print("TRAINING COMPLETE")
print(f"Best model: Epoch {best_epoch}, Val Acc: {best_val_acc:.3f}")
print('='*80)

# VISUALIZE TRAINING HISTORY
# ================================================================================

print("\n" + "="*80)
print("TRAINING VISUALIZATIONS")
print("="*80)

def plot_training_history(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History - Level 1 (Brain vs Non-Brain)', fontsize=16, y=1.02)
    
    # Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=6)
    ax1.plot(epochs, history['val_loss'], color='#FF8C00', marker='s', label='Val Loss', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    best_epoch_idx = np.argmin(history['val_loss'])
    ax1.axvline(x=best_epoch_idx+1, color='g', linestyle='--', alpha=0.5, label='Best Epoch')
    ax1.legend(fontsize=10)
    
    # Accuracy curves
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Acc', linewidth=2, markersize=6)
    ax2.plot(epochs, history['val_acc'], color='#FF8C00', marker='s', label='Val Acc', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=best_epoch_idx+1, color='g', linestyle='--', alpha=0.5)
    best_val_acc = max(history['val_acc'])
    ax2.axhline(y=best_val_acc, color='g', linestyle=':', alpha=0.3)
    ax2.text(0.5, best_val_acc, f'Best: {best_val_acc:.3f}', fontsize=10, color='green', va='bottom')
    
    # Learning rate
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['learning_rates'], 'g-o', linewidth=2, markersize=6)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Learning Rate', fontsize=12)
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Overfitting analysis
    ax4 = axes[1, 1]
    loss_gap = np.array(history['val_loss']) - np.array(history['train_loss'])
    acc_gap = np.array(history['train_acc']) - np.array(history['val_acc'])
    
    ax4_twin = ax4.twinx()
    line1 = ax4.plot(epochs, loss_gap, 'r-o', label='Loss Gap (Val-Train)', linewidth=2, markersize=6)
    line2 = ax4_twin.plot(epochs, acc_gap, 'b-s', label='Acc Gap (Train-Val)', linewidth=2, markersize=6)
    
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Loss Gap', fontsize=12, color='r')
    ax4_twin.set_ylabel('Accuracy Gap', fontsize=12, color='b')
    ax4.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
    ax4.tick_params(axis='y', labelcolor='r')
    ax4_twin.tick_params(axis='y', labelcolor='b')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4_twin.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=300, bbox_inches='tight')
    
    # Upload to WandB
    if config.use_wandb:
        wandb.log({"training/curves": wandb.Image(str(save_dir / "training_curves.png"))})
    
    plt.show()
    print("✓ Training curves saved")
    
    # Summary stats
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total training time: {sum(history['epoch_times']):.1f}s ({sum(history['epoch_times'])/60:.1f} min)")
    print(f"Best validation accuracy: {max(history['val_acc']):.4f} (Epoch {np.argmax(history['val_acc'])+1})")
    print(f"Final train-val gap: {history['train_acc'][-1] - history['val_acc'][-1]:.4f}")
    print("="*60)

plot_training_history(training_history, config.save_dir)

# TEST EVALUATION
# ================================================================================

print("\n" + "="*80)
print("TEST EVALUATION")
print("="*80)

model = best_model
test_results = evaluate(model, test_loader, criterion)

print(f"\nTest Results:")
print(f"  Loss: {test_results['loss']:.4f}")
print(f"  Accuracy: {test_results['accuracy']:.3f}")

# Classification report
class_names = ['non_brain', 'brain']
report = classification_report(
    test_results['labels'],
    test_results['predictions'],
    target_names=class_names,
    digits=3
)
print("\nClassification Report:")
print(report)

# WandB test results
if config.use_wandb:
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_results['labels'], 
        test_results['predictions'], 
        average='binary'
    )
    
    wandb.log({
        "test/loss": test_results['loss'],
        "test/accuracy": test_results['accuracy'],
        "test/precision": precision,
        "test/recall": recall,
        "test/f1": f1,
    })
    
    wandb.run.summary["test_accuracy"] = test_results['accuracy']

# Confusion matrix
cm = confusion_matrix(test_results['labels'], test_results['predictions'])
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Level 1 (Brain vs Non-Brain)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(config.save_dir / "confusion_matrix.png", dpi=300)

if config.use_wandb:
    wandb.log({"test/confusion_matrix": wandb.Image(str(config.save_dir / "confusion_matrix.png"))})

plt.show()


def create_binary_performance_analysis_level1(y_true, y_pred, y_probs, save_dir):
    """
    Create comprehensive binary classification performance analysis.
    Standardized format for Level 1.
    """
    from sklearn.metrics import (
        confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
        roc_curve, auc, precision_recall_curve, average_precision_score
    )
    
    # Calculate all metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    specificity = tn / (tn + fp)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    
    # False positive and false negative rates
    fpr_rate = fp / (fp + tn)
    fnr_rate = fn / (fn + tp)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ============================================
    # 1. METRICS BAR CHART
    # ============================================
    ax1 = axes[0]
    
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
    metrics_values = [accuracy, precision, recall, specificity, f1]
    
    colors = ['#2ca02c' if val >= 0.90 else '#ff7f0e' if val >= 0.80 else '#d62728' 
              for val in metrics_values]
    
    bars = ax1.bar(metrics_names, metrics_values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, val in zip(bars, metrics_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax1.set_title('Binary Classification Metrics', fontsize=16, fontweight='bold')
    ax1.set_ylim([0, 1.05])
    ax1.axhline(y=0.90, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Excellent (≥90%)')
    ax1.axhline(y=0.80, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='Good (≥80%)')
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # ============================================
    # 2. CONFUSION MATRIX WITH DETAILS
    # ============================================
    ax2 = axes[1]
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot heatmap
    im = ax2.imshow(cm_normalized, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    ax2.figure.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    # Labels
    class_names = ['Non-brain', 'Brain']
    tick_marks = np.arange(len(class_names))
    ax2.set_xticks(tick_marks)
    ax2.set_xticklabels(class_names, fontsize=12)
    ax2.set_yticks(tick_marks)
    ax2.set_yticklabels(class_names, fontsize=12)
    
    # Add text annotations
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax2.text(j, i, f'{cm_normalized[i, j]:.2f}\\n({cm[i, j]:,})',
                    ha="center", va="center", fontsize=12,
                    color="white" if cm_normalized[i, j] > thresh else "black",
                    fontweight='bold')
    
    ax2.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / "binary_performance_analysis.png", dpi=300, bbox_inches='tight')
    print(f"✓ Binary performance analysis saved to {save_dir / 'binary_performance_analysis.png'}")
    plt.show()
    
    # ============================================
    # DETAILED BREAKDOWN
    # ============================================
    print("\\n" + "="*60)
    print("CONFUSION MATRIX BREAKDOWN")
    print("="*60)
    print(f"True Negatives (TN):  {tn:6,} - Non-brain correctly identified")
    print(f"False Positives (FP): {fp:6,} - Non-brain wrongly called brain")
    print(f"False Negatives (FN): {fn:6,} - Brain wrongly called non-brain")
    print(f"True Positives (TP):  {tp:6,} - Brain correctly identified")
    print("="*60)
    
    print("\\n" + "="*60)
    print("BINARY METRICS")
    print("="*60)
    print(f"Accuracy:    {accuracy:.4f} - Overall correctness")
    print(f"Precision:   {precision:.4f} - Brain cells correctly identified")
    print(f"Recall:      {recall:.4f} - % of brain cells found")
    print(f"Specificity: {specificity:.4f} - % of non-brain correctly rejected")
    print(f"F1-Score:    {f1:.4f} - Harmonic mean of precision and recall")
    print("="*60)
    
    print("\\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)
    print(f"False Positive Rate: {fpr_rate:.4f} ({fpr_rate*100:.2f}%) - Over-predicting brain")
    print(f"False Negative Rate: {fnr_rate:.4f} ({fnr_rate*100:.2f}%) - Missing brain cells")
    
    if fpr_rate > fnr_rate:
        print("\\n  Model bias: Tends to over-predict brain cells")
    elif fnr_rate > fpr_rate:
        print("\\n  Model bias: Tends to under-predict brain cells")
    else:
        print("\\n Model is balanced (no significant FP/FN bias)")
    print("="*60)
    
    # Save metrics
    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'fpr': fpr_rate,
        'fnr': fnr_rate,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }
    
    import json
    with open(save_dir / 'binary_metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"\\n✓ Metrics saved to {save_dir / 'binary_metrics.json'}")
    
    return metrics_dict


# Call the function
print("\\n" + "="*80)
print("COMPREHENSIVE BINARY PERFORMANCE ANALYSIS")
print("="*80)

binary_metrics = create_binary_performance_analysis_level1(
    y_true=test_results['labels'], 
    y_pred=test_results['predictions'],  
    y_probs=test_results['probabilities'][:, 1],  # Probability of brain class
    save_dir=config.save_dir
)

# ========================================
# STRATEGY 2: Baseline Comparison
# ========================================

print("\n" + "-"*60)
print("BASELINE: Logistic Regression")
print("-"*60)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Prepare data
X_train = adata_train.layers['X_binned']
X_test = adata_test.layers['X_binned']

if issparse(X_train):
    X_train = X_train.toarray()
if issparse(X_test):
    X_test = X_test.toarray()

y_train = adata_train.obs['level1_id'].values
y_test = adata_test.obs['level1_id'].values

# Train simple baseline
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)

baseline_acc = lr.score(X_test_scaled, y_test)
baseline_preds = lr.predict(X_test_scaled)

# Calculate baseline metrics
baseline_precision, baseline_recall, baseline_f1, _ = precision_recall_fscore_support(
    y_test, baseline_preds, average='binary'
)

# Calculate scGPT metrics from test_results
scgpt_precision, scgpt_recall, scgpt_f1, _ = precision_recall_fscore_support(
    test_results['labels'], 
    test_results['predictions'], 
    average='binary'
)

print(f"\n Performance Comparison:")
print(f"{'Method':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
print("-"*60)
print(f"{'Logistic Regression':<25} {baseline_acc:<12.3f} {baseline_precision:<12.3f} {baseline_recall:<12.3f} {baseline_f1:<12.3f}")
print(f"{'scGPT (Human Pretrained)':<25} {test_results['accuracy']:<12.3f} {scgpt_precision:<12.3f} {scgpt_recall:<12.3f} {scgpt_f1:<12.3f}")  # ✅ FIXED
print("-"*60)

if baseline_acc >= 0.95:
    print("\n✓ Baseline achieves >95% accuracy, confirming task is fundamentally easy")
    print("  → This validates the biological separability of brain vs blood tissue")

# WandB logging
if config.use_wandb:
    wandb.log({
        "baseline/logistic_regression_acc": baseline_acc,
        "baseline/logistic_regression_f1": baseline_f1,
        "baseline/scgpt_vs_baseline_acc_diff": test_results['accuracy'] - baseline_acc,  
    })
    
# ============================================================================
# EMBEDDING EXTRACTION FUNCTION
# ============================================================================

def extract_test_embeddings(model):
    """Extract embeddings from test set for visualization"""
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=config.amp):  # Use autocast
            for batch_data in test_loader:
                input_gene_ids = batch_data["gene_ids"].to(device)
                input_values = batch_data["values"].to(device)
                src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
                
                # Forward pass to get embeddings
                output = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=True,
                )
                
                # Extract cell embeddings (CLS token output)
                embeddings = output["cell_emb"]  # Shape: (batch_size, d_model)
                all_embeddings.append(embeddings.cpu().numpy())
    
    # Concatenate all batches
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"✓ Extracted embeddings: {all_embeddings.shape}")
    
    return all_embeddings

# ============================================================================
# UMAP VISUALIZATION - LEVEL 1 (Binary Classification)
# ============================================================================

print("\n" + "="*80)
print("UMAP VISUALIZATION - LEVEL 1 (Brain vs Non-brain)")
print("="*80)

# Extract embeddings
print("Extracting embeddings...")
test_embeddings = extract_test_embeddings(best_model)

# Create visualization AnnData
adata_test_vis = adata_test.copy()

# CREATE MAPPING
id2type = {
    0: 'non_brain',
    1: 'brain'
}

# GET PREDICTIONS FROM DICTIONARY
test_predictions = test_results['predictions']
test_labels = test_results['labels']

# ADD PREDICTIONS AND LABELS
adata_test_vis.obs["predictions"] = [id2type[p] for p in test_predictions]
adata_test_vis.obs["true_labels"] = [id2type[label] for label in test_labels]
adata_test_vis.obsm["X_scgpt"] = test_embeddings

# Compute UMAP
print("Computing UMAP...")
sc.pp.neighbors(adata_test_vis, use_rep="X_scgpt", n_neighbors=15)
sc.tl.umap(adata_test_vis)

# Binary color palette
palette_ = {
    'brain': '#1f77b4',      # Blue
    'non_brain': '#d62728'   # Red
}

# Plot UMAP (side by side: true labels vs predictions)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sc.pl.umap(
    adata_test_vis,
    color="true_labels", 
    palette=palette_,
    ax=axes[0],
    show=False,
    title="True Cell Types",
    alpha=0.6
)

sc.pl.umap(
    adata_test_vis,
    color="predictions",
    palette=palette_,
    ax=axes[1],
    show=False,
    title="Predicted Cell Types",
    alpha=0.6
)

plt.tight_layout()
plt.savefig(config.save_dir / "umap_level1.png", dpi=300, bbox_inches='tight')
plt.show()

print("✓ UMAP visualization saved")
print("="*80)

# ============================================================================
# BINARY CLASSIFICATION PERFORMANCE ANALYSIS - LEVEL 1
# ============================================================================

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import numpy as np

print("\n" + "="*80)
print("BINARY CLASSIFICATION ANALYSIS - LEVEL 1")
print("="*80)

# Get predictions and labels
y_true = test_labels
y_pred = test_predictions

# Assuming binary: 0=non_brain, 1=brain
# Get prediction probabilities (you'll need to extract these from your model)
# For now, using binary predictions

# ============================================================================
# 1. CONFUSION MATRIX ANALYSIS
# ============================================================================

print("\n1. Confusion Matrix Analysis...")

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

# Calculate metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nBinary Classification Metrics:")
print(f"  Accuracy:    {accuracy:.3f}")
print(f"  Precision:   {precision:.3f} (Brain cells correctly identified)")
print(f"  Recall:      {recall:.3f} (% of brain cells found)")
print(f"  Specificity: {specificity:.3f} (% of non-brain correctly rejected)")
print(f"  F1-Score:    {f1:.3f}")

print(f"\nConfusion Matrix Breakdown:")
print(f"  True Negatives (Non-brain → Non-brain):  {tn:,}")
print(f"  False Positives (Non-brain → Brain):     {fp:,}")
print(f"  False Negatives (Brain → Non-brain):     {fn:,}")
print(f"  True Positives (Brain → Brain):          {tp:,}")

# ============================================================================
# 2. METRICS VISUALIZATION
# ============================================================================

print("\n2. Creating metrics visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart of metrics
ax1 = axes[0]
metrics_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
metrics_values = [accuracy, precision, recall, specificity, f1]

bars = ax1.bar(metrics_names, metrics_values, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6'], alpha=0.8)
ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('Binary Classification Metrics', fontsize=14, fontweight='bold')
ax1.set_ylim([0, 1.05])
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(y=0.90, color='green', linestyle='--', alpha=0.5, linewidth=2, label='90% threshold')
ax1.legend(fontsize=10)

# Add value labels on bars
for bar, value in zip(bars, metrics_values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Confusion matrix heatmap (normalized)
ax2 = axes[1]
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

import seaborn as sns
sns.heatmap(
    cm_normalized,
    annot=np.array([[f'{tn}\n({cm_normalized[0,0]:.1%})', f'{fp}\n({cm_normalized[0,1]:.1%})'],
                     [f'{fn}\n({cm_normalized[1,0]:.1%})', f'{tp}\n({cm_normalized[1,1]:.1%})']]),
    fmt='',
    cmap='Blues',
    xticklabels=['Non-brain', 'Brain'],
    yticklabels=['Non-brain', 'Brain'],
    cbar_kws={'label': 'Normalized Frequency'},
    ax=ax2,
    square=True
)

ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax2.set_ylabel('True', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(config.save_dir / "binary_metrics_level1.png", dpi=300, bbox_inches='tight')
plt.show()

print("✓ Saved: binary_metrics_level1.png")

# ============================================================================
# 3. ERROR ANALYSIS
# ============================================================================

print("\n3. Error Analysis...")

total = tp + tn + fp + fn
fp_rate = fp / total
fn_rate = fn / total

print(f"\nError Breakdown:")
print(f"  False Positive Rate: {fp_rate:.2%} ({fp:,} samples)")
print(f"    → Non-brain cells wrongly classified as brain")
print(f"  False Negative Rate: {fn_rate:.2%} ({fn:,} samples)")
print(f"    → Brain cells wrongly classified as non-brain")

if fp > fn:
    print(f"\n Model is more prone to FALSE POSITIVES (over-predicting brain cells)")
elif fn > fp:
    print(f"\n Model is more prone to FALSE NEGATIVES (missing brain cells)")
else:
    print(f"\n Model has balanced error rates")

# ============================================================================
# 4. CLASSIFICATION REPORT
# ============================================================================

print("\n4. Detailed Classification Report:")
print("="*60)
report = classification_report(
    y_true,
    y_pred,
    target_names=['Non-brain', 'Brain'],
    digits=3
)
print(report)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("BINARY CLASSIFICATION SUMMARY - LEVEL 1")
print("="*80)
print(f"Overall Accuracy: {accuracy:.1%}")
print(f"Brain Detection Rate (Recall): {recall:.1%}")
print(f"Non-brain Rejection Rate (Specificity): {specificity:.1%}")

if accuracy >= 0.95 and recall >= 0.90 and specificity >= 0.90:
    print("\n EXCELLENT: High accuracy with balanced performance")
elif accuracy >= 0.90:
    print("\n VERY GOOD: >90% accuracy achieved")
elif accuracy >= 0.85:
    print("\n GOOD: Acceptable binary classification performance")
else:
    print("\n NEEDS IMPROVEMENT: Consider retraining")

print("="*80)

print("\n✓ Binary classification analysis complete!")


# ========================================
# STRATEGY 3: Confidence Distribution
# ========================================

print("\n" + "-"*60)
print("CONFIDENCE DISTRIBUTION ANALYSIS")
print("-"*60)

confidences = np.max(test_results['probabilities'], axis=1)

print(f"\n Confidence Statistics:")
print(f"  Mean confidence:    {confidences.mean():.4f}")
print(f"  Median confidence:  {np.median(confidences):.4f}")
print(f"  Min confidence:     {confidences.min():.4f}")
print(f"  Max confidence:     {confidences.max():.4f}")
print(f"  Std confidence:     {confidences.std():.4f}")

# Count predictions by confidence level
high_conf = (confidences >= 0.95).sum()
medium_conf = ((confidences >= 0.80) & (confidences < 0.95)).sum()
low_conf = (confidences < 0.80).sum()

print(f"\n Prediction Confidence Breakdown:")
print(f"  High confidence (≥95%):     {high_conf:5d} ({100*high_conf/len(confidences):5.1f}%)")
print(f"  Medium confidence (80-95%): {medium_conf:5d} ({100*medium_conf/len(confidences):5.1f}%)")
print(f"  Low confidence (<80%):      {low_conf:5d} ({100*low_conf/len(confidences):5.1f}%)")

if confidences.min() > 0.99:
    print("\n✓ ALL predictions exceed 99% confidence")
    print("  → Model is extremely certain about every prediction")
    print("  → Confirms fundamental transcriptional differences between classes")

# Confidence by class
conf_brain = confidences[test_results['labels'] == 1]
conf_nonbrain = confidences[test_results['labels'] == 0]

print(f"\n Confidence by True Class:")
print(f"  Brain cells:     {conf_brain.mean():.4f} ± {conf_brain.std():.4f}")
print(f"  Non-brain cells: {conf_nonbrain.mean():.4f} ± {conf_nonbrain.std():.4f}")

# Plot confidence distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Histogram
ax1 = axes[0]
ax1.hist(conf_nonbrain, bins=30, alpha=0.6, color='#FF6B6B', label='Non-brain', edgecolor='black')
ax1.hist(conf_brain, bins=30, alpha=0.6, color='#4ECDC4', label='Brain', edgecolor='black')
ax1.axvline(0.95, color='red', linestyle='--', linewidth=2, label='95% threshold')
ax1.set_xlabel('Prediction Confidence', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Predictions', fontsize=12, fontweight='bold')
ax1.set_title('Confidence Distribution by True Class', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Right: Box plot
ax2 = axes[1]
bp = ax2.boxplot([conf_nonbrain, conf_brain], 
                  labels=['Non-brain', 'Brain'],
                  patch_artist=True,
                  showmeans=True,
                  widths=0.5)
colors = ['#FF6B6B', '#4ECDC4']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax2.axhline(0.95, color='red', linestyle='--', linewidth=2, alpha=0.5, label='95% threshold')
ax2.set_ylabel('Prediction Confidence', fontsize=12, fontweight='bold')
ax2.set_title('Confidence by Class (Boxplot)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend(fontsize=10)

plt.suptitle('Model Confidence Analysis - Level 1', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(config.save_dir / "confidence_analysis.png", dpi=300, bbox_inches='tight')

if config.use_wandb:
    wandb.log({"test/confidence_analysis": wandb.Image(str(config.save_dir / "confidence_analysis.png"))})

plt.show()

# WandB confidence metrics
if config.use_wandb:
    wandb.log({
        "test/mean_confidence": confidences.mean(),
        "test/min_confidence": confidences.min(),
        "test/high_confidence_pct": high_conf / len(confidences),
    })

print("\n" + "="*80)
print("✓ Baseline comparison and confidence analysis complete")
print("="*80)

# ================================================================================
# SPARSITY CHECK (After Test Evaluation)
# ================================================================================

print("\n" + "="*80)
print("SPARSITY vs BIOLOGY CHECK")
print("="*80)

# Get gene counts per cell from test set
if issparse(adata_test.X):
    genes_per_cell = np.array((adata_test.X > 0).sum(axis=1)).flatten()
else:
    genes_per_cell = (adata_test.X > 0).sum(axis=1)

# Get predictions and labels
test_preds = test_results['predictions']
test_labels = test_results['labels']

# Correlation between sparsity and predictions
corr, pval = pearsonr(genes_per_cell, test_preds)

print(f"\n Sparsity Analysis:")
print(f"Correlation(gene_count, prediction): {corr:.3f} (p={pval:.3e})")

if abs(corr) > 0.7:
    print(" WARNING: Model strongly relies on gene count!")
    print("   → Likely learned sparsity, not biology")
elif abs(corr) > 0.4:
    print(" MODERATE: Model partially relies on gene count")
    print("   → Mix of sparsity and biological features")
else:
    print("✓ Model doesn't overly rely on gene count")
    print("   → Likely learned biological features")

# Detailed breakdown
print(f"\nGene detection stats:")
print(f"  Non-brain (pred): {genes_per_cell[test_preds==0].mean():.0f} ± {genes_per_cell[test_preds==0].std():.0f}")
print(f"  Brain (pred):     {genes_per_cell[test_preds==1].mean():.0f} ± {genes_per_cell[test_preds==1].std():.0f}")
print(f"  Difference:       {abs(genes_per_cell[test_preds==1].mean() - genes_per_cell[test_preds==0].mean()):.0f} genes")

print("="*80)


# CONFIDENCE ANALYSIS
# ================================================================================

print("\n" + "="*80)
print("CONFIDENCE ANALYSIS")
print("="*80)

confidences = np.max(test_results['probabilities'], axis=1)
high_conf_mask = confidences >= config.confidence_threshold

print(f"Threshold: {config.confidence_threshold}")
print(f"Accepted: {high_conf_mask.sum()} cells ({100*high_conf_mask.mean():.1f}%)")
print(f"Rejected: {(~high_conf_mask).sum()} cells ({100*(~high_conf_mask).mean():.1f}%)")

if len(test_results['predictions'][high_conf_mask]) > 0:
    accepted_acc = accuracy_score(
        test_results['labels'][high_conf_mask],
        test_results['predictions'][high_conf_mask]
    )
    print(f"Accuracy on accepted: {accepted_acc:.3f}")
    
    if config.use_wandb:
        wandb.log({
            "test/rejection_rate": (~high_conf_mask).mean(),
            "test/accepted_accuracy": accepted_acc,
        })

# Confidence distribution
plt.figure(figsize=(10, 5))
plt.hist(confidences[test_results['labels']==0], bins=50, alpha=0.5, label='Non-brain')
plt.hist(confidences[test_results['labels']==1], bins=50, alpha=0.5, label='Brain')
plt.axvline(config.confidence_threshold, color='r', linestyle='--', label='Threshold')
plt.xlabel('Confidence')
plt.ylabel('Count')
plt.title('Confidence Distribution by Class')
plt.legend()
plt.tight_layout()
plt.savefig(config.save_dir / "confidence_distribution.png", dpi=300)

if config.use_wandb:
    wandb.log({"test/confidence_dist": wandb.Image(str(config.save_dir / "confidence_distribution.png"))})

plt.show()

# After getting test results
print("\n" + "="*60)
print("SANITY CHECKS")
print("="*60)

# 1. Check test set distribution
print(f"Test set distribution:")
print(f"  Non-brain (0): {(test_results['labels'] == 0).sum()}")
print(f"  Brain (1):     {(test_results['labels'] == 1).sum()}")

# 2. Check prediction distribution
print(f"\nPrediction distribution:")
print(f"  Predicted non-brain (0): {(test_results['predictions'] == 0).sum()}")
print(f"  Predicted brain (1):     {(test_results['predictions'] == 1).sum()}")

# 3. Check confidence distribution
confidences = np.max(test_results['probabilities'], axis=1)
print(f"\nConfidence statistics:")
print(f"  Mean:   {confidences.mean():.4f}")
print(f"  Median: {np.median(confidences):.4f}")
print(f"  Min:    {confidences.min():.4f}")
print(f"  Q1:     {np.percentile(confidences, 25):.4f}")
print(f"  Q3:     {np.percentile(confidences, 75):.4f}")

# 4. Check if any predictions were uncertain
low_conf = (confidences < 0.95).sum()
print(f"\nLow confidence predictions (<95%): {low_conf} ({low_conf/len(confidences)*100:.1f}%)")

print("="*60)


#  SAVE RESULTS
# ================================================================================

results_dict = {
    'test_accuracy': test_results['accuracy'],
    'test_loss': test_results['loss'],
    'predictions': test_results['predictions'],
    'labels': test_results['labels'],
    'probabilities': test_results['probabilities'],
    'best_epoch': best_epoch,
    'best_val_acc': best_val_acc,
    'training_history': training_history,
    'config': vars(config),
    'classification_report': report,
}

with open(config.save_dir / "results.pkl", 'wb') as f:
    pickle.dump(results_dict, f)

print(f"\n✓ Results saved to {config.save_dir}")

if config.use_wandb:
    # Save model artifact
    artifact = wandb.Artifact('level1_model', type='model')
    artifact.add_file(str(config.save_dir / "best_model.pt"))
    wandb.log_artifact(artifact)
    
    wandb.finish()
    print("✓ WandB run finished")

print("\n" + "="*80)
print("LEVEL 1 COMPLETE")
print("="*80)
print(f"Test Accuracy: {test_results['accuracy']:.3f}")
print(f"Best Epoch: {best_epoch}")
print(f"Model saved: {config.save_dir / 'best_model.pt'}")
if config.use_wandb:
    print(f"WandB dashboard: {wandb.run.get_url()}")
print("="*80)
