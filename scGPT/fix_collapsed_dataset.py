#!/usr/bin/env python
"""
Fix collapsed dataset: remove duplicates, NaN/Inf values
"""

import scanpy as sc
import numpy as np
from scipy.sparse import issparse
import gc

print("="*80)
print("FIXING COLLAPSED DATASET")
print("="*80)

# Load
print("\n1. Loading dataset...")
adata = sc.read_h5ad("./data/Nonneuron_collapsed.h5ad")
print(f"   Original: {adata.shape}")

# Fix gene names
print("\n2. Fixing duplicate gene names...")
n_dup_before = adata.var.index.duplicated().sum()
print(f"   Duplicates: {n_dup_before}")

if n_dup_before > 0:
    # Convert to string first (gene names are categorical)
    adata.var.index = adata.var.index.astype(str)
    adata.var_names_make_unique()
    print("   ✓ Made unique")

# Check for NaN/Inf (sample-based)
print("\n3. Checking for bad values...")
sample_size = min(50000, adata.n_obs)
sample_idx = np.random.choice(adata.n_obs, sample_size, replace=False)

if issparse(adata[sample_idx, :].X):
    X_sample = adata[sample_idx, :].X.toarray()
else:
    X_sample = adata[sample_idx, :].X

has_nan = np.isnan(X_sample).any()
has_inf = np.isinf(X_sample).any()

print(f"   NaN in sample: {has_nan}")
print(f"   Inf in sample: {has_inf}")

if has_nan or has_inf:
    print("\n   ⚠️  Found bad values, cleaning...")
    
    # Find bad cells (check full dataset by chunks)
    bad_cells = []
    chunk_size = 10000
    
    for i in range(0, adata.n_obs, chunk_size):
        end = min(i + chunk_size, adata.n_obs)
        chunk = adata[i:end, :]
        
        if issparse(chunk.X):
            X_chunk = chunk.X.toarray()
        else:
            X_chunk = chunk.X
        
        bad_in_chunk = np.isnan(X_chunk).any(axis=1) | np.isinf(X_chunk).any(axis=1)
        bad_cells.extend(np.where(bad_in_chunk)[0] + i)
        
        if (i // chunk_size) % 10 == 0:
            print(f"   Checked {end:,}/{adata.n_obs:,} cells...")
    
    print(f"\n   Found {len(bad_cells)} bad cells")
    
    if len(bad_cells) > 0:
        # Remove bad cells
        good_cells = np.ones(adata.n_obs, dtype=bool)
        good_cells[bad_cells] = False
        adata = adata[good_cells, :].copy()
        print(f"   ✓ Removed bad cells")
        print(f"   New shape: {adata.shape}")

# Check cell type balance
print("\n4. Checking cell type distribution...")
type_counts = adata.obs['subtype'].value_counts()
print(f"   Total types: {len(type_counts)}")
print(f"   Min count: {type_counts.min():,}")
print(f"   Max count: {type_counts.max():,}")

# Save fixed version
print("\n5. Saving fixed dataset...")
output_path = "./data/Nonneuron_collapsed_FIXED.h5ad"
adata.write_h5ad(output_path)
print(f"   ✓ Saved: {output_path}")

print("\n" + "="*80)
print("✓ DATASET FIXED!")
print("="*80)
print(f"Final shape: {adata.shape}")
print(f"Cell types: {adata.obs['subtype'].nunique()}")
print(f"\nNext steps:")
print(f"1. Update config: dataset_path='./data/Nonneuron_collapsed_FIXED.h5ad'")
print(f"2. Delete old processed files: rm ./save/adata_*_level3_processed.h5ad")
print(f"3. Restart training")
print("="*80)