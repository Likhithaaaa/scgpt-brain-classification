#!/usr/bin/env python
"""
Memory-safe validation for sparse matrices
"""

import scanpy as sc
import numpy as np
from scipy.sparse import issparse

print("="*80)
print("VALIDATING COLLAPSED DATASET (MEMORY-SAFE)")
print("="*80)

# Load collapsed dataset
print("\nLoading Nonneuron_collapsed.h5ad...")
adata = sc.read_h5ad("./data/Nonneuron_collapsed.h5ad")
print(f"Shape: {adata.shape}")
print(f"Subtypes: {adata.obs['subtype'].nunique()}")
print(f"Sparse: {issparse(adata.X)}")

# FIX DUPLICATE GENE NAMES
print("\nChecking gene names...")
n_duplicates = adata.var.index.duplicated().sum()
print(f"  Duplicate gene names: {n_duplicates}")

if n_duplicates > 0:
    print("  ⚠️  FIXING: Making gene names unique...")
    # Convert to string first (gene names are categorical)
    adata.var.index = adata.var.index.astype(str)
    adata.var_names_make_unique()
    print("  ✓ Fixed!")

# Check for bad values (SPARSE-SAFE)
print("\nChecking for bad values (sampling 10k cells)...")
n_sample = min(10000, adata.n_obs)
sample_indices = np.random.choice(adata.n_obs, n_sample, replace=False)
adata_sample = adata[sample_indices, :]

if issparse(adata_sample.X):
    X_sample = adata_sample.X.toarray()
else:
    X_sample = adata_sample.X

has_nan = np.isnan(X_sample).any()
has_inf = np.isinf(X_sample).any()
has_negative = (X_sample < 0).any()

print(f"  NaN values in sample: {has_nan}")
print(f"  Inf values in sample: {has_inf}")
print(f"  Negative values in sample: {has_negative}")

if has_nan:
    nan_fraction = np.isnan(X_sample).sum() / X_sample.size
    print(f"    NaN fraction: {nan_fraction:.2%}")

if has_inf:
    inf_fraction = np.isinf(X_sample).sum() / X_sample.size
    print(f"    Inf fraction: {inf_fraction:.2%}")

# Check metadata
print("\nChecking metadata...")
print(f"  Columns: {list(adata.obs.columns)}")
print(f"  Missing subtype: {adata.obs['subtype'].isna().sum()}")
print(f"  Duplicate cell IDs: {adata.obs.index.duplicated().sum()}")

print("\nSubtype distribution (top 10):")
for subtype, count in adata.obs['subtype'].value_counts().head(10).items():
    print(f"  {subtype:40s}: {count:>8,}")

# Check expression values
print("\nExpression statistics (on sample):")
print(f"  Min: {X_sample.min():.4f}")
print(f"  Max: {X_sample.max():.4f}")
print(f"  Mean: {X_sample.mean():.4f}")
print(f"  Median: {np.median(X_sample):.4f}")

print("="*80)

if n_duplicates > 0:
    print("⚠️  DUPLICATE GENE NAMES - NEEDS FIXING!")
    print("   Run: python fix_collapsed_dataset.py")
elif has_nan or has_inf:
    print("❌ DATASET HAS BAD VALUES - NEEDS CLEANING!")
else:
    print("✅ DATASET LOOKS MOSTLY CLEAN")
    
print("="*80)