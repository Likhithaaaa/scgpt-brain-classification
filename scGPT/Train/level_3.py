# ================================================================================
# LEVEL 3: BRAIN CELL SUBTYPE CLASSIFICATION WITH HIERARCHICAL REJECTION
# ================================================================================
# Fine-tunes scGPT for detailed brain cell subtype classification
# Includes hierarchical confidence-based rejection system
# ================================================================================

import os
import sys
import gc
import time
import json
import copy
import shutil
import pickle
import traceback
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional
from collections import Counter

import pandas as pd
import numpy as np
from scipy.sparse import issparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc
import scvi

import torch
torch.set_default_dtype(torch.float32) 
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    adjusted_rand_score,
    normalized_mutual_info_score,
    precision_recall_fscore_support
)
from sklearn.utils.class_weight import compute_class_weight

import wandb

import scgpt as scg
from scgpt import SubsetsBatchSampler
from scgpt.model import TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.utils import (
    set_seed,
    category_str2int,
    eval_scib_metrics,
)
from scgpt.utils.util import load_pretrained

from pathlib import Path
import gc

sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

# CONFIGURATION - LEVEL 3 SPECIFIC
# ================================================================================

hyperparameter_defaults = dict(
    seed=0,
    
    # LEVEL 3: Only brain cells, detailed subtypes
    dataset_path="./data/Nonneuron_collapsed_FIXED.h5ad",
    dataset_name="FIXED Allen Atlas - Collpased",
    balance_type="auto",  # auto-detect imbalance
    metadata_path=None,   #"./data/nonneuron_metadata.csv",
    gene_list_path=None,   #"./data/nonneuron_gene_names.csv",
    
    do_train=True,
    load_model="save/scGPT_brain/best_model.pt",  # Start from Level 2
    mask_ratio=0.0,
    epochs=8,  # More epochs for fine-grained classification with early stopping
    n_bins=51,
    MVC=False,
    ecs_thres=0.0,
    dab_weight=0.0,
    
    # Learning parameters
    lr=1e-4,
    batch_size=3, #changed from 8
    gradient_accumulation_steps=16,  #changed from 16 # Effective batch = 8 * 16 = 128
    layer_size=128,
    nlayers=4,
    nhead=4,
    dropout=0.2,
    schedule_ratio=0.9,
    save_eval_interval=2,
    fast_transformer=False,
    use_fast_transformer=True,      
    #fast_transformer_backend="flash",
    pre_norm=False,
    amp=True,
    use_wandb=True, 
    
    include_zero_gene=False,
    freeze=False,
    DSBN=False,
    
    # Hierarchical rejection thresholds
    l3_confidence_threshold=0.70,  # Reject to Level 2 if below
    l2_confidence_threshold=0.65,  # Reject to Level 1 if below
    l1_confidence_threshold=0.50,  # Reject to unknown if below
)

run = wandb.init(
    config=hyperparameter_defaults,
    project="scGPT_brain_l3",
    reinit=True,
)
config = wandb.config
print(config)

set_seed(config.seed)

# GPU SETUP + MEMORY MONITORING
# Memory Monitor Class
# ============================================================================

class MemoryMonitor:
    """Track and report GPU memory usage"""

    def __init__(self, device):
        self.device = device
        self.peak_memory = 0

    def reset(self):
        """Reset peak memory stats"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            self.peak_memory = 0

    def log_memory(self, stage: str = ""):
        """Log current and peak memory usage"""
        if not torch.cuda.is_available():
            return

        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        reserved = torch.cuda.memory_reserved(self.device) / 1024**3
        peak = torch.cuda.max_memory_allocated(self.device) / 1024**3

        self.peak_memory = max(self.peak_memory, peak)

        print(f"[{stage}] GPU Memory: {allocated:.2f}GB allocated | "
              f"{reserved:.2f}GB reserved | {peak:.2f}GB peak")

    def clear_cache(self):
        """Clear CUDA cache and run garbage collection"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ============================================================================
# GPU Setup
# ============================================================================

gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode != 0:
        print("WARNING: Could not get GPU info from nvidia-smi")
except:
    print("WARNING: nvidia-smi not available")

print("GPU Setup:")
print("="*80)
print(f"    Device: {torch.cuda.get_device_name()}")
print(f"    PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB total")
torch.cuda.set_per_process_memory_fraction(0.75, device=0)
print(f"    Memory optimization: 75% limit")
print(f"    Gradient accumulation: ENABLED")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize memory monitor
memory_monitor = MemoryMonitor(device)
memory_monitor.reset()

print("[OK] Memory monitoring initialized")

# LEVEL 3 SPECIFIC: BRAIN CELL SUBTYPE DEFINITIONS
# ================================================================================

# Define brain cell type hierarchies
BRAIN_CELL_HIERARCHY = {
    # Neurons by location and function
    'Neuron': {
        'Excitatory': ['L2/3_IT', 'L4_IT', 'L5_IT', 'L5_PT', 'L6_IT', 'L6_CT', 'L6b'],
        'Inhibitory': ['Pvalb', 'Sst', 'Vip', 'Lamp5', 'Sncg'],
        'Other': ['CR', 'Meis2']
    },
    
    # Glial cells
    'Astrocyte': {
        'Protoplasmic': ['Astro_PP'],
        'Fibrous': ['Astro_FB'],
        'Other': ['Astro']
    },
    
    'Oligodendrocyte': {
        'Mature': ['Oligo_mature'],
        'Precursor': ['OPC'],
        'Myelinating': ['Oligo_myelin']
    },
    
    # Immune cells
    'Microglia': {
        'Homeostatic': ['Micro_home'],
        'Activated': ['Micro_act'],
        'Other': ['Micro']
    },
    
    # Vascular cells
    'Endothelial': {
        'Arterial': ['Endo_art'],
        'Venous': ['Endo_ven'],
        'Capillary': ['Endo_cap']
    },
    
    'Pericyte': {
        'VLMC': ['Peri_VLMC'],
        'Other': ['Peri']
    },
    
    # Other
    'SMC': ['SMC'],
    'Macrophage': ['Macro_BAM', 'Macro_PVM'],
}

def get_level2_from_level3(subtype: str) -> str:
    """Map Level 3 subtype back to Level 2 cell type"""
    for l2_type, subtypes_dict in BRAIN_CELL_HIERARCHY.items():
        if isinstance(subtypes_dict, dict):
            for category, subtype_list in subtypes_dict.items():
                if subtype in subtype_list:
                    return l2_type
        elif subtype in subtypes_dict:
            return l2_type
    return 'Unknown'

def get_level1_from_level2(celltype: str) -> str:
    """Map Level 2 to Level 1 (Brain vs Non-Brain)"""
    # All Level 3 cells are brain cells
    return 'Brain'


# UNIVERSAL DATA LOADING (Works for Allen Atlas, Uniklinikum, and any h5ad)
# ================================================================================

print("\n" + "="*80)
print("STEP 1: Universal Data Loading")
print("="*80)

def load_dataset_universal(dataset_path, metadata_path=None, gene_list_path=None):
    """
    Universal dataset loader - works for any .h5ad file
    Handles: Allen Atlas, Uniklinikum, custom datasets
    
    Args:
        dataset_path: Path to .h5ad file
        metadata_path: Optional path to metadata CSV
        gene_list_path: Optional path to gene list for filtering
    
    Returns:
        adata_full: Loaded AnnData object
        ori_batch_col: Name of original batch column
        data_is_raw: Whether data is raw counts
    """
    
    print(f" Loading dataset from: {dataset_path}")
    
    # Load the h5ad file
    adata_full = sc.read_h5ad(dataset_path)
    print(f"✓ Loaded raw data: {adata_full.shape}")
    print(f"  Observations: {adata_full.n_obs:,} cells")
    print(f"  Variables: {adata_full.n_vars:,} genes")
    
    # ============================================================================
    # STEP 1: Handle gene filtering (if provided)
    # ============================================================================
    if gene_list_path and os.path.exists(gene_list_path):
        print(f"\n🧬 Applying gene filtering from: {gene_list_path}")
        gene_list = pd.read_csv(gene_list_path, header=None)[0].tolist()
        genes_before = adata_full.n_vars
        adata_full = adata_full[:, adata_full.var_names.isin(gene_list)].copy()
        print(f"✓ Gene filtering: {genes_before:,} → {adata_full.n_vars:,} genes")
    
    # ============================================================================
    # STEP 2: Handle metadata (if provided)
    # ============================================================================
    if metadata_path and os.path.exists(metadata_path):
        print(f"\n Loading metadata from: {metadata_path}")
        meta = pd.read_csv(metadata_path)
        print(f"✓ Metadata loaded: {meta.shape}")
        print(f"  Metadata columns: {list(meta.columns)}")
        
        # Detect metadata format and merge
        if "barcode" in meta.columns:
            # UKF-style or barcode-based metadata
            print("  Detected barcode-based metadata")
            
            # FIXED: Safe string conversion for barcodes
            meta["barcode"] = meta["barcode"].astype(str).str.strip()
            adata_full.obs_names = adata_full.obs_names.astype(str).str.strip()
            
            # Check for overlapping columns
            overlap_cols = set(adata_full.obs.columns).intersection(set(meta.columns))
            overlap_cols.discard("barcode")  # Barcode is used as index, so it's OK
            
            if overlap_cols:
                print(f"   Found {len(overlap_cols)} overlapping columns: {list(overlap_cols)[:5]}...")
                print(f"  Merging with suffixes to avoid conflicts")
                
                # Merge with suffixes to handle overlaps
                adata_full.obs = adata_full.obs.merge(
                    meta.set_index("barcode"),
                    left_index=True,
                    right_index=True,
                    how="left",
                    suffixes=("_original", "_metadata")
                )
            else:
                # No overlaps - clean merge
                print("  No overlapping columns - clean merge")
                adata_full.obs = adata_full.obs.join(
                    meta.set_index("barcode"),
                    how="left"
                )
            
            print(f"✓ Metadata merged: {adata_full.obs.shape[1]} total columns")
            
        else:
            print("   Metadata format not recognized (no barcode column), skipping merge...")
    
    # ============================================================================
    # STEP 3: Standardize cell type column name
    # ============================================================================
    print("\n Standardizing cell type annotations...")
    
    # Try to find cell type column (many possible names)
    celltype_candidates = [
        "celltype", "cell_type", "CellType", "CellTypes",  # Standard names
        "cluster", "clusters", "Cluster",                   # Clustering results
        "labels", "label", "annotation", "annotations",     # General labels
        "class_label", "subclass_label",                    # Allen Atlas
        "cell_ontology_class", "free_annotation",           # Other formats
    ]
    
    celltype_col_found = None
    for col in celltype_candidates:
        if col in adata_full.obs.columns:
            celltype_col_found = col
            print(f"✓ Found cell type column: '{col}'")
            break
    
    if celltype_col_found is None:
        # Last resort: check if there's ANY categorical column
        cat_cols = adata_full.obs.select_dtypes(include=['category', 'object']).columns
        if len(cat_cols) > 0:
            print(f" No standard cell type column found. Available columns: {list(cat_cols)}")
            print(f"  Using first categorical column: '{cat_cols[0]}'")
            celltype_col_found = cat_cols[0]
        else:
            raise ValueError(
                f" No cell type column found!\n"
                f"   Available columns: {list(adata_full.obs.columns)}\n"
                f"   Please specify cell type column in metadata or h5ad file."
            )
    
    # Standardize to 'celltype' column
    adata_full.obs["celltype"] = adata_full.obs[celltype_col_found].astype("category")
    
    # Count cell types
    celltype_counts = adata_full.obs["celltype"].value_counts()
    print(f"\n Cell type distribution:")
    print(f"  Total cell types: {len(celltype_counts)}")
    print(f"  Cells per type (top 5):")
    for ct, count in celltype_counts.items(): #celltype_counts.head(5).items():
        print(f"    - {ct}: {count:,} cells")
    
    # ============================================================================
    # STEP 4: Handle batch information
    # ============================================================================
    print("\n Handling batch information...")
    
    # Try to find batch column
    batch_candidates = [
        "batch", "batch_id", "str_batch",           # Standard batch names
        "sample", "sample_id", "donor",             # Sample-level batches
        "dataset", "study", "dataset_id",           # Dataset-level batches
    ]
    
    batch_col_found = None
    for col in batch_candidates:
        if col in adata_full.obs.columns:
            batch_col_found = col
            print(f"✓ Found batch column: '{col}'")
            break
    
    if batch_col_found is None:
        print(" No batch column found - creating single batch")
        adata_full.obs["str_batch"] = "0"
        ori_batch_col = "str_batch"
    else:
        # Standardize batch column - FIXED: handle non-string types
        print(f"  Converting '{batch_col_found}' to string format...")
        
        # Force convert to string, handling any data type
        batch_values = adata_full.obs[batch_col_found]
        
        # Check if it's already string-like
        if pd.api.types.is_string_dtype(batch_values):
            adata_full.obs["str_batch"] = batch_values.astype(str)
        elif pd.api.types.is_categorical_dtype(batch_values):
            # Categorical - convert categories to string
            adata_full.obs["str_batch"] = batch_values.astype(str)
        else:
            # Numeric or other - direct string conversion
            adata_full.obs["str_batch"] = batch_values.astype(str)
        
        ori_batch_col = "str_batch"
        
        n_batches = adata_full.obs["str_batch"].nunique()
        print(f"  ✓ Number of batches: {n_batches}")
        
        # Show batch distribution
        if n_batches <= 10:
            batch_counts = adata_full.obs["str_batch"].value_counts()
            print(f"  Batch distribution:")
            for batch, count in batch_counts.items():
                print(f"    - Batch '{batch}': {count:,} cells")
    
    # Create numeric batch IDs
    adata_full.obs["batch_id"] = adata_full.obs["str_batch"].astype("category").cat.codes.values
    
    # ============================================================================
    # STEP 5: Detect if data is raw counts
    # ============================================================================
    print("\n🔍 Detecting data type...")
    
    # Check if data is raw counts (integers) or normalized
    if issparse(adata_full.X):
        sample_data = adata_full.X[:100, :100].toarray()
    else:
        sample_data = adata_full.X[:100, :100]
    
    # Check if values are integers
    is_integer = np.allclose(sample_data, np.round(sample_data))
    max_val = sample_data.max()
    
    if is_integer and max_val > 10:
        data_is_raw = True
        print("✓ Data type: Raw counts (will apply log1p)")
    else:
        data_is_raw = False
        print("✓ Data type: Normalized/transformed (no log1p needed)")
    
    # ============================================================================
    # STEP 6: Final standardization
    # ============================================================================
    adata_full.var["gene_name"] = adata_full.var.index.tolist()
    
    print("\n" + "="*80)
    print(" FINAL DATASET SUMMARY")
    print("="*80)
    print(f"  Shape: {adata_full.shape}")
    print(f"  Cells: {adata_full.n_obs:,}")
    print(f"  Genes: {adata_full.n_vars:,}")
    print(f"  Cell types: {adata_full.obs['celltype'].nunique()}")
    print(f"  Batches: {adata_full.obs['str_batch'].nunique()}")
    print(f"  Data type: {'Raw counts' if data_is_raw else 'Normalized'}")
    print("="*80)
    
    return adata_full, ori_batch_col, data_is_raw


# ============================================================================
# LOAD DATA USING UNIVERSAL LOADER
# ============================================================================

dataset_path = config.dataset_path
balance_type = config.balance_type
metadata_path = getattr(config, 'metadata_path', None)
gene_list_path = getattr(config, 'gene_list_path', None)

adata_full, ori_batch_col, data_is_raw = load_dataset_universal(
    dataset_path,
    metadata_path=metadata_path,
    gene_list_path=gene_list_path
)

# DATASET SPLITTER CLASS
# ================================================================================

class DatasetSplitter:
    """
    Simple splitter - just specify balanced, imbalanced, or auto-detect
    """
    
    def __init__(self, balance_type="auto"):
        """
        balance_type:
        - "balanced" → random split
        - "imbalanced" → stratified split  
        - "auto" → auto-detect based on data characteristics
        """
        self.balance_type = balance_type
    
    def analyze_and_split(self, adata_full, test_size=0.15, val_size=0.15, random_state=42):
        """Analyze dataset balance and create appropriate split"""
        
        print(f"="*60)
        print(f"DATASET ANALYSIS AND SPLITTING")
        print(f"="*60)
        
        # Get cell type distribution
        celltype_counts = adata_full.obs["celltype"].value_counts()
        total_samples = len(adata_full)
        
        # Calculate balance metrics
        max_class_size = celltype_counts.max()
        min_class_size = celltype_counts.min()
        imbalance_ratio = max_class_size / min_class_size
        min_class_percentage = min_class_size / total_samples * 100
        
        print(f"\nDataset Statistics:")
        print(f"  Total samples: {total_samples:,}")
        print(f"  Cell types: {len(celltype_counts)}")
        print(f"  Largest class: {max_class_size:,} samples")
        print(f"  Smallest class: {min_class_size:,} samples ({min_class_percentage:.2f}%)")
        print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")
        print(f"-"*60)
        
        # Determine split strategy
        if self.balance_type == "auto":
            # Auto-detect based on data characteristics
            use_stratify = imbalance_ratio > 3.0 or min_class_percentage < 2.0
            detected_type = "imbalanced" if use_stratify else "balanced"
            print(f" Auto-detected as: {detected_type}")
        elif self.balance_type == "imbalanced":
            use_stratify = True
            print(f"  User-specified: stratified split")
        else:  # "balanced"
            use_stratify = False
            print(f"  User-specified: random split")
        
        # Check if stratification is possible
        if use_stratify:
            # Stratification variable
            stratify_var = adata_full.obs["celltype"]
            
            # Check minimum requirements for stratification
            min_test_samples = int(min_class_size * test_size)
            min_val_samples = int(min_class_size * val_size)
            
            if min_test_samples < 1 or min_val_samples < 1:
                print(f" WARNING: Smallest class too small for stratification!")
                print(f"  → Falling back to RANDOM split...")
                use_stratify = False
                stratify_var = None
        else:
            stratify_var = None
        
        # Split 1: train+val vs test
        train_val_idx, test_idx = train_test_split(
            range(len(adata_full)),
            test_size=test_size,
            stratify=stratify_var,
            random_state=random_state,
            shuffle=True
        )
        
        # Split 2: train vs val
        val_relative_size = val_size / (1 - test_size)
        
        if use_stratify:
            stratify_train_val = adata_full.obs.iloc[train_val_idx]["celltype"]
        else:
            stratify_train_val = None
        
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_relative_size,
            stratify=stratify_train_val,
            random_state=random_state,
            shuffle=True
        )
        
        # Create splits
        adata_train = adata_full[train_idx].copy()
        adata_val = adata_full[val_idx].copy()
        adata_test = adata_full[test_idx].copy()
        
        # Verify splits
        print(f"\n✓ SPLIT RESULTS:")
        print(f"  Train: {len(adata_train):,} samples ({len(adata_train)/len(adata_full)*100:.1f}%)")
        print(f"  Val:   {len(adata_val):,} samples ({len(adata_val)/len(adata_full)*100:.1f}%)")
        print(f"  Test:  {len(adata_test):,} samples ({len(adata_test)/len(adata_full)*100:.1f}%)")
        
        # Check if stratification is possible
        if use_stratify:
            print(f"\n✓ Split completed using {'STRATIFIED' if stratify_var is not None else 'RANDOM'} strategy")
        
        return adata_train, adata_val, adata_test, use_stratify

#  DATASET SPLITTING WITH STRATIFICATION
# ================================================================================

print("\n" + "="*80)
print("STEP 2: Dataset Splitting (Level 3 Stratified)")
print("="*80)

splitter = DatasetSplitter(balance_type=balance_type)
adata_train_raw, adata_val_raw, adata_test_raw, used_stratify = splitter.analyze_and_split(
    adata_full,
    test_size=0.15,
    val_size=0.15,
    random_state=42
)

print(f" Final result: Used {'STRATIFIED' if used_stratify else 'RANDOM'} split")

# Create consistent subtype encoding for all splits
subtype_categories = adata_full.obs["subtype"].astype("category").cat.categories
global_subtype_categories = subtype_categories
num_types = len(subtype_categories)
id2type = dict(enumerate(subtype_categories))

print(f" Ready for training with {num_types} subtypes")

# HVG SELECTION & SAVE PROCESSED SPLITS
# ============================================================================

print("\n" + "="*80)
print("HVG SELECTION & SAVING PROCESSED SPLITS")
print("="*80)

# ============================================================================
# STEP 1: SELECT HIGHLY VARIABLE GENES
# ============================================================================

print("\nSelecting highly variable genes...")

# Use 1500 HVGs for Level 3 (more genes for fine-grained classification)
n_hvg = 1500

# Compute HVG on training set
print(f"Computing top {n_hvg} highly variable genes on training set...")
sc.pp.highly_variable_genes(
    adata_train_raw,
    n_top_genes=n_hvg,
    subset=False,
    flavor='seurat_v3',
    batch_key=None  # No batch correction for now
)

print(f"✓ Identified {adata_train_raw.var['highly_variable'].sum()} HVGs")

# ============================================================================
# STEP 2: SUBSET ALL SPLITS TO HVG
# ============================================================================

print("\nSubsetting all splits to HVG...")

# Get HVG genes - handle duplicates
hvg_mask = adata_train_raw.var['highly_variable'].values
hvg_genes = adata_train_raw.var_names[hvg_mask]

# Check for duplicate gene names
if not hvg_genes.is_unique:
    print(f"   Found duplicate gene names, making unique...")
    # Convert to string (gene names might be categorical)
    adata_train_raw.var.index = adata_train_raw.var.index.astype(str)
    adata_val_raw.var.index = adata_val_raw.var.index.astype(str)
    adata_test_raw.var.index = adata_test_raw.var.index.astype(str)
    # Now make unique
    adata_train_raw.var_names_make_unique()
    adata_val_raw.var_names_make_unique()
    adata_test_raw.var_names_make_unique()
    hvg_genes = adata_train_raw.var_names[hvg_mask]

print(f"  Selected {len(hvg_genes)} genes")

# Subset to HVG (use boolean mask - more memory efficient)
print("  Subsetting train...")
adata_train_hvg = adata_train_raw[:, hvg_mask].copy()

print("  Subsetting val...")
adata_val_hvg = adata_val_raw[:, hvg_mask].copy()

print("  Subsetting test...")
adata_test_hvg = adata_test_raw[:, hvg_mask].copy()

print(f"✓ All splits now have {adata_train_hvg.n_vars} genes")

# Clean up immediately
del adata_train_raw, adata_val_raw, adata_test_raw, hvg_mask
gc.collect()

# ============================================================================
# STEP 3: ENSURE CATEGORICAL CONSISTENCY
# ============================================================================

print("\nEnsuring categorical consistency across splits...")

# Get all unique subtypes from training
all_subtypes = sorted(adata_train_hvg.obs['subtype'].unique())
print(f"  Total subtypes: {len(all_subtypes)}")

# Set categories on all splits
for split_name, split_data in [('train', adata_train_hvg), ('val', adata_val_hvg), ('test', adata_test_hvg)]:
    # Ensure subtype is categorical with same categories
    if not isinstance(split_data.obs['subtype'].dtype, pd.CategoricalDtype):
        split_data.obs['subtype'] = split_data.obs['subtype'].astype('category')
    
    split_data.obs['subtype'] = split_data.obs['subtype'].cat.set_categories(all_subtypes)
    split_data.obs['subtype_id'] = split_data.obs['subtype'].cat.codes
    
    print(f"  ✓ {split_name}: categories updated")

# ============================================================================
# STEP 4: SAVE PROCESSED SPLITS
# ============================================================================

print("\nSaving processed splits...")

save_dir = Path("./save")
save_dir.mkdir(parents=True, exist_ok=True)

# Save each split
adata_train_hvg.write_h5ad(save_dir / "adata_train_level3_processed.h5ad")
print(f"  ✓ Saved: {save_dir / 'adata_train_level3_processed.h5ad'}")

adata_val_hvg.write_h5ad(save_dir / "adata_val_level3_processed.h5ad")
print(f"  ✓ Saved: {save_dir / 'adata_val_level3_processed.h5ad'}")

adata_test_hvg.write_h5ad(save_dir / "adata_test_level3_processed.h5ad")
print(f"  ✓ Saved: {save_dir / 'adata_test_level3_processed.h5ad'}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✓ PROCESSED FILES SAVED!")
print("="*80)
print(f"  Train: {adata_train_hvg.n_obs:,} cells × {adata_train_hvg.n_vars} genes")
print(f"  Val:   {adata_val_hvg.n_obs:,} cells × {adata_val_hvg.n_vars} genes")
print(f"  Test:  {adata_test_hvg.n_obs:,} cells × {adata_test_hvg.n_vars} genes")
print(f"  Classes: {len(all_subtypes)}")
print("\n✓ Ready for 150k sampling in next cell!")
print("="*80)

# Clean up - but keep the _hvg variables for next cell if needed
gc.collect()

n_hvg = 1500  # number of highly variable genes
max_seq_len = n_hvg + 1
per_seq_batch_sample = False
dataset_name = config.dataset_name

# PREPROCESSING CELL - 150K SAMPLING (NO FILTERING)
#  Sample 150k cells directly, skip rare class filtering entirely
# ============================================================================

#  DEFINE PATHS

save_dir = Path("./save")
save_dir.mkdir(parents=True, exist_ok=True)

processed_train_path = save_dir / "adata_train_level3_processed.h5ad"
processed_val_path = save_dir / "adata_val_level3_processed.h5ad"
processed_test_path = save_dir / "adata_test_level3_processed.h5ad"

print(f"Using processed files:")
print(f"  Train: {processed_train_path}")
print(f"  Val:   {processed_val_path}")
print(f"  Test:  {processed_test_path}")

# ============================================================================
# LOAD AND SAMPLE TRAIN (150K CELLS)
# ============================================================================

print("\n" + "="*80)
print("STEP 1: LOADING & SAMPLING TRAIN (150K)")
print("="*80)

print("\n[1/3] Loading train dataset...")
adata_train_full = sc.read_h5ad(processed_train_path)
print(f"  Full train: {adata_train_full.shape}")

# Sample 150k cells
np.random.seed(42)
n_sample = min(150000, adata_train_full.n_obs)  # Don't exceed available cells
sample_indices = np.random.choice(adata_train_full.n_obs, n_sample, replace=False)
sample_indices = np.sort(sample_indices)

print(f"\n  Sampling {n_sample:,} cells...")
adata_train = adata_train_full[sample_indices].copy()
'''
# Use full dataset - NO COPY NEEDED
n_sample = adata_train_full.n_obs

print(f"Using full dataset: {n_sample:,} cells")

# Just rename - no memory duplication!
adata_train = adata_train_full
'''
#####################################
# Delete old reference (but data still exists as adata_train)
del adata_train_full
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"  ✓ Training: {adata_train.n_obs:,} cells")
print(f"  ✓ Genes: {adata_train.n_vars}")

# ============================================================================
# UPDATE CATEGORIES FROM SAMPLED DATA (NO FILTERING)
# ============================================================================

print("\n" + "="*80)
print("STEP 2: UPDATE CATEGORIES (NO FILTERING)")
print("="*80)

# Get all unique cell types present in sampled data
unique_subtypes = sorted(adata_train.obs['subtype'].unique())
num_types = len(unique_subtypes)

print(f"  Cell types in sample: {num_types}")
print(f"  ✓ Keeping all naturally sampled classes")

# Update global categories
global_subtype_categories = unique_subtypes
id2type = dict(enumerate(global_subtype_categories))

# Update categorical encoding
adata_train.obs['subtype'] = adata_train.obs['subtype'].cat.set_categories(global_subtype_categories)
adata_train.obs['subtype_id'] = adata_train.obs['subtype'].cat.codes.values

# Check class distribution
class_counts = adata_train.obs['subtype'].value_counts()
print(f"\n  Per-class stats:")
print(f"    Min samples: {class_counts.min()}")
print(f"    Max samples: {class_counts.max()}")
print(f"    Avg samples: {class_counts.mean():.0f}")
print(f"    Median samples: {class_counts.median():.0f}")
print(f"    Imbalance ratio: {class_counts.max()/class_counts.min():.1f}:1")

# Store HVG count
n_hvg_level3 = adata_train.n_vars

gc.collect()

# ============================================================================
# LOAD VAL (MEMORY-SAFE)
# ============================================================================

print("\n" + "="*80)
print("STEP 3: LOADING VAL")
print("="*80)

print("\n[2/3] Loading validation...")
adata_val = sc.read_h5ad(processed_val_path)
print(f"  Val loaded: {adata_val.shape}")

# Filter to only include cell types present in training
mask = adata_val.obs['subtype'].isin(global_subtype_categories)
adata_val = adata_val[mask].copy()
del mask

# Update categories
adata_val.obs['subtype'] = adata_val.obs['subtype'].cat.set_categories(global_subtype_categories)
adata_val.obs['subtype_id'] = adata_val.obs['subtype'].cat.codes.values

print(f"  ✓ Filtered val: {adata_val.n_obs:,} cells")
print(f"  ✓ Cell types: {adata_val.obs['subtype'].nunique()}")

gc.collect()

# ============================================================================
# LOAD TEST (MEMORY-SAFE)
# ============================================================================

print("\n" + "="*80)
print("STEP 4: LOADING TEST")
print("="*80)

print("\n[3/3] Loading test...")
adata_test = sc.read_h5ad(processed_test_path)
print(f"  Test loaded: {adata_test.shape}")

# Filter to only include cell types present in training
mask = adata_test.obs['subtype'].isin(global_subtype_categories)
adata_test = adata_test[mask].copy()
del mask

# Update categories
adata_test.obs['subtype'] = adata_test.obs['subtype'].cat.set_categories(global_subtype_categories)
adata_test.obs['subtype_id'] = adata_test.obs['subtype'].cat.codes.values

print(f"  ✓ Filtered test: {adata_test.n_obs:,} cells")
print(f"  ✓ Cell types: {adata_test.obs['subtype'].nunique()}")

gc.collect()

# ============================================================================
# UPDATE GLOBAL VARIABLES
# ============================================================================

adata = adata_train
subtypes = global_subtype_categories
celltypes = global_subtype_categories

print("\n" + "="*80)
print("✓ DATA READY FOR TRAINING")
print("="*80)
print(f"  Training:   {adata_train.n_obs:,} samples")
print(f"  Validation: {adata_val.n_obs:,} samples")
print(f"  Test:       {adata_test.n_obs:,} samples")
print(f"  Classes:    {num_types}")
print(f"  Genes:      {n_hvg_level3}")
print("="*80)

# ============================================================================
# FINAL MEMORY CLEANUP
# ============================================================================

print("\n  Final memory cleanup...")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("✓ Ready for training!")

# After preprocessing, delete the full adata object to save memory
if 'adata_full' in locals():
    del adata_full
    print("Deleted adata_full")
                                                                                                                     
# Force garbage collection
gc.collect()
                                                                                                                     
# Log memory                                                                                                         
if 'memory_monitor' in locals():
    memory_monitor.log_memory("After preprocessing cleanup")
    memory_monitor.clear_cache()

# VOCABULARY AND GENE FILTERING
# ================================================================================

print("\n" + "="*80)
print("STEP 4: Building vocabulary and filtering genes")
print("="*80)

# Load vocabulary
vocab_file = Path("./save/scGPT_brain/vocab.json")

if not vocab_file.exists():
    raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")

vocab = scg.tokenizer.GeneVocab.from_file(vocab_file)
print(f"Loaded vocabulary from: {vocab_file}")
print(f"Initial vocabulary size: {len(vocab)}")

# Add special tokens
special_tokens = ["<pad>", "<cls>", "<eoc>"]
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)

vocab.set_default_index(vocab["<pad>"])
print(f"Added special tokens: {special_tokens}")
print(f"Final vocabulary size: {len(vocab)}")

# Check gene overlap with vocabulary (using var_names directly)
print("\nChecking gene overlap with vocabulary...")

adata_train.var["id_in_vocab"] = [
    1 if gene in vocab else -1 for gene in adata_train.var_names
]
gene_ids_in_vocab = np.array(adata_train.var["id_in_vocab"])

print(f"Matched {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
      f"in vocabulary of size {len(vocab)}")

# Filter to genes in vocabulary
genes_in_vocab = adata_train.var["id_in_vocab"] >= 0
print(f"Keeping {np.sum(genes_in_vocab)} genes that are in vocabulary")

adata_train = adata_train[:, genes_in_vocab]
adata_val = adata_val[:, genes_in_vocab]
adata_test = adata_test[:, genes_in_vocab]

print(f"\nFiltered dataset shapes:")
print(f"  Train: {adata_train.shape}")
print(f"  Val:   {adata_val.shape}")
print(f"  Test:  {adata_test.shape}")

# Create gene IDs array (use var_names directly)
gene_ids = np.array([vocab[gene] for gene in adata_train.var_names], dtype=int)
print(f"\nGene IDs created: {len(gene_ids)} genes")
print(f"Sample gene IDs: {gene_ids[:5]}")
print(f"Sample gene names: {list(adata_train.var_names[:5])}")

# Store genes list
genes = adata.var["gene_name"].tolist()

print(f"\nFinal data summary:")
print(f"Training: {adata_train.shape} samples")
print(f"Validation: {adata_val.shape} samples")
print(f"Test: {adata_test.shape} samples")
print(f"Features: {adata_train.shape[1]} genes (HVGs)")
print(f"Cell types: {num_types}")

# DATASET SUMMARY
# ================================================================================

print("\n" + "="*80)
print("DATASET SUMMARY")
print("="*80)

# Batch info
num_batch_types = len(adata_train.obs["batch_id"].unique())
print(f"Batch types: {num_batch_types}")

# Subtype info
print(f"Subtypes: {num_types}")

# Gene info
print(f"Genes (HVGs): {adata_train.n_vars}")

# Sample sizes
print(f"\nSample sizes:")
print(f"  Train: {adata_train.n_obs:,} cells")
print(f"  Val:   {adata_val.n_obs:,} cells")
print(f"  Test:  {adata_test.n_obs:,} cells")

# Subtype distribution in training
print(f"\nSubtype distribution (training):")
print(adata_train.obs["subtype"].value_counts().head(10))

print("="*80)

#  CLASS WEIGHTS
# ================================================================================

print("\n" + "="*80)
print("CLASS WEIGHTING FOR IMBALANCED DATA - LEVEL 3")
print("="*80)

# Calculate class weights from training data
train_labels = adata_train.obs['subtype_id'].values
class_weights = compute_class_weight(
    'balanced',
    classes=np.arange(num_types),
    y=train_labels
)

class_weights_tensor = torch.FloatTensor(class_weights).to(device)

print("\n" + "="*80)
print("CLASS WEIGHTS (for imbalanced data):")
print("="*80)
for i, (subtype, weight) in enumerate(zip(subtypes, class_weights)):
    count = (train_labels == i).sum()
    print(f"{subtype:20s}: weight={weight:.3f}, count={count:5d}")
print("="*80 + "\n")

# HIERARCHICAL DEGRADATION 
# ================================================================================

print("\n" + "="*80)
print("HIERARCHICAL SUBTYPE CLASSIFICATION WITH DEGRADATION")
print("="*80)

# ============================================================================
# STEP 1: Extract Parent Cell Types from Subtype Names
# ============================================================================

def extract_parent_type(subtype_name):
    """
    Extract parent cell type from subtype name
    E.g., 'COP_37_3033' → 'COP'
         'Oligo_47_3023' → 'Oligo'
    """
    # Split by underscore and take first part
    parent = subtype_name.split('_')[0]
    return parent

# Create parent type mappings
print("\n1. Creating parent-child hierarchy...")

# Get all subtypes
all_subtypes = global_subtype_categories

# Create mapping: subtype → parent
subtype_to_parent = {}
parent_types_set = set()

for subtype in all_subtypes:
    parent = extract_parent_type(subtype)
    subtype_to_parent[subtype] = parent
    parent_types_set.add(parent)

# Create parent type vocab
parent_types_list = sorted(list(parent_types_set))
parent_to_id = {parent: idx for idx, parent in enumerate(parent_types_list)}

print(f"   Found {len(parent_types_list)} parent cell types:")
for parent in parent_types_list:
    # Count subtypes per parent
    n_subtypes = sum(1 for st in all_subtypes if extract_parent_type(st) == parent)
    print(f"     - {parent}: {n_subtypes} subtypes")

# ============================================================================
# STEP 2: Add Parent Type Information to Datasets
# ============================================================================

print("\n2. Adding parent type labels to all splits...")

for split_name, split_data in [("train", adata_train), ("val", adata_val), ("test", adata_test)]:
    # Get parent type for each cell
    parent_labels = [subtype_to_parent[subtype] for subtype in split_data.obs['subtype']]
    parent_ids = [parent_to_id[parent] for parent in parent_labels]
    
    # Add to obs
    split_data.obs['parent_type'] = parent_labels
    split_data.obs['parent_id'] = parent_ids
    
    print(f"   {split_name}: Added parent type labels")

# Update globals
num_parent_types = len(parent_types_list)
parent_id2type = {idx: parent for parent, idx in parent_to_id.items()}

print(f"\n✓ Hierarchy created:")
print(f"   Level 2 (Parent): {num_parent_types} types")
print(f"   Level 3 (Subtype): {num_types} types")

# ============================================================================
# STEP 3: Hierarchical Class Weights
# ============================================================================

print("\n3. Calculating hierarchical sampling weights...")

train_labels = adata_train.obs['subtype_id'].values
parent_labels = adata_train.obs['parent_id'].values
class_counts = np.bincount(train_labels)

# Show rare classes and their parents
rare_threshold = 50
rare_classes = np.where(class_counts < rare_threshold)[0]

print(f"\n   Rare classes (<{rare_threshold} samples) with parent types:")
for idx in rare_classes[:10]:  # Show first 10
    subtype = id2type[idx]
    parent = subtype_to_parent[subtype]
    count = class_counts[idx]
    print(f"     {subtype:25s} → {parent:10s} ({count:3d} samples)")

# Use sqrt balancing (prevents extreme oversampling)
class_weights_sampling = 1.0 / np.sqrt(class_counts)
class_weights_sampling = class_weights_sampling / class_weights_sampling.sum() * len(class_weights_sampling)

sample_weights = class_weights_sampling[train_labels]

# Create weighted sampler
train_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_labels),
    replacement=True
)

print(f"\n✓ Hierarchical sampler created")
print(f"   All {num_types} subtypes retained (no filtering!)")

# ============================================================================
# STEP 4: Hierarchical Loss Configuration
# ============================================================================

print("\n4. Setting up hierarchical loss...")

# Calculate parent class weights for loss
parent_counts = np.bincount(parent_labels)
parent_weights_loss = 1.0 / parent_counts
parent_weights_loss = torch.FloatTensor(parent_weights_loss / parent_weights_loss.sum() * len(parent_weights_loss))

# Calculate subtype class weights for loss
subtype_weights_loss = 1.0 / class_counts
subtype_weights_loss = torch.FloatTensor(subtype_weights_loss / subtype_weights_loss.sum() * len(subtype_weights_loss))

print(f"   Parent type weights: shape {parent_weights_loss.shape}")
print(f"   Subtype weights: shape {subtype_weights_loss.shape}")

# ============================================================================
# STEP 5: Create Hierarchical Mapping Tensor
# ============================================================================

# Create tensor mapping: subtype_id → parent_id
subtype_to_parent_tensor = torch.zeros(num_types, dtype=torch.long)
for subtype_id in range(num_types):
    subtype_name = id2type[subtype_id]
    parent_name = subtype_to_parent[subtype_name]
    parent_id = parent_to_id[parent_name]
    subtype_to_parent_tensor[subtype_id] = parent_id

print(f"\n✓ Hierarchical mapping tensor created: {subtype_to_parent_tensor.shape}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("HIERARCHICAL CONFIGURATION SUMMARY")
print("="*80)

print(f"\nHierarchy:")
print(f"  Parent types (Level 2): {num_parent_types}")
print(f"  Subtypes (Level 3): {num_types}")

print(f"\nDataset sizes:")
print(f"  Train: {adata_train.n_obs:,} cells")
print(f"  Val:   {adata_val.n_obs:,} cells")
print(f"  Test:  {adata_test.n_obs:,} cells")

print(f"\nRare class handling:")
print(f"  Classes <50 samples: {len(rare_classes)}")
print(f"  Strategy: Hierarchical degradation to parent type")
print(f"  ✅ No data lost!")

print(f"\nImbalance stats:")
print(f"  Subtype level: {class_counts.max() / class_counts.min():.1f}:1")
print(f"  Parent level: {parent_counts.max() / parent_counts.min():.1f}:1")

print("="*80)

# ============================================================================
# Store for training
# ============================================================================

# These will be used in training loop
hierarchy_config = {
    'num_parent_types': num_parent_types,
    'num_subtypes': num_types,
    'parent_to_id': parent_to_id,
    'parent_id2type': parent_id2type,
    'subtype_to_parent': subtype_to_parent,
    'subtype_to_parent_tensor': subtype_to_parent_tensor,
    'parent_weights_loss': parent_weights_loss,
    'subtype_weights_loss': subtype_weights_loss,
}

print("\n✓ Hierarchy configuration saved for training")

# ── SAVE INFERENCE ARTIFACTS ──────────────────────────────────────────────────

INFERENCE_SAVE_DIR = Path("./save/scGPT_level3_subtype")
INFERENCE_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 1. HVG gene list
hvg_genes_l3 = adata_train_hvg.var_names.tolist()
with open(INFERENCE_SAVE_DIR / "hvg_genes.pkl", "wb") as f:
    pickle.dump(hvg_genes_l3, f)
print(f"✓ Saved hvg_genes.pkl ({len(hvg_genes_l3)} genes)")

# 2. Label mapping (int → subtype name)
with open(INFERENCE_SAVE_DIR / "id_to_label.json", "w") as f:
    json.dump(id2type, f, indent=2)
print(f"✓ Saved id_to_label.json ({len(id2type)} subtypes)")

# 3. Taxonomy — built from subtype_to_parent already computed above
parent_to_family = {
    "Astro":      "Glial",
    "Oligo":      "Glial",
    "OPC":        "Glial",
    "COP":        "Glial",
    "Micro":      "Immune",
    "Endo":       "Vascular",
    "Peri":       "Vascular",
    "VLMC":       "Vascular",
    "SMC":        "Vascular",
    "Fibroblast": "Vascular",
    "Ependymal":  "Glial",
    "Choroid":    "Glial",
    "Bergmann":   "Glial",
    "Neuron":     "Neuronal",
}

taxonomy_l3 = {}
for subtype, parent in subtype_to_parent.items():
    family = parent_to_family.get(parent, parent)  # fallback: parent name = family
    taxonomy_l3[subtype] = {"parent": parent, "family": family}

with open(INFERENCE_SAVE_DIR / "taxonomy.json", "w") as f:
    json.dump(taxonomy_l3, f, indent=2)

# Print all parents so you can verify parent_to_family is complete
unmapped = [p for p in set(subtype_to_parent.values()) if p not in parent_to_family]
if unmapped:
    print(f"⚠ Parents with no family mapping (using parent as fallback): {unmapped}")
print(f"✓ Saved taxonomy.json ({len(taxonomy_l3)} subtypes)")
# ─────────────────────────────────────────────────────────────────────────────

# GENE EXPRESSION PREPROCESSING & BINNING (CHUNKED - MEMORY SAFE)
# ============================================================================

print("\n" + "="*80)
print("GENE EXPRESSION PREPROCESSING (CHUNKED)")
print("="*80)

# ============================================================================
# CHUNK PROCESSING FUNCTION
# ============================================================================

def preprocess_in_chunks(adata, preprocessor, chunk_size=25000):
    """Process large dataset in chunks to avoid memory issues"""
    n_chunks = int(np.ceil(adata.n_obs / chunk_size))
    print(f"  Processing {n_chunks} chunks of {chunk_size} cells...")
    
    # Initialize X_binned layer
    adata.layers["X_binned"] = np.zeros(adata.shape, dtype=np.int32)
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, adata.n_obs)
        
        # Extract chunk
        chunk = adata[start_idx:end_idx].copy()
        
        # Process chunk
        preprocessor(chunk, batch_key=None)
        
        # Copy binned data back
        adata.layers["X_binned"][start_idx:end_idx] = chunk.layers["X_binned"]
        
        print(f"    ✓ Chunk {i+1}/{n_chunks} ({start_idx:,}-{end_idx:,})")
        
        # Cleanup
        del chunk
        gc.collect()
    
    return adata

# ============================================================================
# INITIALIZE PREPROCESSOR
# ============================================================================

preprocessor = Preprocessor(
    use_key="X",
    filter_gene_by_counts=False,
    filter_cell_by_counts=False,
    normalize_total=False,
    log1p=data_is_raw,
    subset_hvg=False,
    binning=config.n_bins,
)

print(f"Preprocessor config:")
print(f"  Bins: {config.n_bins}")
print(f"  Log1p: {data_is_raw}")
print(f"  Chunk size: 25,000 cells")

# ============================================================================
# PROCESS TRAINING DATA
# ============================================================================

print("\n1. Processing training data (chunked)...")
print(f"   Input: {adata_train.shape}")

adata_train = preprocess_in_chunks(adata_train, preprocessor, chunk_size=25000)

print(f"   ✓ Complete: X_binned layer created")

# ============================================================================
# PROCESS VALIDATION DATA
# ============================================================================

print("\n2. Processing validation data (chunked)...")
print(f"   Input: {adata_val.shape}")

adata_val = preprocess_in_chunks(adata_val, preprocessor, chunk_size=25000)

print(f"   ✓ Complete")

# ============================================================================
# PROCESS TEST DATA
# ============================================================================

print("\n3. Processing test data (chunked)...")
print(f"   Input: {adata_test.shape}")

adata_test = preprocess_in_chunks(adata_test, preprocessor, chunk_size=25000)

print(f"   ✓ Complete")

# ============================================================================
# UPDATE VARIABLES
# ============================================================================

# Update global for model
adata = adata_train

# Final cleanup
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n" + "="*80)
print("✓ PREPROCESSING COMPLETE")
print("="*80)
print(f"  All splits processed in chunks")
print(f"  'X_binned' layer ready for training")
print(f"  Memory-safe approach used")
print("="*80)

#  HIERARCHICAL LOSS FUNCTION
# ================================================================================

class HierarchicalLoss(nn.Module):
    """
    Combined loss for hierarchical classification:
    - Subtype prediction (fine-grained)
    - Parent type prediction (coarse-grained)
    """
    
    def __init__(self, subtype_weights, parent_weights, subtype_to_parent_tensor, 
                 subtype_weight=0.7, parent_weight=0.3):
        super().__init__()
        self.subtype_weight = subtype_weight
        self.parent_weight = parent_weight
        
        # Loss functions
        self.subtype_criterion = nn.CrossEntropyLoss(weight=subtype_weights)
        self.parent_criterion = nn.CrossEntropyLoss(weight=parent_weights)
        
        # Mapping tensor
        self.register_buffer('subtype_to_parent', subtype_to_parent_tensor)
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch, num_subtypes] - model predictions
            targets: [batch] - ground truth subtype IDs
        """
        # Subtype loss (fine-grained)
        subtype_loss = self.subtype_criterion(logits, targets)
        
        # Convert subtype logits to parent logits by summing probabilities
        probs = F.softmax(logits, dim=1)  # [batch, num_subtypes]
        
        # Create parent logits by summing child probabilities
        num_parents = self.subtype_to_parent.max().item() + 1
        parent_logits = torch.zeros(logits.size(0), num_parents, device=logits.device)
        
        for parent_id in range(num_parents):
            # Sum probabilities of all subtypes belonging to this parent
            child_mask = (self.subtype_to_parent == parent_id)
            parent_logits[:, parent_id] = probs[:, child_mask].sum(dim=1)
        
        # Convert back to logits
        parent_logits = torch.log(parent_logits + 1e-10)
        
        # Get parent targets
        parent_targets = self.subtype_to_parent[targets]
        
        # Parent loss (coarse-grained)
        parent_loss = self.parent_criterion(parent_logits, parent_targets)
        
        # Combined loss
        total_loss = (self.subtype_weight * subtype_loss + 
                     self.parent_weight * parent_loss)
        
        return total_loss, subtype_loss, parent_loss

# Create the loss function
print("\n" + "="*80)
print("CREATING HIERARCHICAL LOSS FUNCTION")
print("="*80)

criterion_cls = HierarchicalLoss(
    subtype_weights=hierarchy_config['subtype_weights_loss'].cuda(),
    parent_weights=hierarchy_config['parent_weights_loss'].cuda(),
    subtype_to_parent_tensor=hierarchy_config['subtype_to_parent_tensor'].cuda(),
    subtype_weight=0.7,  # 70% weight on subtype
    parent_weight=0.3    # 30% weight on parent
)

print("✓ Hierarchical loss function created")
print(f"  Subtype weight: 0.7 (fine-grained)")
print(f"  Parent weight: 0.3 (coarse-grained)")
print(f"  Device: cuda")
print("="*80)


# Load vocabulary
if config.load_model is not None:
    model_file = Path(config.load_model)
    model_dir = model_file.parent
    vocab_file = model_dir / "vocab.json"
    
    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    
    #  Memory-safe: Filter in-place, no copying
    def filter_genes_by_vocab_inplace(adata_split, split_name):
        """Filter genes in a dataset based on vocab - IN PLACE"""
        adata_split.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in adata_split.var["gene_name"]
        ]
        gene_ids_in_vocab = np.array(adata_split.var["id_in_vocab"])
        
        n_matched = np.sum(gene_ids_in_vocab >= 0)
        n_total = len(gene_ids_in_vocab)
        print(f"  {split_name}: matched {n_matched}/{n_total} genes in vocabulary")
        
        genes_in_vocab = adata_split.var["id_in_vocab"] >= 0
        
        # Filter in-place (creates view, not copy)
        return adata_split[:, genes_in_vocab]
    
    print("Filtering genes by vocabulary:")
    adata = filter_genes_by_vocab_inplace(adata, "Train")
    gc.collect()  # Clean up immediately
    
    adata_val = filter_genes_by_vocab_inplace(adata_val, "Val")
    gc.collect()
    
    adata_test = filter_genes_by_vocab_inplace(adata_test, "Test")
    gc.collect()
    
    print(f"\nFinal shapes after vocab filtering:")
    print(f"  Train: {adata.shape}")
    print(f"  Val:   {adata_val.shape}")
    print(f"  Test:  {adata_test.shape}")
    
else:
    vocab = Vocab(
        VocabPybind(adata.var["gene_name"].tolist() + special_tokens, None)
    )

vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(vocab(adata.var["gene_name"].tolist()), dtype=int)                                     
                                                                                                                     
print(f"Vocabulary size: {len(vocab)}")                                                                              
print(f"Training samples: {adata.n_obs:,}")                                                                          
print(f"Validation samples: {adata_val.n_obs:,}")                                                                    
print(f"Test samples: {adata_test.n_obs:,}")                                                                         

# SPECIAL TOKENS AND TOKENIZATION CONFIG
# ================================================================================

pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]  # special tokens used in tokenizer

# Masking and padding configuration
mask_ratio = config.mask_ratio
mask_value = "auto"  
pad_value = -2 

include_zero_gene = config.include_zero_gene  # if True, include zero genes among HVGs
max_seq_len = n_hvg + 1

# input/output representation
input_style = "binned"  # "normed_raw", "log1p", or "binned"
output_style = "binned"  # "normed_raw", "log1p", or "binned"

print(f"Tokenization config:")
print(f"  Pad token: {pad_token}")
print(f"  Pad value: {pad_value}")
print(f"  Mask value: {mask_value}")
print(f"  Mask ratio: {mask_ratio}")
print(f"  Max seq length: {max_seq_len}")

# SETTINGS FOR TRAINING
# ================================================================================

MLM = False  # whether to use masked language modeling
CLS = True  # celltype classification objective
ADV = False  # Adversarial training for batch correction
CCE = False  # Contrastive cell embedding objective
MVC = config.MVC  # Masked value prediction for cell embedding
ECS = config.ecs_thres > 0  # Elastic cell similarity objective
DAB = False  # Domain adaptation by reverse backpropagation
INPUT_BATCH_LABELS = False  
input_emb_style = "continuous"  # "category" or "continuous" or "scaling"
cell_emb_style = "cls"  # "avg-pool" or "w-pool" or "cls"
adv_E_delay_epochs = 0  # delay adversarial training on encoder for a few epochs
adv_D_delay_epochs = 0
mvc_decoder_style = "inner product"
fast_transformer = config["use_fast_transformer"]
#fast_transformer_backend = config["fast_transformer_backend"]
ecs_threshold = config.ecs_thres
dab_weight = config.dab_weight

explicit_zero_prob = MLM and include_zero_gene  # whether explicit bernoulli for zeros
do_sample_in_train = False and explicit_zero_prob  # sample the bernoulli in training

per_seq_batch_sample = False

# SETTINGS FOR OPTIMIZER
# ================================================================================

lr = config["lr"]
batch_size = config["batch_size"]
eval_batch_size = config["batch_size"]
epochs = config["epochs"]
schedule_interval = 1

# LOGGING
# ================================================================================

log_interval = 100  # iterations
save_eval_interval = config["save_eval_interval"]  # epochs
do_eval_scib_metrics = True


#  DATASET AND SAVE DIRECTORY SETUP
# ================================================================================

dataset_name = config.dataset_name
save_dir = Path(f"./save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")
logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")

#  FUNCTION DEFINITIONS
# ================================================================================

def prepare_data(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    print(
        f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
        f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}"
    )

    input_gene_ids_train, input_gene_ids_valid = (
        tokenized_train["genes"],
        tokenized_valid["genes"],
    )
    input_values_train, input_values_valid = masked_values_train, masked_values_valid
    target_values_train, target_values_valid = (
        tokenized_train["target_values"],
        tokenized_valid["target_values"],
    )

    tensor_batch_labels_train = torch.from_numpy(tokenized_train["batch_labels"]).long()
    tensor_batch_labels_valid = torch.from_numpy(tokenized_valid["batch_labels"]).long()

    tensor_celltype_labels_train = torch.from_numpy(tokenized_train["celltype_labels"]).long()
    tensor_celltype_labels_valid = torch.from_numpy(tokenized_valid["celltype_labels"]).long()

    if sort_seq_batch:
        train_sort_ids = np.argsort(train_batch_labels)
        input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
        tensor_celltype_labels_train = tensor_celltype_labels_train[train_sort_ids]

        valid_sort_ids = np.argsort(valid_batch_labels)
        input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
        input_values_valid = input_values_valid[valid_sort_ids]
        target_values_valid = target_values_valid[valid_sort_ids]
        tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]
        tensor_celltype_labels_valid = tensor_celltype_labels_valid[valid_sort_ids]

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "batch_labels": tensor_batch_labels_train,
        "celltype_labels": tensor_celltype_labels_train,
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid,
        "batch_labels": tensor_batch_labels_valid,
        "celltype_labels": tensor_celltype_labels_valid,
    }

    return train_data_pt, valid_data_pt

class SeqDataset(Dataset):
    """
    Memory-efficient dataset that loads cells on-demand from AnnData
    Compatible with existing prepare_dataloader function
    """
    def __init__(self, data_pt):
        """
        Args:
            data_pt: Can be either:
                - Dict with pre-tokenized data (old way)
                - AnnData object (new memory-safe way)
        """
        # Check if it's the old dict format or new AnnData format
        if isinstance(data_pt, dict):
            # Old format - keep for compatibility
            self.use_dict = True
            self.gene_ids = data_pt["gene_ids"]
            self.values = data_pt["values"]
            self.target_values = data_pt["target_values"]
            self.batch_labels = data_pt["batch_labels"]
            self.celltype_labels = data_pt["celltype_labels"]
        else:
            # New format - AnnData object (memory efficient)
            self.use_dict = False
            self.adata = data_pt
            
            # Gene IDs - same for all cells
            self.gene_ids = torch.from_numpy(
                np.array([vocab[g] if g in vocab else vocab["<pad>"] 
                         for g in data_pt.var["gene_name"].tolist()], dtype=int)
            ).long()
            
            # Values - keep as reference to sparse matrix
            if "X_binned" in data_pt.layers:
                self.values = data_pt.layers["X_binned"]
            else:
                raise ValueError("X_binned layer missing")
            
            # Labels
            self.celltype_labels = torch.from_numpy(data_pt.obs["subtype_id"].values).long()
            self.batch_labels = torch.from_numpy(data_pt.obs["batch_id"].values).long()

    def __len__(self):
        if self.use_dict:
            return self.gene_ids.shape[0]
        else:
            return self.adata.n_obs

    def __getitem__(self, idx):
        if self.use_dict:
            # Old way - data already in memory
            return {
                "gene_ids": self.gene_ids[idx],
                "values": self.values[idx],
                "target_values": self.target_values[idx],
                "batch_labels": self.batch_labels[idx],
                "celltype_labels": self.celltype_labels[idx],
            }
        else:
            # New way - load cell on demand
            from scipy.sparse import issparse
            cell_values = self.values[idx]
            if issparse(cell_values):
                cell_values = cell_values.toarray().squeeze()
            
            cell_values = torch.from_numpy(cell_values).float()
            
            return {
                "gene_ids": self.gene_ids,
                "values": cell_values,
                "target_values": cell_values,
                "batch_labels": self.batch_labels[idx],
                "celltype_labels": self.celltype_labels[idx],
            }

# data_loader
def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
    sampler=None,  # ← NEW: Add sampler parameter
) -> DataLoader:
    
    dataset = SeqDataset(data_pt)

    if per_seq_batch_sample:
        # find the indices of samples in each seq batch
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=SubsetsBatchSampler(
                subsets,
                batch_size,
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=shuffle,
                drop_last=drop_last,
            ),
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader

    # Standard dataloader with optional sampler
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,  # Don't shuffle if using sampler
        sampler=sampler,  
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader


# ================================================================================
# TRAINING FUNCTION (Memory-Safe)
# ================================================================================

def train(model, loader, accumulation_steps=1):
    """
    Training function with memory safeguards
    """
    model.train()
    total_loss = 0.0
    total_error = 0.0
    total_num = 0
    log_interval = 100  # Log every 100 batches
    
    optimizer.zero_grad()  # Initialize
    
    for batch_idx, batch_data in enumerate(loader):
        try:
            # Move data to device
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            
            # Forward pass with AMP
            with torch.cuda.amp.autocast(enabled=config["amp"]):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if INPUT_BATCH_LABELS else None,
                    CLS=CLS,
                    CCE=CCE,
                    MVC=MVC,
                    ECS=ECS,
                )
                
                loss = 0.0
                
                # Classification loss (hierarchical)
                if CLS:
                    total_cls_loss, subtype_loss, parent_loss = criterion_cls(
                        output_dict["cls_output"], 
                        celltype_labels
                    )
                    loss = loss + total_cls_loss
                
                # MVC loss
                if MVC:
                    loss = loss + criterion(
                        output_dict["mvc_output"],
                        batch_data["target_values"].to(device),
                        input_values,
                    )
                
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Optimizer step (every N accumulation steps)
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Track metrics
            total_loss += loss.item() * accumulation_steps
            total_num += 1
            
            # Calculate error
            if CLS:
                error_rate = 1 - (
                    (output_dict["cls_output"].argmax(1) == celltype_labels)
                    .sum()
                    .item()
                ) / celltype_labels.size(0)
                total_error += error_rate
            
            # Clear GPU cache periodically
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect() 
            
            # Logging
            if batch_idx % log_interval == 0 and batch_idx > 0:
                lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - epoch_start_time) * 1000 / batch_idx
                cur_loss = total_loss / total_num
                cur_error = total_error / total_num if CLS else 0.0
                
                # Memory usage
                if torch.cuda.is_available():
                    mem_used = torch.cuda.max_memory_allocated() / 1e9
                    mem_reserved = torch.cuda.max_memory_reserved() / 1e9
                else:
                    mem_used = mem_reserved = 0
                
                print(
                    f"| batch {batch_idx:5d}/{len(loader):5d} | "
                    f"lr {lr:.2e} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.4f} | err {cur_error:5.4f} | "
                    f"GPU {mem_used:.1f}GB/{mem_reserved:.1f}GB"
                )
                
                # CRITICAL: Check if memory is too high
                if mem_reserved > 14.5:  # 14.5GB threshold on 16GB GPU
                    print("  WARNING: GPU memory high, clearing cache...")
                    torch.cuda.empty_cache()
                    gc.collect()

                # ============================================================
                # BATCH CHECKPOINT - Save every 5000 batches
                # ============================================================
                if batch_idx % 5000 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    checkpoint_path = save_dir / f"checkpoint_batch_{batch_idx}_epoch_{epoch}.pt"
                    torch.save({
                        'batch_idx': batch_idx,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'training_history': training_history,
                    }, checkpoint_path)
                    print(f"  ✓ Saved batch checkpoint at step {batch_idx}")
                    
                    # Keep only last 2 batch checkpoints
                    batch_checkpoints = sorted(save_dir.glob("checkpoint_batch_*.pt"))
                    if len(batch_checkpoints) > 2:
                        for old_ckpt in batch_checkpoints[:-2]:
                            old_ckpt.unlink()
                    
                    torch.cuda.empty_cache()
            
            # Delete tensors to free memory
            del input_gene_ids, input_values, celltype_labels, batch_labels
            del src_key_padding_mask, output_dict, loss
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n OOM ERROR at batch {batch_idx}")
                print(f"   Batch size: {batch_data['gene_ids'].shape[0]}")
                print(f"   GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
                
                # Emergency cleanup
                torch.cuda.empty_cache()
                gc.collect()
                
                # Skip this batch and continue
                print("   Skipping batch and continuing...")
                optimizer.zero_grad()
                continue
            else:
                raise e
    
    # Step scheduler after epoch
    scheduler.step()
    
    return total_loss / total_num, total_error / total_num if CLS else 0.0


def evaluate(model: nn.Module, loader: DataLoader, return_raw: bool = False) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_dab = 0.0
    total_num = 0
    predictions = []
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=config["amp"]):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if INPUT_BATCH_LABELS or config["DSBN"] else None,
                    CLS=CLS,  # evaluation does not need CLS or CCE
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=do_sample_in_train,
                    #generative_training = False,
                )

                output_values = output_dict["cls_output"]
                loss = criterion_cls(output_values, celltype_labels)

            if DAB:
                loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)

            total_loss += loss.item() * len(input_gene_ids)
            accuracy = (
                (output_values.argmax(1) == celltype_labels).sum().item()
            )
            total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
            total_dab += loss_dab.item() * len(input_gene_ids) if DAB else 0.0
            total_num += len(input_gene_ids)
            preds = output_values.argmax(1).cpu().numpy()
            predictions.append(preds)

    wandb.log(
        {
            "valid/mse": total_loss / total_num,
            "valid/err": total_error / total_num,
            "epoch": epoch,
        },
    )

    if return_raw:
        return np.concatenate(predictions, axis=0)

    return total_loss / total_num, total_error / total_num

#  CREATE DATALOADERS (Maximum Memory Safety)
# ================================================================================

print("\n" + "="*80)
print("CREATING DATALOADERS")
print("="*80)

# ========================================================================
# STEP 1: Pre-compute gene IDs
# ========================================================================

print("\n🔧 Pre-computing gene IDs...")

train_gene_ids = np.array(
    [vocab[g] if g in vocab else vocab["<pad>"] 
     for g in adata.var["gene_name"].tolist()],
    dtype=np.int64
)
val_gene_ids = np.array(
    [vocab[g] if g in vocab else vocab["<pad>"] 
     for g in adata_val.var["gene_name"].tolist()],
    dtype=np.int64
)
test_gene_ids = np.array(
    [vocab[g] if g in vocab else vocab["<pad>"] 
     for g in adata_test.var["gene_name"].tolist()],
    dtype=np.int64
)

print(f"  ✓ Gene IDs: {len(train_gene_ids)} genes")

# ========================================================================
# STEP 2: Simplified Dataset Class
# ========================================================================

class SeqDatasetWithGeneIds(Dataset):
    """Ultra memory-efficient dataset"""
    def __init__(self, adata, gene_ids):
        self.adata = adata
        self.gene_ids = torch.from_numpy(gene_ids).long()
        self.values = adata.layers["X_binned"]
        self.celltype_labels = torch.from_numpy(adata.obs["subtype_id"].values).long()
        self.batch_labels = torch.from_numpy(adata.obs["batch_id"].values).long()
    
    def __len__(self):
        return self.adata.n_obs
    
    def __getitem__(self, idx):
        from scipy.sparse import issparse
        cell_values = self.values[idx]
        if issparse(cell_values):
            cell_values = cell_values.toarray().squeeze()
        cell_values = torch.from_numpy(cell_values.astype(np.float32))
        
        return {
            "gene_ids": self.gene_ids,
            "values": cell_values,
            "target_values": cell_values,
            "batch_labels": self.batch_labels[idx],
            "celltype_labels": self.celltype_labels[idx],
        }

# ========================================================================
# STEP 3: Create loaders with maximum cleanup
# ========================================================================

print("\n Creating DataLoaders...\n")

# TRAIN
print("1. Train loader...")
try:
    train_loader = DataLoader(
        SeqDatasetWithGeneIds(adata, train_gene_ids),
        batch_size=config.batch_size,
        shuffle=False,
        sampler=train_sampler,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )
    print(f"   ✓ Created: {len(train_loader)} batches")
    
    # Test it works
    test_batch = next(iter(train_loader))
    print(f"   ✓ Tested: batch shape {test_batch['values'].shape}")
    del test_batch
    
except Exception as e:
    print(f"  FAILED: {e}")
    raise

del train_gene_ids
gc.collect()
torch.cuda.empty_cache()
memory_monitor.log_memory("After train loader")

# VAL
print("\n2. Val loader...")
try:
    val_loader = DataLoader(
        SeqDatasetWithGeneIds(adata_val, val_gene_ids),
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )
    print(f"   ✓ Created: {len(val_loader)} batches")
    
    # Test it works
    test_batch = next(iter(val_loader))
    print(f"   ✓ Tested: batch shape {test_batch['values'].shape}")
    del test_batch
    
except Exception as e:
    print(f"  FAILED: {e}")
    raise

del val_gene_ids
gc.collect()
torch.cuda.empty_cache()
memory_monitor.log_memory("After val loader")

# Clear more aggressively before test
import ctypes
ctypes.CDLL("libc.so.6").malloc_trim(0)  # Force memory release
gc.collect()
gc.collect()  
gc.collect()  # three times is the charm

print("\n3. Test loader...")
print(f"   Pre-creation memory check...")
memory_monitor.log_memory("Before test loader")

try:
    test_loader = DataLoader(
        SeqDatasetWithGeneIds(adata_test, test_gene_ids),
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )
    print(f"   ✓ Created: {len(test_loader)} batches")
    
    # Test it works
    test_batch = next(iter(test_loader))
    print(f"   ✓ Tested: batch shape {test_batch['values'].shape}")
    del test_batch
    
except Exception as e:
    print(f"  FAILED: {e}")
    print("\n Test loader creation failed - this is OK!")
    print("   We can create it on-demand during testing")
    test_loader = None  # Mark as failed

del test_gene_ids
gc.collect()
torch.cuda.empty_cache()
memory_monitor.log_memory("After all loaders")

# ========================================================================
# STEP 4: Summary
# ========================================================================

print("\n" + "="*80)
print("DATALOADERS STATUS")
print("="*80)
print(f"✓ Train: {len(train_loader)} batches")
print(f"✓ Val:   {len(val_loader)} batches")

if test_loader is not None:
    print(f"✓ Test:  {len(test_loader)} batches")
else:
    print(f"  Test:  Will create on-demand during evaluation")

print("="*80)

# ========================================================================
# STEP 5: Helper function to create test loader on-demand
# ========================================================================

if test_loader is None:
    print("\n Defining on-demand test loader function...")
    
    def create_test_loader_on_demand():
        """Create test loader when needed, after clearing train/val from memory"""
        test_gene_ids = np.array(
            [vocab[g] if g in vocab else vocab["<pad>"] 
             for g in adata_test.var["gene_name"].tolist()],
            dtype=np.int64
        )
        
        loader = DataLoader(
            SeqDatasetWithGeneIds(adata_test, test_gene_ids),
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=False,
        )
        
        del test_gene_ids
        gc.collect()
        
        return loader
    
    print("   ✓ Test loader will be created before final evaluation")

# VERIFY BALANCED BATCH COMPOSITION
# ================================================================================

print("\n" + "="*80)
print(" VERIFYING BALANCED BATCH COMPOSITION")
print("="*80)

print(f"\nDataLoader Summary:")
print(f"  Training:   {len(train_loader)} batches (balanced sampling)")
print(f"  Validation: {len(val_loader)} batches (natural distribution)")
print(f"  Test:       {len(test_loader)} batches (natural distribution)")

print("\n Sampling first 5 training batches...")
batch_compositions = []
temp_iterator = iter(train_loader)

for batch_idx in range(min(5, len(train_loader))):
    try:
        batch_data = next(temp_iterator)
        celltype_labels = batch_data["celltype_labels"]
        
        if torch.is_tensor(celltype_labels):
            celltype_labels = celltype_labels.cpu().numpy()
        
        unique_labels, counts = np.unique(celltype_labels, return_counts=True)
        batch_compositions.append((unique_labels, counts))
        
        print(f"\n Batch {batch_idx + 1}:")
        for label_idx, count in zip(unique_labels, counts):
            subtype_name = id2type[label_idx]
            print(f"  {subtype_name:<30}: {count:2d} cells")
            
    except StopIteration:
        break

del temp_iterator
gc.collect()

print("\n✓ Batch composition verified - sampling working correctly!")
print("="*80 + "\n")

# BATCH INFORMATION
# ================================================================================

print("\n" + "="*80)
print("CALCULATING BATCH INFORMATION")
print("="*80)

# Calculate number of unique batches
num_batch_types = len(adata.obs["batch_id"].unique())

print(f"Number of batch types: {num_batch_types}")
print(f"Number of subtypes: {num_types}")
print(f"Number of parent types: {num_parent_types}")

print("\nBatch distribution:")
print(adata.obs["str_batch"].value_counts().head(10))

print("="*80)

# DEVICE AND TRAINING SETTINGS
# ================================================================================

print("\n" + "="*80)
print("DEVICE AND TRAINING CONFIGURATION")
print("="*80)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Training parameters from config
lr = config.lr
epochs = config.epochs
save_eval_interval = config.save_eval_interval
schedule_interval = config.get("schedule_interval", 1)

print(f"Learning rate: {lr:.2e}")
print(f"Epochs: {epochs}")
print(f"Save interval: {save_eval_interval}")
print("="*80)

# SETTINGS FOR THE MODEL
# ================================================================================

print("\n" + "="*80)
print("MODEL CONFIGURATION")
print("="*80)

# Load model configs from pretrained model
if config.load_model is not None:
    model_file = Path(config.load_model)
    model_dir = model_file.parent
    config_file = model_dir / "args.json"
    
    if config_file.exists():
        print(f"Loading model config from: {config_file}")
        with open(config_file, 'r') as f:
            model_configs = json.load(f)
        print("✓ Loaded pretrained model config")
    else:
        print(f"  Config file not found: {config_file}")
        print("Using default configuration from hyperparameters")
        model_configs = {}
else:
    print("No pretrained model specified - using fresh configuration")
    model_configs = {}

# Model architecture settings (with fallbacks)
embsize = model_configs.get("embsize", config.layer_size)
d_hid = model_configs.get("d_hid", config.layer_size)
nlayers = model_configs.get("nlayers", config.nlayers)
nhead = model_configs.get("nhead", config.nhead)
n_layers_cls = model_configs.get("n_layers_cls", 3)
dropout = model_configs.get("dropout", config.dropout)

# Input/output settings
n_input_bins = model_configs.get("n_bins", config.n_bins)
input_emb_style = model_configs.get("input_emb_style", "continuous")
cell_emb_style = model_configs.get("cell_emb_style", "cls")
mvc_decoder_style = model_configs.get("mvc_decoder_style", "inner product")

# Task settings
explicit_zero_prob = model_configs.get("explicit_zero_prob", False)
ecs_threshold = model_configs.get("ecs_threshold", config.ecs_thres)

# Model components
CLS = True  # True for classification
CCE = False
MVC = config.MVC
ECS = config.ecs_thres > 0
DAB = False
INPUT_BATCH_LABELS = False
input_style = model_configs.get("input_style", "binned")

# Training settings
do_sample_in_train = False
per_seq_batch_sample = False

# Transformer backend
fast_transformer = config.fast_transformer
fast_transformer_backend = "flash" if fast_transformer else "linear"

print("\n Model Configuration:")
print(f"  Embedding size: {embsize}")
print(f"  Hidden size: {d_hid}")
print(f"  Layers: {nlayers}")
print(f"  Attention heads: {nhead}")
print(f"  Classification layers: {n_layers_cls}")
print(f"  Dropout: {dropout}")
print(f"  Input bins: {n_input_bins}")
print(f"  Fast transformer: {fast_transformer}")
print(f"\n Task Configuration:")
print(f"  Classification (CLS): {CLS}")
print(f"  MVC: {MVC}")
print(f"  ECS: {ECS}")
print(f"  Number of cell types: {num_types}")
print(f"  Number of batches: {num_batch_types}")
print("="*80)


#  MODEL CREATION
# ================================================================================

print("\n" + "="*80)
print("MODEL CREATION")
print("="*80)

ntokens = len(vocab)  # size of vocabulary
model = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    nlayers_cls=n_layers_cls,
    n_cls=num_types if CLS else 1,
    vocab=vocab,
    dropout=dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=MVC,
    do_dab=DAB,
    use_batch_labels=INPUT_BATCH_LABELS,
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=config["DSBN"],
    input_emb_style=input_emb_style,
    n_input_bins=n_input_bins,
    cell_emb_style=cell_emb_style,
    mvc_decoder_style=mvc_decoder_style,
    ecs_threshold=ecs_threshold,
    explicit_zero_prob=explicit_zero_prob,
    use_fast_transformer=fast_transformer,
    fast_transformer_backend=fast_transformer_backend,
    pre_norm=config["pre_norm"],
)

print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")


# MOVE MODEL TO DEVICE AND LOAD WEIGHTS
# ================================================================================

print("\n" + "="*80)
print("MODEL INITIALIZATION")
print("="*80)

# Move model to device FIRST
model.to(device)
print(f"✓ Model moved to {device}")

# THEN load pretrained weights (if specified)
if config["load_model"] is not None:
    model_file = Path(config["load_model"])
    print(f"\nLoading pretrained weights from: {model_file}")
    
    try:
        model.load_state_dict(torch.load(model_file, map_location=device))
        print(f"✓ Loaded all model params from {model_file}")
    except:
        # Only load params that match
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file, map_location=device)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            print(f"  Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"✓ Loaded {len(pretrained_dict)} matching params")

# Freeze parameters if specified
pre_freeze_param_count = sum(p.numel() for p in model.parameters())

if config["freeze"] and config["load_model"] is not None:
    print("\nFreezing encoder parameters...")
    frozen_count = 0
    for name, para in model.named_parameters():
        if config["freeze"] and "encoder" in name and "transformer_encoder" not in name:
            para.requires_grad = False
            frozen_count += 1
            print(f"  Frozen: {name}")
    print(f"✓ Frozen {frozen_count} parameters")

post_freeze_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nModel parameter summary:")
print(f"  Total params: {pre_freeze_param_count:,}")
print(f"  Trainable params: {post_freeze_param_count:,}")
print(f"  Frozen params: {pre_freeze_param_count - post_freeze_param_count:,}")
print("="*80)

#  HIERARCHICAL REJECTION SYSTEM - LEVEL 3
# ================================================================================

def integrate_hierarchical_with_evaluation(model, test_loader, device):
    """
    Integrate hierarchical rejection system with evaluation
    Returns hierarchical results and all predictions
    """
    hierarchical_classifier = HierarchicalRejectClassifier()
    model.eval()
    
    total_loss = 0.0
    total_error = 0.0

    all_logits, all_labels, all_predictions_standard = [], [], []

    with torch.no_grad():    
        with torch.cuda.amp.autocast(enabled=config["amp"]): 
            for batch_data in test_loader:
                input_gene_ids = batch_data["gene_ids"].to(device)
                input_values = batch_data["values"].to(device)
                celltype_labels = batch_data["celltype_labels"].to(device)
                src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
                
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                    CLS=True,
                )
                
                logits = output_dict["cls_output"]
                total_cls_loss, subtype_loss, parent_loss = criterion_cls(logits, celltype_labels)
                total_loss += total_cls_loss.item()
                
                error_rate = 1 - (
                    (logits.argmax(1) == celltype_labels).sum().item()
                ) / celltype_labels.size(0)
                total_error += error_rate
                
                all_logits.append(logits.cpu())
                all_labels.extend(celltype_labels.cpu().numpy())
                standard_preds = logits.argmax(1).cpu().numpy()
                all_predictions_standard.extend(standard_preds)
    
    num_batches = len(test_loader)
    val_loss = total_loss / num_batches
    val_err = total_error / num_batches
    
    all_logits_tensor = torch.cat(all_logits, dim=0)
    all_logits_tensor = all_logits_tensor.float()  
    all_labels_tensor = torch.tensor(all_labels)
    
    hierarchical_predictions = hierarchical_classifier.predict_with_hierarchy(all_logits_tensor)
    from sklearn.metrics import accuracy_score
    standard_accuracy = accuracy_score(all_labels, all_predictions_standard)
    hierarchical_results = hierarchical_classifier.evaluate_hierarchical_performance(
        all_logits_tensor, all_labels_tensor
    )
    
    print("\n" + "="*60)
    print("STANDARD vs HIERARCHICAL CLASSIFICATION COMPARISON")
    print("="*60)
    print(f"Standard Accuracy (forced predictions): {standard_accuracy:.1%}")
    print(f"Hierarchical Accuracy (confidence-aware): {hierarchical_results['overall_hierarchical_accuracy']:.1%}")
    
    dist = hierarchical_results['level_distribution']
    print("\n--- Hierarchical Decision Distribution ---")
    print(f"Specific predictions: {dist['specific_pct']:.1f}% ({dist['specific_predictions']} cells)")
    print(f"Family predictions:  {dist['family_pct']:.1f}% ({dist['family_predictions']} cells)")
    print(f"Broad predictions:   {dist['broad_pct']:.1f}% ({dist['broad_predictions']} cells)")
    print(f"Rejected predictions: {dist['rejected_pct']:.1f}% ({dist['rejected_predictions']} cells)")
    
    acc = hierarchical_results['accuracy_by_level']
    print("\n--- Accuracy by Confidence Level ---")
    print(f"Specific accuracy: {acc['specific_accuracy']:.1f}%")
    print(f"Family accuracy:   {acc['family_accuracy']:.1f}%")
    print(f"Broad accuracy:    {acc['broad_accuracy']:.1f}%")
    
    hierarchical_classifier.print_sample_results(hierarchical_predictions, n_samples=15)
    analyze_confidence_by_cell_type(hierarchical_predictions, all_labels)
    
    return val_loss, val_err, hierarchical_results, hierarchical_predictions


# HIERARCHICAL REJECTION - THEUNISSEN METHODOLOGY
# ============================================================================

class HierarchicalRejectClassifier:
    """
    Hierarchical reject system with bottom-up probability aggregation.
    Implements Theunissen et al. methodology properly.
    """
    
    def __init__(self, cell_types: List[str] = None):
        """
        Args:
            cell_types: List of cell type names from the dataset
        """
        if cell_types is None:
            # Default: use all known cell types from training
            self.cell_types = global_subtype_categories
        else:
            self.cell_types = cell_types
        
        # Build hierarchy dynamically
        self.cell_type_mapping = self._build_dynamic_mapping(self.cell_types)
        
        print(f"✓ Hierarchical classifier initialized with {len(self.cell_types)} cell types")
    
    def _build_dynamic_mapping(self, cell_types: List[str]) -> Dict:
        """
        Automatically map cell types to family/category based on keywords.
        Works with any brain cell dataset.
        """
        mapping = {}
        
        for idx, cell_type in enumerate(cell_types):
            cell_type_lower = cell_type.lower()
            
            # Level 3 (specific) → Level 2 (family) → Level 1 (broad)
            # Determine family based on keywords
            family, category = self._classify_cell_type(cell_type_lower)
            
            mapping[idx] = {
                'name': cell_type,
                'family': family,
                'category': category
            }
        
        return mapping
    
    def _classify_cell_type(self, cell_type_name: str) -> Tuple[str, str]:
        """
        Classify cell type into family and category based on collapsed biological groups.
        Handles both original cluster names and collapsed names with ranges.
        
        Returns: (family, category)
        """
        name_lower = cell_type_name.lower()
        
        # ========================================================================
        # GLIAL CELLS
        # ========================================================================
        
        # Oligodendrocytes (handles all formats with ranges)
        if any(x in name_lower for x in ['oligo_early', 'oligo_mid', 'oligo_late', 'oligo_other', 'oligo_', 'opc', 'cop']):
            return ('Glial cells', 'Oligodendrocytes')
        
        # Astrocytes (handles all formats with ranges)
        elif any(x in name_lower for x in ['astro_proto', 'astro_fibrous', 'astro_reactive', 'astro_other', 'astro_', 'bergmann', 'bgl_']):
            return ('Glial cells', 'Astrocytes')
        
        # ========================================================================
        # IMMUNE CELLS
        # ========================================================================
        
        # Microglia (handles all formats with ranges)
        elif any(x in name_lower for x in ['microglia_home', 'microglia_activ', 'microglia_other', 'microglia', 'mgl_']):
            return ('Immune cells', 'Microglia')
        
        # ========================================================================
        # VASCULAR CELLS
        # ========================================================================
        
        # Vascular endothelial
        elif any(x in name_lower for x in ['endothelial_cap', 'endothelial_art', 'endothelial_ven', 'endothelial_spec', 'endothelial', 'vend']):
            return ('Vascular cells', 'Endothelial')
        
        # Pericytes
        elif any(x in name_lower for x in ['pericyte', 'per_']):
            return ('Vascular cells', 'Pericytes')
        
        # Vascular smooth muscle
        elif any(x in name_lower for x in ['vascular_smc', 'vsmc_', 'smooth muscle']):
            return ('Vascular cells', 'SMC')
        
        # ========================================================================
        # EPITHELIAL CELLS
        # ========================================================================
        
        # Ependymal
        elif any(x in name_lower for x in ['ependymal', 'epen_']):
            return ('Epithelial cells', 'Ependymal')
        
        # Choroid plexus
        elif any(x in name_lower for x in ['choroid', 'chrp_']):
            return ('Epithelial cells', 'Choroid plexus')
        
        # ========================================================================
        # STROMAL CELLS
        # ========================================================================
        
        # Fibroblasts
        elif any(x in name_lower for x in ['fibroblast', 'fbl_']):
            return ('Stromal cells', 'Fibroblasts')
        
        # ========================================================================
        # UNKNOWN
        # ========================================================================
        
        else:
            return ('Unclassified', 'Unknown')
    
    def predict_with_hierarchy(self, model_output: torch.Tensor) -> List[Dict]:
        """
        Theunissen methodology with bottom-up probability aggregation.
        
        Args:
            model_output: Logits from model [batch_size, num_classes]
        
        Returns:
            List of prediction dictionaries with hierarchical decisions
        """
        # Get probabilities
        probabilities = F.softmax(model_output, dim=1)
        confidences, predicted_classes = torch.max(probabilities, dim=1)
        
        results = []
        for i, (confidence, predicted_class) in enumerate(zip(confidences, predicted_classes)):
            conf_val = confidence.item()
            class_idx = predicted_class.item()
            
            cell_info = self.cell_type_mapping.get(class_idx, {
                'name': f'Unknown_{class_idx}',
                'family': 'Unknown',
                'category': 'Unknown'
            })
            
            # BOTTOM-UP AGGREGATION (Theunissen methodology)
            # Calculate aggregated confidence at each hierarchical level
            category_confidence = self._calculate_category_confidence(
                probabilities[i], cell_info['category']
            )
            family_confidence = self._calculate_family_confidence(
                probabilities[i], cell_info['family']
            )
            
            # Hierarchical decision with aggregated confidences
            if conf_val >= 0.80:  # High confidence - specific level
                decision_level = 0
                final_prediction = cell_info['name']
                reasoning = f"High confidence ({conf_val:.1%}) - specific classification"
            
            elif category_confidence >= 0.65:  # Family level (aggregated)
                decision_level = 1
                final_prediction = cell_info['category']
                reasoning = f"Partial rejection ({category_confidence:.1%} aggregated) - family level"
            
            elif family_confidence >= 0.45:  # Category level (aggregated)
                decision_level = 2
                final_prediction = cell_info['family']
                reasoning = f"Partial rejection ({family_confidence:.1%} aggregated) - broad level"
            
            else:  # Full rejection
                decision_level = 3
                final_prediction = "Unknown"
                reasoning = f"Full rejection ({conf_val:.1%} max) - all hierarchical levels uncertain"
            
            results.append({
                'cell_id': i,
                'confidence': conf_val,
                'category_confidence': category_confidence,
                'family_confidence': family_confidence,
                'raw_prediction': cell_info['name'],
                'decision_level': decision_level,
                'final_prediction': final_prediction,
                'reasoning': reasoning
            })
        
        return results
    
    def _calculate_category_confidence(self, probabilities: torch.Tensor, target_category: str) -> float:
        """
        Calculate combined confidence for all cell types in a category.
        Bottom-up aggregation: sum probabilities of all related subtypes.
        
        Example: Oligodendrocytes category includes:
        - Oligo_mature: 0.35
        - Oligo_myelin: 0.25
        - OPC: 0.15
        → Category confidence: 0.75
        """
        category_prob_sum = 0.0
        for class_idx, prob in enumerate(probabilities):
            cell_info = self.cell_type_mapping.get(class_idx, {})
            if cell_info.get('category') == target_category:
                category_prob_sum += prob.item()
        return category_prob_sum
    
    def _calculate_family_confidence(self, probabilities: torch.Tensor, target_family: str) -> float:
        """
        Calculate combined confidence for all cell types in a family.
        Bottom-up aggregation: sum probabilities of all related categories.
        
        Example: Glial cells family includes:
        - All Oligodendrocytes: 0.45
        - All Astrocytes: 0.20
        → Family confidence: 0.65
        """
        family_prob_sum = 0.0
        for class_idx, prob in enumerate(probabilities):
            cell_info = self.cell_type_mapping.get(class_idx, {})
            if cell_info.get('family') == target_family:
                family_prob_sum += prob.item()
        return family_prob_sum
    
    def evaluate_hierarchical_performance(self, model_output: torch.Tensor, true_labels: torch.Tensor) -> Dict:
        """
        Evaluate hierarchical classification performance
        
        Returns comprehensive metrics about hierarchical decisions
        """
        predictions = self.predict_with_hierarchy(model_output)
        
        # Count decisions by level
        level_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        level_correct = {0: 0, 1: 0, 2: 0, 3: 0}
        
        total_cells = len(predictions)
        
        for pred, true_label_idx in zip(predictions, true_labels):
            level = pred['decision_level']
            level_counts[level] += 1
            
            true_cell_info = self.cell_type_mapping.get(true_label_idx.item(), {})
            
            # Check correctness at each level
            if level == 0:  # Specific
                if pred['final_prediction'] == true_cell_info.get('name'):
                    level_correct[0] += 1
            elif level == 1:  # Family
                if pred['final_prediction'] == true_cell_info.get('category'):
                    level_correct[1] += 1
            elif level == 2:  # Broad
                if pred['final_prediction'] == true_cell_info.get('family'):
                    level_correct[2] += 1
            # Level 3 (rejected) is never "correct"
        
        # Calculate metrics
        total_correct = sum(level_correct.values())
        overall_accuracy = total_correct / total_cells if total_cells > 0 else 0
        
        results = {
            'overall_hierarchical_accuracy': overall_accuracy,
            'level_distribution': {
                'specific_predictions': level_counts[0],
                'family_predictions': level_counts[1],
                'broad_predictions': level_counts[2],
                'rejected_predictions': level_counts[3],
                'specific_pct': 100 * level_counts[0] / total_cells,
                'family_pct': 100 * level_counts[1] / total_cells,
                'broad_pct': 100 * level_counts[2] / total_cells,
                'rejected_pct': 100 * level_counts[3] / total_cells,
            },
            'accuracy_by_level': {
                'specific_accuracy': 100 * level_correct[0] / level_counts[0] if level_counts[0] > 0 else 0,
                'family_accuracy': 100 * level_correct[1] / level_counts[1] if level_counts[1] > 0 else 0,
                'broad_accuracy': 100 * level_correct[2] / level_counts[2] if level_counts[2] > 0 else 0,
            },
            'total_cells': total_cells,
            'total_correct': total_correct
        }
        
        return results
    
    def print_sample_results(self, predictions: List[Dict], n_samples: int = 10):
        """Print sample hierarchical predictions"""
        print(f"\nSample Hierarchical Predictions (first {n_samples}):")
        print("="*80)
        for i, pred in enumerate(predictions[:n_samples]):
            print(f"Cell {pred['cell_id']}:")
            print(f"  Raw prediction: {pred['raw_prediction']}")
            print(f"  Confidences:")
            print(f"    - Specific: {pred['confidence']:.1%}")
            print(f"    - Category: {pred['category_confidence']:.1%}")
            print(f"    - Family:   {pred['family_confidence']:.1%}")
            print(f"  Decision: Level {pred['decision_level']} → {pred['final_prediction']}")
            print(f"  Reasoning: {pred['reasoning']}")
            print()


# ============================================================================
# HELPER FUNCTION FOR CONFIDENCE ANALYSIS
# ============================================================================

def analyze_confidence_by_cell_type(hierarchical_predictions, all_labels):
    """
    Analyze confidence and rejection patterns by cell type.
    
    Args:
        hierarchical_predictions: List of prediction dicts
        all_labels: True labels (tensor or array)
    """
    # Analyze rejection rates by cell type
    rejection_by_type = {}
    
    for pred, true_label in zip(hierarchical_predictions, all_labels):
        level = pred['decision_level']
        cell_type = id2type[true_label.item() if torch.is_tensor(true_label) else true_label]
        
        if cell_type not in rejection_by_type:
            rejection_by_type[cell_type] = {0: 0, 1: 0, 2: 0, 3: 0, 'total': 0}
        
        rejection_by_type[cell_type][level] += 1
        rejection_by_type[cell_type]['total'] += 1
    
    print("\n" + "="*80)
    print("CONFIDENCE PATTERNS BY CELL TYPE")
    print("="*80)
    
    for cell_type in sorted(rejection_by_type.keys()):
        if cell_type in rejection_by_type:
            stats = rejection_by_type[cell_type]
            total = stats['total']
            print(f"\n{cell_type}:")
            print(f"  Level 0 (specific): {stats[0]:4d} ({stats[0]/total*100:5.1f}%)")
            print(f"  Level 1 (family):   {stats[1]:4d} ({stats[1]/total*100:5.1f}%)")
            print(f"  Level 2 (broad):    {stats[2]:4d} ({stats[2]/total*100:5.1f}%)")
            print(f"  Level 3 (rejected): {stats[3]:4d} ({stats[3]/total*100:5.1f}%)")
    
    # Overall rejection stats
    level_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for pred in hierarchical_predictions:
        level_counts[pred['decision_level']] += 1
    
    total_preds = len(hierarchical_predictions)
    print(f"\n" + "="*80)
    print("OVERALL HIERARCHICAL DISTRIBUTION")
    print("="*80)
    print(f"  Specific predictions:  {level_counts[0]:5d} ({level_counts[0]/total_preds*100:.1f}%)")
    print(f"  Family predictions:    {level_counts[1]:5d} ({level_counts[1]/total_preds*100:.1f}%)")
    print(f"  Broad predictions:     {level_counts[2]:5d} ({level_counts[2]/total_preds*100:.1f}%)")
    print(f"  Full rejections:       {level_counts[3]:5d} ({level_counts[3]/total_preds*100:.1f}%)")
    print("="*80)


print("✓ Sophisticated hierarchical classifier with Theunissen aggregation loaded")


def analyze_confidence_by_cell_type(hierarchical_predictions, all_labels):
    """Analyze which cell types are confidently classified vs rejected"""
    
    print("\n" + "="*60)
    print("HIERARCHICAL REJECTION ANALYSIS BY CELL TYPE")
    print("="*60)
    
    confidence_by_type = {}
    
    for pred, true_label in zip(hierarchical_predictions, all_labels):
        cell_type = global_subtype_categories[true_label]
        
        if cell_type not in confidence_by_type:
            confidence_by_type[cell_type] = {
                0: 0,  # Specific
                1: 0,  # Family
                2: 0,  # Broad
                3: 0   # Rejected
            }
        
        confidence_by_type[cell_type][pred['decision_level']] += 1
    
    print("\nCell Type Rankings (by average confidence):")
    print(f"{'Cell Type':<20} | {'Specific':>8} | {'Family':>8} | {'Broad':>8} | {'Rejected':>8}")
    print("-" * 70)
    
    # Sort by specific prediction rate
    sorted_types = sorted(
        confidence_by_type.items(),
        key=lambda x: x[1][0],
        reverse=True
    )
    
    for cell_type, counts in sorted_types:
        total = sum(counts.values())
        if total > 0:  #  Safety check
            specific_pct = counts[0] / total * 100
            family_pct = counts[1] / total * 100
            broad_pct = counts[2] / total * 100
            rejected_pct = counts[3] / total * 100
            
            #  ALL percentages calculated properly
            print(f"{cell_type:<20} | {specific_pct:>7.1f}% | {family_pct:>7.1f}% | "
                  f"{broad_pct:>7.1f}% | {rejected_pct:>7.1f}%")

# ================================================================================
# OPTIMIZER, SCHEDULER, SCALER SETUP
# ================================================================================

print("\n" + "="*80)
print("TRAINING SETUP")
print("="*80)

# Get schedule parameters from config (or use defaults)
schedule_interval = config.get("schedule_interval", 1)  #  From config or default to 1
schedule_ratio = config.get("schedule_ratio", 0.9)      #  From config or default to 0.9

print(f"Schedule settings: interval={schedule_interval}, ratio={schedule_ratio}")

# Create optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config["lr"],
    eps=1e-4 if config["amp"] else 1e-8,
)

# Create scheduler
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=schedule_interval,
    gamma=schedule_ratio
)

# Create scaler for mixed precision
scaler = torch.cuda.amp.GradScaler(enabled=config["amp"])

# Training history
training_history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'learning_rates': [],
    'epoch_times': [],
}

print(f"✓ Optimizer: Adam (lr={config['lr']:.2e})")
print(f"✓ Scheduler: StepLR (step={schedule_interval}, gamma={schedule_ratio})")
print(f"✓ Scaler: AMP={config['amp']}")
print(f"✓ Training history initialized")
print("="*80)

# ================================================================================
# PRE-TRAINING MEMORY CLEANUP
# ================================================================================

print("\n" + "="*80)
print("PRE-TRAINING MEMORY CLEANUP")
print("="*80)

# Clear any lingering data
import gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Check current memory
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"GPU memory before training:")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved: {reserved:.2f} GB")
    print(f"  Free: {(14.76 - allocated):.2f} GB")
    
    if allocated > 8.0:
        print(f"\n  WARNING: Already using {allocated:.2f}GB before training!")
        print("  Likely causes:")
        print("  - Model is very large")
        print("  - Data loaders materialized data")
        print("  - Previous cells left tensors in memory")

print("="*80)


# ================================================================================
# MEMORY DIAGNOSTIC
# ================================================================================

import gc

print("\n" + "="*80)
print("MEMORY DIAGNOSTIC")
print("="*80)

# Check tensors in memory
print("\nLarge tensors in memory:")
for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj):
            size_mb = obj.element_size() * obj.nelement() / 1e6
            if size_mb > 100:  # Show tensors > 100MB
                print(f"  {type(obj).__name__}: {obj.shape} = {size_mb:.0f} MB")
    except:
        pass

# Check model size
if 'model' in dir():
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    print(f"\nModel size: {model_size:.2f} GB")

# Check data loaders
if 'train_loader' in dir():
    print(f"\nTrain loader: {len(train_loader.dataset):,} samples")

print("="*80)


# PRE-TRAINING MEMORY CHECK
# ================================================================================

print("\n" + "="*80)
print("PRE-TRAINING MEMORY CHECK")
print("="*80)

# Test one forward+backward pass
print("Testing memory requirements...")
model.train()
test_batch = next(iter(train_loader))

try:
    with torch.cuda.amp.autocast(enabled=config["amp"]):
        input_gene_ids = test_batch["gene_ids"].to(device)
        input_values = test_batch["values"].to(device)
        celltype_labels = test_batch["celltype_labels"].to(device)
        
        output_dict = model(
            input_gene_ids,
            input_values,
            src_key_padding_mask=input_gene_ids.eq(vocab[pad_token]),
            batch_labels=None,
            CLS=True,
        )
        
        total_cls_loss, _, _ = criterion_cls(output_dict["cls_output"], celltype_labels)
        loss = total_cls_loss / config.gradient_accumulation_steps
    
    scaler.scale(loss).backward()
    
    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    print(f"✓ Memory test passed")
    print(f"  Peak GPU memory: {peak_memory:.2f}GB")
    
    if peak_memory > 14.0:
        print(f"\n  WARNING: Peak memory {peak_memory:.2f}GB is close to limit!")
        print(f"  Recommendation: Reduce batch_size to 4 or enable CPU offloading")
    
    # Cleanup
    optimizer.zero_grad()
    del input_gene_ids, input_values, celltype_labels, output_dict, loss
    torch.cuda.empty_cache()
    
    print("✓ Ready to start training")
    
except RuntimeError as e:
    print(f"\n MEMORY TEST FAILED: {e}")
    print("\nSuggested fixes:")
    print("  1. Reduce batch_size to 4 (or even 2)")
    print("  2. Disable MVC if enabled")
    print("  3. Reduce model size (fewer layers)")
    raise

print("="*80)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

early_stopping = EarlyStopping(patience=2, min_delta=0.0005)


# TRAINING LOOP with Gradient Accumulation
# ================================================================================

best_val_loss = float("inf")
best_model = None
best_model_epoch = 0

# Track global training state
global epoch_start_time

print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)
print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
print(f"Total epochs: {epochs}")
print(f"Steps per epoch: {len(train_loader)}")
print("="*80)

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"EPOCH {epoch}/{epochs}")
    print('='*80)
    
    try:
        # Training
        if config["do_train"]:
            train_loss, train_err = train(
                model, 
                train_loader, 
                accumulation_steps=config.gradient_accumulation_steps
            )
            train_acc = 1 - train_err 
            print(f"\nTrain - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        
        # Validation
        print("\nValidating...")
        val_loss, val_err, hierarchical_results, _ = integrate_hierarchical_with_evaluation(
            model, val_loader, device
        )
        val_acc = 1 - val_err 
        
        # Get learning rate
        current_lr = scheduler.get_last_lr()[0]
        
        elapsed = time.time() - epoch_start_time
        
        # LOG TO HISTORY 
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        training_history['learning_rates'].append(current_lr)
        training_history['epoch_times'].append(elapsed)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"Epoch {epoch} complete - Time: {elapsed:.1f}s")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  LR: {current_lr:.2e}")
        print(f"  Hierarchical Accuracy: {hierarchical_results['overall_hierarchical_accuracy']:.1%}")
        print('='*80)
        
        # delete to free memory (after logging)
        del train_loss, train_err, train_acc
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            best_model_epoch = epoch
            torch.save(best_model.state_dict(), save_dir / "best_model.pt")
            print(f"✓ Saved new best model (epoch {epoch})")

        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\n{'='*80}")
            print(f" EARLY STOPPING at epoch {epoch}")
            print(f"   Best validation loss: {best_val_loss:.4f} at epoch {best_model_epoch}")
            print(f"   No improvement for {early_stopping.patience} epochs")
            print(f"{'='*80}")
            break
        
        # Checkpoint (with memory safety)
        if epoch % save_eval_interval == 0:
            # Clear memory before saving (checkpoint creation allocates memory)
            torch.cuda.empty_cache()
            gc.collect()
    
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'training_history': training_history, 
            }, save_dir / f"checkpoint_epoch_{epoch}.pt")
            print(f"✓ Saved checkpoint")
    
            torch.cuda.empty_cache()

        # After validation, BEFORE next epoch
        del val_loss, val_err, hierarchical_results
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n OOM during epoch {epoch}")
            print("Attempting recovery...")
            
            # Emergency cleanup
            torch.cuda.empty_cache()
            gc.collect()
            
            # Save emergency checkpoint
            torch.save(model.state_dict(), save_dir / f"emergency_epoch_{epoch}.pt")
            print(f"✓ Saved emergency checkpoint")
            
            print("\n  Training halted due to OOM")
            print("Try: Reduce batch_size or increase gradient_accumulation_steps")
            break
        else:
            raise e

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
if best_model is not None:
    print(f"Best model: Epoch {best_model_epoch}, Loss: {best_val_loss:.4f}")
print("="*80)


'''
# Load best model weights
print("Loading best model...")
best_model_path = Path("./save/dev_FIXED_Allen_Atlas-Collpased-Nov24-13-08/best_model.pt")
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()
print(f"✓ Loaded: {best_model_path}")

# Load training history
print("Loading training history...")
checkpoint = torch.load("./save/dev_FIXED_Allen_Atlas-Collpased-Nov24-13-08/checkpoint_epoch_4.pt", map_location=device)
training_history = checkpoint['training_history']
best_val_loss = checkpoint['best_val_loss']
print(f"✓ Loaded: {len(training_history['train_loss'])} epochs")
'''


# TRAINING VISUALIZATIONS (After Training Completes)
# ================================================================================

print("\n" + "="*80)
print("TRAINING VISUALIZATIONS")
print("="*80)

def plot_training_history_level3(history, save_dir):
    """Create comprehensive training visualization for Level 3"""
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure with subplots - SAME LAYOUT AS LEVEL 1
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History - Level 3 (214 Subtypes)', fontsize=16, y=1.02)
    
    # Colors - blue/orange like standard ML papers
    train_color = '#1f77b4'  # Blue
    val_color = '#FF8C00'    # Orange
    
    # 1. Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], color=train_color, linewidth=2.5, 
             marker='o', markersize=4, label='Train Loss')
    ax1.plot(epochs, history['val_loss'], color=val_color, linewidth=2.5, 
             marker='s', markersize=4, label='Val Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Mark best epoch
    best_epoch_idx = np.argmin(history['val_loss'])
    ax1.axvline(x=best_epoch_idx+1, color='green', linestyle='--', alpha=0.5, linewidth=2)
    
    # 2. Accuracy curves
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_acc'], color=train_color, linewidth=2.5,
             marker='o', markersize=4, label='Train Acc')
    ax2.plot(epochs, history['val_acc'], color=val_color, linewidth=2.5,
             marker='s', markersize=4, label='Val Acc')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=best_epoch_idx+1, color='green', linestyle='--', alpha=0.5, linewidth=2)
    
    # Add best accuracy annotation
    best_val_acc = max(history['val_acc'])
    ax2.axhline(y=best_val_acc, color='green', linestyle=':', alpha=0.3, linewidth=2)
    ax2.text(0.5, best_val_acc, f'Best: {best_val_acc:.3f}', 
             fontsize=10, color='green', va='bottom')
    
    # 3. Learning rate schedule
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['learning_rates'], color='green', linewidth=2.5,
             marker='o', markersize=4)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Learning Rate', fontsize=12)
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # 4. Overfitting analysis (Train-Val gap)
    ax4 = axes[1, 1]
    loss_gap = np.array(history['val_loss']) - np.array(history['train_loss'])
    acc_gap = np.array(history['train_acc']) - np.array(history['val_acc'])
    
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(epochs, loss_gap, color='#d62728', linewidth=2.5,
                     marker='o', markersize=4, label='Loss Gap (Val-Train)')
    line2 = ax4_twin.plot(epochs, acc_gap, color=train_color, linewidth=2.5,
                          marker='s', markersize=4, label='Acc Gap (Train-Val)')
    
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Loss Gap', fontsize=12, color='#d62728')
    ax4_twin.set_ylabel('Accuracy Gap', fontsize=12, color=train_color)
    ax4.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
    ax4.tick_params(axis='y', labelcolor='#d62728')
    ax4_twin.tick_params(axis='y', labelcolor=train_color)
    ax4.grid(True, alpha=0.3)
    
    # Add zero lines
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax4_twin.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=300, bbox_inches='tight')
    
    # WandB upload
    try:
        if config.get('use_wandb', False):
            wandb.log({"training/curves": wandb.Image(str(save_dir / "training_curves.png"))})
    except:
        pass  # Skip if wandb not configured
    
    plt.show()
    
    print("✓ Training curves saved")
    
    # Summary statistics
    print("\n" + "="*60)
    print("TRAINING SUMMARY STATISTICS")
    print("="*60)
    print(f"Total training time: {sum(history['epoch_times']):.1f}s ({sum(history['epoch_times'])/60:.1f} min)")
    print(f"Average epoch time: {np.mean(history['epoch_times']):.1f}s")
    print(f"Best validation accuracy: {max(history['val_acc']):.4f} (Epoch {np.argmax(history['val_acc'])+1})")
    print(f"Best validation loss: {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss'])+1})")
    print()
    print(f"Final train accuracy: {history['train_acc'][-1]:.4f}")
    print(f"Final val accuracy: {history['val_acc'][-1]:.4f}")
    print(f"Final train-val gap: {history['train_acc'][-1] - history['val_acc'][-1]:.4f}")
    
    # Check for overfitting
    if history['train_acc'][-1] - history['val_acc'][-1] > 0.15:
        print("\n  Warning: Significant overfitting detected (gap > 15%)")
    elif history['train_acc'][-1] - history['val_acc'][-1] > 0.08:
        print("\n  Moderate overfitting detected (gap > 8%)")
    else:
        print("\n✓ Model generalizes well (gap < 8%)")
    
    print("="*60)


# Call the function
plot_training_history_level3(training_history, save_dir)


# ============================================================================
# LOAD BEST MODEL 
# ============================================================================

print("\n" + "="*80)
print("LOADING BEST MODEL FOR TEST EVALUATION")
print("="*80)

# Case 1: best_model exists and is a model object (from fresh training)
if 'best_model' in globals() and hasattr(best_model, 'eval'):
    print("  ✓ Using best_model from training")
    best_model.to(device)
    best_model.eval()

# Case 2: best_model is a Path or doesn't exist (from checkpoint resume)
else:
    print("  Loading from checkpoint...")
    best_model_path = Path("./save/dev_FIXED_Allen_Atlas-Collpased-Nov24-13-08/best_model.pt")
    
    if best_model_path.exists():
        # Use the existing model and load best weights
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        best_model = model
        best_model.to(device)
        best_model.eval()
        print(f"  ✓ Loaded from: {best_model_path}")
    else:
        print(f" Checkpoint not found: {best_model_path}")
        raise FileNotFoundError(f"Cannot find best model at {best_model_path}")

print("✓ Best model ready for evaluation")
print("="*80)


# TEST EVALUATION
# ================================================================================

print("\n" + "="*60)
print("EVALUATING ON TEST SET")
print("="*60)

# Evaluate on test set using hierarchical system
test_loss, test_err, test_hierarchical_results, test_predictions_hier = integrate_hierarchical_with_evaluation(
    best_model, test_loader, device
)

print(f"\nTest Performance:")
print(f"  Accuracy: {(1 - test_err) * 100:.2f}%")
print(f"  Loss: {test_loss:.4f}")
print(f"  Hierarchical Accuracy: {test_hierarchical_results['overall_hierarchical_accuracy']:.1%}")

print("\n✓ Test evaluation complete!")
print("="*60)

# ============================================================================
# FULL SUBTYPE CONFUSION MATRIX (22 Classes)
# ============================================================================

print("\n" + "="*80)
print("CREATING CONFUSION MATRICES")
print("="*80)

# Extract labels and predictions
test_labels_actual = []
test_preds_actual = []

for batch in test_loader:
    test_labels_actual.extend(batch['celltype_labels'].cpu().numpy())

for pred in test_predictions_hier:
    raw_pred_name = pred['raw_prediction']
    pred_idx = [i for i, name in id2type.items() if name == raw_pred_name][0]
    test_preds_actual.append(pred_idx)

test_labels_actual = np.array(test_labels_actual)
test_preds_actual = np.array(test_preds_actual)

# Get class names in order
unique_classes = sorted(set(test_labels_actual) | set(test_preds_actual))
class_names = [id2type[i] for i in unique_classes]

# ============================================================================
# SUBTYPE CONFUSION MATRIX
# ============================================================================

print("\n1. Creating full subtype confusion matrix (22x22)...")

cm = confusion_matrix(test_labels_actual, test_preds_actual, labels=unique_classes)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

cm_df = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)

# Plot
fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(
    cm_df,
    annot=False,  # Too cluttered with 22 classes
    fmt='.2f',
    cmap='Blues',
    square=True,
    ax=ax,
    cbar_kws={'label': 'Normalized Frequency'},
    vmin=0,
    vmax=1
)

plt.title('Subtype-Level Confusion Matrix (22 Classes)', fontsize=16, pad=20)
plt.xlabel('Predicted Subtype', fontsize=13)
plt.ylabel('True Subtype', fontsize=13)
plt.xticks(rotation=90, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()

plt.savefig(save_dir / "confusion_matrix_subtype.png", dpi=300, bbox_inches='tight')
plt.show()
print("  ✓ Saved: confusion_matrix_subtype.png")

# ============================================================================
# PARENT CONFUSION MATRIX 
# ============================================================================

print("\n2. Creating parent-level confusion matrix...")

test_labels_parent = [subtype_to_parent[id2type[label]] for label in test_labels_actual]
test_preds_parent = [subtype_to_parent[id2type[pred]] for pred in test_preds_actual]

cm_parent = confusion_matrix(test_labels_parent, test_preds_parent)
cm_parent_normalized = cm_parent.astype('float') / cm_parent.sum(axis=1)[:, np.newaxis]

parent_types_unique = sorted(set(test_labels_parent) | set(test_preds_parent))
cm_parent_df = pd.DataFrame(
    cm_parent_normalized,
    index=parent_types_unique,
    columns=parent_types_unique
)

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    cm_parent_df,
    annot=True,
    fmt='.2f',
    cmap='Blues',
    square=True,
    ax=ax,
    cbar_kws={'label': 'Normalized Frequency'}
)

plt.title('Parent-Level Confusion Matrix (12 Parent Types)', fontsize=16, pad=20)
plt.xlabel('Predicted Parent Type', fontsize=13)
plt.ylabel('True Parent Type', fontsize=13)
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()

plt.savefig(save_dir / "confusion_matrix_parent.png", dpi=300, bbox_inches='tight')
plt.show()
print("  ✓ Saved: confusion_matrix_parent.png")

print("\n✓ Confusion matrices complete!")
print("="*80)

# ============================================================================
# UMAP VISUALIZATION - EMERGENCY FIX
# ============================================================================
# This version DYNAMICALLY gets exact cell type names from your data
# No more spacing issues!
# ============================================================================

import matplotlib.patches as mpatches

print("\n" + "="*80)
print("UMAP VISUALIZATION - LEVEL 3 (DYNAMIC COLOR ASSIGNMENT)")
print("="*80)

# ============================================================================
# GET ACTUAL CELL TYPE NAMES FROM DATA
# ============================================================================

print("\n✓ Reading actual cell type names from data...")
actual_cell_types = [id2type[i] for i in range(len(id2type))]
print(f"  Found {len(actual_cell_types)} cell types")
print(f"\n  First 10 cell types (with exact spacing):")
for i, ct in enumerate(actual_cell_types[:10]):
    print(f"    {i}: '{ct}'")

# ============================================================================
# SMART COLOR ASSIGNMENT (PATTERN MATCHING)
# ============================================================================

def assign_colors_to_actual_types(cell_types, vibrant=False):
    """
    Assign colors based on cell type name patterns
    This works regardless of spacing variations!
    """
    color_map = {}
    
    for ct in cell_types:
        ct_lower = ct.lower().replace(' ', '')  # Remove all spaces for matching
        
        # OLIGODENDROCYTES - MAXIMALLY different colors!
        if 'oligo' in ct_lower and 'mid' in ct_lower:
            color_map[ct] = '#00FFFF' if vibrant else '#00CED1'  # CYAN
            print(f"  → {ct}: CYAN")
        elif 'oligo' in ct_lower and 'late' in ct_lower:
            color_map[ct] = '#FF00FF' if vibrant else '#FF1493'  # PINK
            print(f"  → {ct}: PINK")
        elif 'oligo' in ct_lower and 'early' in ct_lower:
            color_map[ct] = '#FFFF00' if vibrant else '#FFD700'  # GOLD
            print(f"  → {ct}: GOLD")
        
        # ASTROCYTES - Color wheel opposites!
        elif 'astro' in ct_lower and 'fibrous' in ct_lower:
            color_map[ct] = '#FF0000' if vibrant else '#DC143C'  # RED
            print(f"  → {ct}: RED")
        elif 'astro' in ct_lower and 'protoplasmic' in ct_lower:
            color_map[ct] = '#0000FF' if vibrant else '#4169E1'  # BLUE
            print(f"  → {ct}: BLUE")
        elif 'astro' in ct_lower and 'reactive' in ct_lower:
            color_map[ct] = '#FF6600' if vibrant else '#FF8C00'  # ORANGE
            print(f"  → {ct}: ORANGE")
        
        # MICROGLIA - Opposite colors
        elif 'microglia' in ct_lower and 'activated' in ct_lower:
            color_map[ct] = '#CC0000' if vibrant else '#8B0000'  # DARK RED
            print(f"  → {ct}: DARK RED")
        elif 'microglia' in ct_lower and 'homeostatic' in ct_lower:
            color_map[ct] = '#00FF00' if vibrant else '#32CD32'  # GREEN
            print(f"  → {ct}: GREEN")
        
        # OPC
        elif 'opc' in ct_lower:
            color_map[ct] = '#9933FF' if vibrant else '#9400D3'  # VIOLET
            print(f"  → {ct}: VIOLET")
        
        # ENDOTHELIAL - Check longest names first!
        elif 'endothelial' in ct_lower and 'arterial' in ct_lower and 'capillary' in ct_lower:
            color_map[ct] = '#FF9900' if vibrant else '#FFA500'  # ORANGE
            print(f"  → {ct}: ORANGE")
        elif 'endothelial' in ct_lower and 'venous' in ct_lower and 'capillary' in ct_lower:
            color_map[ct] = '#00CCFF' if vibrant else '#00BFFF'  # DEEP SKY BLUE
            print(f"  → {ct}: DEEP SKY BLUE")
        elif 'endothelial' in ct_lower and 'arterial' in ct_lower:
            color_map[ct] = '#FF4444' if vibrant else '#FF6347'  # TOMATO
            print(f"  → {ct}: TOMATO")
        elif 'endothelial' in ct_lower and 'venous' in ct_lower:
            color_map[ct] = '#0099FF' if vibrant else '#1E90FF'  # DODGER BLUE
            print(f"  → {ct}: DODGER BLUE")
        elif 'endothelial' in ct_lower and 'capillary' in ct_lower:
            color_map[ct] = '#FF1493' if vibrant else '#FF69B4'  # HOT PINK
            print(f"  → {ct}: HOT PINK")
        elif 'endothelial' in ct_lower and 'specialized' in ct_lower:
            color_map[ct] = '#CC66FF' if vibrant else '#BA55D3'  # ORCHID
            print(f"  → {ct}: ORCHID")
        
        # OTHER TYPES
        elif 'bergmann' in ct_lower:
            color_map[ct] = '#00CC00' if vibrant else '#228B22'  # GREEN
            print(f"  → {ct}: GREEN")
        elif 'cop' in ct_lower:
            color_map[ct] = '#FF5500' if vibrant else '#FF4500'  # ORANGE RED
            print(f"  → {ct}: ORANGE RED")
        elif 'choroid' in ct_lower:
            color_map[ct] = '#996633' if vibrant else '#8B4513'  # BROWN
            print(f"  → {ct}: BROWN")
        elif 'ependymal' in ct_lower:
            color_map[ct] = '#6600CC' if vibrant else '#4B0082'  # INDIGO
            print(f"  → {ct}: INDIGO")
        elif 'fibroblast' in ct_lower:
            color_map[ct] = '#336666' if vibrant else '#2F4F4F'  # SLATE
            print(f"  → {ct}: SLATE")
        elif 'pericyte' in ct_lower:
            color_map[ct] = '#669933' if vibrant else '#556B2F'  # OLIVE
            print(f"  → {ct}: OLIVE")
        elif 'vascular' in ct_lower and 'smc' in ct_lower:
            color_map[ct] = '#990099' if vibrant else '#800080'  # PURPLE
            print(f"  → {ct}: PURPLE")
        
        # FALLBACK
        else:
            print(f"  UNMATCHED: {ct} → using gray")
            color_map[ct] = '#808080'  # Gray
    
    return color_map

print("\nAssigning STRATEGIC colors:")
STRATEGIC_COLORS = assign_colors_to_actual_types(actual_cell_types, vibrant=False)

print("\nAssigning VIBRANT colors:")
VIBRANT_COLORS = assign_colors_to_actual_types(actual_cell_types, vibrant=True)

# Verify all types have colors
if len(STRATEGIC_COLORS) == len(actual_cell_types):
    print(f"\n✓ SUCCESS! All {len(actual_cell_types)} cell types have colors!")
else:
    print(f"\n  WARNING: Only {len(STRATEGIC_COLORS)}/{len(actual_cell_types)} types have colors")

# ============================================================================
# EXTRACT EMBEDDINGS
# ============================================================================

def extract_test_embeddings_l3(model, test_loader, device):
    """Extract embeddings from test set"""
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=config.amp):
            for batch_data in test_loader:
                input_gene_ids = batch_data["gene_ids"].to(device)
                input_values = batch_data["values"].to(device)
                src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
                
                output = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=True,
                )
                
                embeddings = output["cell_emb"]
                all_embeddings.append(embeddings.cpu().numpy())
    
    return np.concatenate(all_embeddings, axis=0)

print("\nExtracting test embeddings...")
test_embeddings = extract_test_embeddings_l3(best_model, test_loader, device)
print(f"✓ Extracted {test_embeddings.shape[0]} embeddings")

# Create visualization AnnData
adata_test_vis = adata_test.copy()
adata_test_vis.obs["predictions"] = [id2type[p] for p in test_preds_actual] 
adata_test_vis.obs["true_labels"] = [id2type[label] for label in test_labels_actual]  
adata_test_vis.obsm["X_scgpt"] = test_embeddings

# Compute UMAP
print("\nComputing UMAP...")
sc.pp.neighbors(adata_test_vis, use_rep="X_scgpt", n_neighbors=15)
sc.tl.umap(adata_test_vis)
print("✓ UMAP computed")

# ============================================================================
# VISUALIZATION 1: STRATEGIC COLORS
# ============================================================================

print("\nCreating UMAP with strategic colors...")
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

sc.pl.umap(
    adata_test_vis, 
    color="true_labels", 
    ax=axes[0], 
    show=False, 
    title="True Cell Subtypes (Level 3 - Strategic Colors)", 
    palette=STRATEGIC_COLORS,
    alpha=0.7,
    s=3,
    legend_loc='none',
)

sc.pl.umap(
    adata_test_vis, 
    color="predictions", 
    ax=axes[1], 
    show=False, 
    title="Predicted Cell Subtypes (Level 3 - Strategic Colors)", 
    palette=STRATEGIC_COLORS,
    alpha=0.7,
    s=3,
    legend_loc='none',
)

plt.tight_layout()
plt.savefig(save_dir / "umap_level3_strategic.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: umap_level3_strategic.png")

# ============================================================================
# VISUALIZATION 2: VIBRANT COLORS
# ============================================================================

print("\nCreating UMAP with vibrant colors...")
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

sc.pl.umap(
    adata_test_vis, 
    color="true_labels", 
    ax=axes[0], 
    show=False, 
    title="True Cell Subtypes (Level 3 - Vibrant Colors)", 
    palette=VIBRANT_COLORS,
    alpha=0.8,
    s=3,
    legend_loc='none',
)

sc.pl.umap(
    adata_test_vis, 
    color="predictions", 
    ax=axes[1], 
    show=False, 
    title="Predicted Cell Subtypes (Level 3 - Vibrant Colors)", 
    palette=VIBRANT_COLORS,
    alpha=0.8,
    s=3,
    legend_loc='none',
)

plt.tight_layout()
plt.savefig(save_dir / "umap_level3_vibrant.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: umap_level3_vibrant.png")

# ============================================================================
# LEGENDS WITH RECTANGLES
# ============================================================================

sorted_types = sorted(STRATEGIC_COLORS.keys())

print("\nCreating legends with rectangles...")

# Legend 1: Horizontal Strategic
fig, ax = plt.subplots(1, 1, figsize=(24, 3))
ax.axis('off')
patches = [mpatches.Patch(facecolor=STRATEGIC_COLORS[ct], edgecolor='black', linewidth=0.5, label=ct) 
           for ct in sorted_types]
ax.legend(handles=patches, loc='center', ncol=6, fontsize=11, frameon=True, fancybox=False,
          title='Cell Subtypes (22 Classes)', title_fontsize=13, columnspacing=1.5,
          handlelength=2.0, handleheight=1.5)
plt.tight_layout()
plt.savefig(save_dir / "legend_horizontal_strategic.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Legend 2: Vertical Strategic
fig, ax = plt.subplots(1, 1, figsize=(6, 14))
ax.axis('off')
patches = [mpatches.Patch(facecolor=STRATEGIC_COLORS[ct], edgecolor='black', linewidth=0.5, label=ct) 
           for ct in sorted_types]
ax.legend(handles=patches, loc='center', ncol=1, fontsize=12, frameon=True, fancybox=False,
          title='Cell Subtypes (22 Classes)', title_fontsize=14, handlelength=2.5, handleheight=1.5)
plt.tight_layout()
plt.savefig(save_dir / "legend_vertical_strategic.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Legend 3: Horizontal Vibrant
fig, ax = plt.subplots(1, 1, figsize=(24, 3))
ax.axis('off')
patches = [mpatches.Patch(facecolor=VIBRANT_COLORS[ct], edgecolor='black', linewidth=0.5, label=ct) 
           for ct in sorted_types]
ax.legend(handles=patches, loc='center', ncol=6, fontsize=11, frameon=True, fancybox=False,
          title='Cell Subtypes (22 Classes)', title_fontsize=13, columnspacing=1.5,
          handlelength=2.0, handleheight=1.5)
plt.tight_layout()
plt.savefig(save_dir / "legend_horizontal_vibrant.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Legend 4: Vertical Vibrant
fig, ax = plt.subplots(1, 1, figsize=(6, 14))
ax.axis('off')
patches = [mpatches.Patch(facecolor=VIBRANT_COLORS[ct], edgecolor='black', linewidth=0.5, label=ct) 
           for ct in sorted_types]
ax.legend(handles=patches, loc='center', ncol=1, fontsize=12, frameon=True, fancybox=False,
          title='Cell Subtypes (22 Classes)', title_fontsize=14, handlelength=2.5, handleheight=1.5)
plt.tight_layout()
plt.savefig(save_dir / "legend_vertical_vibrant.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("\n✓ All legends saved!")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✓ UMAP VISUALIZATION COMPLETE!")
print("="*80)
print("\nGenerated 6 files:")
print("  1. umap_level3_strategic.png")
print("  2. umap_level3_vibrant.png")
print("  3. legend_horizontal_strategic.png")
print("  4. legend_vertical_strategic.png")
print("  5. legend_horizontal_vibrant.png")
print("  6. legend_vertical_vibrant.png")
print("\nAll files saved to:", save_dir)
print("="*80)
# ============================================================================
# COMPREHENSIVE PERFORMANCE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("COMPREHENSIVE PERFORMANCE ANALYSIS")
print("="*80)

# ============================================================================
# 1. PER-CLASS METRICS (Bar Chart)
# ============================================================================

print("\n1. Calculating per-class metrics...")

# Get precision, recall, F1 for each class
precision, recall, f1, support = precision_recall_fscore_support(
    test_labels_actual, 
    test_preds_actual,
    labels=unique_classes,
    zero_division=0
)

# Create DataFrame
metrics_df = pd.DataFrame({
    'Class': class_names,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support
})

# Sort by F1 (worst to best)
metrics_df = metrics_df.sort_values('F1-Score')

# Plot
fig, ax = plt.subplots(figsize=(14, 10))

x = np.arange(len(metrics_df))
width = 0.25

bars1 = ax.barh(x - width, metrics_df['Precision'], width, label='Precision', alpha=0.8)
bars2 = ax.barh(x, metrics_df['Recall'], width, label='Recall', alpha=0.8)
bars3 = ax.barh(x + width, metrics_df['F1-Score'], width, label='F1-Score', alpha=0.8)

ax.set_xlabel('Score', fontsize=13)
ax.set_ylabel('Cell Type', fontsize=13)
ax.set_title('Per-Class Performance Metrics (Sorted by F1-Score)', fontsize=15, pad=20)
ax.set_yticks(x)
ax.set_yticklabels(metrics_df['Class'], fontsize=10)
ax.legend(fontsize=11)
ax.set_xlim([0, 1])
ax.grid(axis='x', alpha=0.3)

# Add mean line
f1_mean = metrics_df['F1-Score'].mean()
ax.axvline(f1_mean, color='red', linestyle='--', linewidth=2, 
           label=f'Mean F1: {f1_mean:.3f}', alpha=0.7)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(save_dir / "per_class_performance.png", dpi=300, bbox_inches='tight')
plt.show()
print("  ✓ Saved: per_class_performance.png")

# ============================================================================
# 2. MACRO VS MICRO AVERAGES (Table + Bar)
# ============================================================================

print("\n2. Computing macro vs micro averages...")

# Calculate averages
macro_precision = precision.mean()
macro_recall = recall.mean()
macro_f1 = f1.mean()

# Micro averages (weighted by support)
micro_precision = (precision * support).sum() / support.sum()
micro_recall = (recall * support).sum() / support.sum()
micro_f1 = (f1 * support).sum() / support.sum()

# Create comparison
avg_comparison = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1-Score'],
    'Macro (Unweighted)': [macro_precision, macro_recall, macro_f1],
    'Micro (Weighted)': [micro_precision, micro_recall, micro_f1]
})

print("\nMacro vs Micro Averages:")
print(avg_comparison.to_string(index=False))

# Plot comparison
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(avg_comparison))
width = 0.35

bars1 = ax.bar(x - width/2, avg_comparison['Macro (Unweighted)'], 
               width, label='Macro (Unweighted)', alpha=0.8)
bars2 = ax.bar(x + width/2, avg_comparison['Micro (Weighted)'], 
               width, label='Micro (Weighted)', alpha=0.8)

ax.set_xlabel('Metric', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Macro vs Micro Average Performance', fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(avg_comparison['Metric'], fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(save_dir / "macro_vs_micro.png", dpi=300, bbox_inches='tight')
plt.show()
print("  ✓ Saved: macro_vs_micro.png")

# ============================================================================
# 3. PERFORMANCE VS FREQUENCY (Scatter Plot)
# ============================================================================

print("\n3. Analyzing performance vs class frequency...")

fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot
scatter = ax.scatter(
    metrics_df['Support'], 
    metrics_df['F1-Score'],
    s=100,
    alpha=0.6,
    c=metrics_df['F1-Score'],
    cmap='RdYlGn',
    edgecolors='black',
    linewidth=0.5
)

# Add labels for worst performers (F1 < 0.6)
for idx, row in metrics_df[metrics_df['F1-Score'] < 0.6].iterrows():
    ax.annotate(
        row['Class'],
        (row['Support'], row['F1-Score']),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=9,
        alpha=0.8
    )

ax.set_xlabel('Number of Samples (Support)', fontsize=12)
ax.set_ylabel('F1-Score', fontsize=12)
ax.set_title('Performance vs Class Frequency', fontsize=14, pad=15)
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
ax.axhline(f1_mean, color='red', linestyle='--', linewidth=2, 
           label=f'Mean F1: {f1_mean:.3f}', alpha=0.7)
ax.legend(fontsize=11)

plt.colorbar(scatter, ax=ax, label='F1-Score')
plt.tight_layout()
plt.savefig(save_dir / "performance_vs_frequency.png", dpi=300, bbox_inches='tight')
plt.show()
print("  ✓ Saved: performance_vs_frequency.png")

# ============================================================================
# 4. WORST PERFORMERS TABLE
# ============================================================================

print("\n4. Identifying worst performers...")

worst_5 = metrics_df.head(5)

print("\nBottom 5 Classes by F1-Score:")
print(worst_5[['Class', 'Precision', 'Recall', 'F1-Score', 'Support']].to_string(index=False))

# Save to CSV
metrics_df_full = metrics_df.sort_values('F1-Score', ascending=False)
metrics_df_full.to_csv(save_dir / "per_class_metrics.csv", index=False)
print("\n  ✓ Saved: per_class_metrics.csv")


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("PERFORMANCE ANALYSIS SUMMARY")
print("="*80)
print(f"Overall Test Accuracy: {(test_labels_actual == test_preds_actual).mean():.1%}")
print(f"\nMacro Averages (treats all classes equally):")
print(f"  Precision: {macro_precision:.3f}")
print(f"  Recall:    {macro_recall:.3f}")
print(f"  F1-Score:  {macro_f1:.3f}")
print(f"\nMicro Averages (weighted by support):")
print(f"  Precision: {micro_precision:.3f}")
print(f"  Recall:    {micro_recall:.3f}")
print(f"  F1-Score:  {micro_f1:.3f}")
print(f"\nClass Imbalance Impact:")
gap = micro_f1 - macro_f1
if gap > 0.05:
    print(f"  Micro-Macro gap: {gap:.3f} (model biased toward frequent classes)")
else:
    print(f" Micro-Macro gap: {gap:.3f} (balanced performance)")
print("="*80)

print("\n✓ Comprehensive performance analysis complete!")
