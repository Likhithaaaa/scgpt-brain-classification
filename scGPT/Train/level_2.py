#!/usr/bin/env python
# coding: utf-8

# =========================
# Standard Library
# =========================
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

# =========================
# Data Handling
# =========================
import pandas as pd
import numpy as np
from scipy.sparse import issparse

# =========================
# Scientific Libraries
# =========================
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Single-cell / Bioinformatics
# =========================
import scanpy as sc
import scvi
from anndata import AnnData

# =========================
# PyTorch Core
# =========================
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =========================
# PyTorch NLP (TorchText)
# =========================
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind

# =========================
# ML Evaluation Metrics
# =========================
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.utils.class_weight import compute_class_weight

# =========================
# Experiment Tracking
# =========================
import wandb

# =========================
# scGPT Core Modules
# =========================
# Make sure project root is in path
sys.path.insert(0, "../")

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

# =========================
# Scanpy Plot Settings
# =========================
sc.set_figure_params(figsize=(6, 6))

# =========================
# Suppress Warnings
# =========================
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

from torch.cuda.amp import autocast, GradScaler


# =========================
# Balanced dataset with weights
# =========================
def weighted_cross_entropy(output, target, weight=None):
    """Cross entropy with class weights"""
    return F.cross_entropy(output, target, weight=weight)


# criterion_cls
criterion_cls = weighted_cross_entropy

# =========================
# Dataset Splitter
# =========================
class DatasetSplitter:
    """
    Simple splitter - just specify balanced, imbalanced, or auto-detect
    """
    
    def __init__(self, balance_type="auto"):
        """
        balance_type:
        - "balanced" → random split
        - "imbalanced" → stratified split
        - "auto" → auto-detect based on data
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
        
        print(f"Dataset Statistics:")
        print(f"  Total samples: {total_samples:,}")
        print(f"  Cell types: {len(celltype_counts)}")
        print(f"  Largest class: {max_class_size:,} samples")
        print(f"  Smallest class: {min_class_size:,} samples ({min_class_percentage:.2f}%)")
        print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")
        print(f"="*60)
        
        # Determine split strategy
        if self.balance_type == "auto":
            # Auto-detect based on data characteristics
            use_stratify = imbalance_ratio > 3.0 or min_class_percentage < 2.0
            detected_type = "imbalanced" if use_stratify else "balanced"
            print(f" Auto-detected as: {detected_type}")
        elif self.balance_type == "balanced":
            use_stratify = False
        elif self.balance_type == "imbalanced":
            use_stratify = True
        else:
            raise ValueError("balance_type must be 'balanced', 'imbalanced', or 'auto'")
        
        strategy = "STRATIFIED" if use_stratify else "RANDOM"
        print(f"Split strategy: {strategy}")
        
        # Prepare stratification variable
        stratify_var = adata_full.obs["celltype"] if use_stratify else None
        
        # Check if stratification is possible
        if use_stratify:
            min_test_samples = int(min_class_size * test_size)
            min_val_samples = int(min_class_size * val_size)
            
            if min_test_samples < 1 or min_val_samples < 1:
                print(f" WARNING: Smallest class too small for stratification!")
                print(f"   Falling back to RANDOM split instead...")
                use_stratify = False
                stratify_var = None
        
        try:
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
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_relative_size,
                stratify=adata_full.obs.iloc[train_val_idx]["celltype"] if use_stratify else None,
                random_state=random_state,
                shuffle=True
            )
            
        except ValueError as e:
            print(f"Stratification failed: {e}")
            print(f"   Using RANDOM split instead...")
            
            # Fallback to random split
            train_val_idx, test_idx = train_test_split(
                range(len(adata_full)),
                test_size=test_size,
                random_state=random_state,
                shuffle=True
            )
            
            val_relative_size = val_size / (1 - test_size)
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_relative_size,
                random_state=random_state,
                shuffle=True
            )
            
            use_stratify = False
        
        # Create splits
        adata_train = adata_full[train_idx].copy()
        adata_val = adata_full[val_idx].copy()
        adata_test = adata_full[test_idx].copy()
        
        # Verify splits
        print(f"\n SPLIT RESULTS:")
        print(f"  Train: {len(adata_train):,} samples ({len(adata_train)/len(adata_full)*100:.1f}%)")
        print(f"  Val:   {len(adata_val):,} samples ({len(adata_val)/len(adata_full)*100:.1f}%)")
        print(f"  Test:  {len(adata_test):,} samples ({len(adata_test)/len(adata_full)*100:.1f}%)")
        print(f" Split completed using {strategy} strategy")
        print(f"-"*60)
        
        return adata_train, adata_val, adata_test, use_stratify


# In[4]:


def load_dataset_simple(dataset_path, metadata_path=None, gene_list_path=None):
    """
    Universal dataset loader - works for any .h5ad file
    """
    
    print(f" Loading dataset from: {dataset_path}")
    
    # Load the raw data file
    adata_full = sc.read_h5ad(dataset_path)
    print(f" Loaded raw data: {adata_full.shape}")
    
    # Handle gene filtering if provided
    if gene_list_path and os.path.exists(gene_list_path):
        print("Applying gene filtering...")
        gene_list = pd.read_csv(gene_list_path, header=None)[0].tolist()
        adata_full = adata_full[:, adata_full.var_names.isin(gene_list)].copy()
        print(f"After gene filtering: {adata_full.shape}")
    

    # Handle metadata if provided
    if metadata_path and os.path.exists(metadata_path):
        print("Loading metadata...")
        meta = pd.read_csv(metadata_path)
        
        # Handle UKF-style metadata
        if "barcode" in meta.columns and "CellTypes" in meta.columns:
            meta["barcode"] = meta["barcode"].str.strip()
            meta = meta.set_index("barcode")
            adata_full.obs_names = adata_full.obs_names.str.strip()
            adata_full.obs = meta.loc[adata_full.obs_names][["CellTypes"]].copy()
            adata_full.obs["celltype"] = adata_full.obs["CellTypes"].astype("category")
        else:
            print(" Metadata format not recognized, skipping...")
    
    # Standardize celltype column
    if "celltype" not in adata_full.obs.columns:
        # Try to find celltype column
        celltype_candidates = ["cell_type", "CellType", "cluster", "labels", "annotation"]
        celltype_col = None
        for col in celltype_candidates:
            if col in adata_full.obs.columns:
                celltype_col = col
                break
        
        if celltype_col:
            adata_full.obs["celltype"] = adata_full.obs[celltype_col].astype("category")
            print(f" Found celltype column: {celltype_col}")
        else:
            raise ValueError(f"No celltype column found. Available columns: {list(adata_full.obs.columns)}")
    
    # Handle batch information
    if "batch" in adata_full.obs.columns:
        ori_batch_col = "batch"
        adata_full.obs["str_batch"] = adata_full.obs["batch"].astype(str)
    elif "sample_id" in adata_full.obs.columns:
        ori_batch_col = "sample_id"
        adata_full.obs["str_batch"] = adata_full.obs["sample_id"].astype(str)
    else:
        # Default: single batch
        ori_batch_col = "str_batch"
        adata_full.obs["str_batch"] = "0"
    
    adata_full.obs["batch_id"] = adata_full.obs["str_batch"].astype("category").cat.codes.values
    
    # Common final processing
    adata_full.var["gene_name"] = adata_full.var.index.tolist()
    data_is_raw = True
    
    print(f" Dataset processed: {adata_full.shape}")
    print(f" Cell types found: {list(adata_full.obs['celltype'].unique())}")


    
    return adata_full, ori_batch_col, data_is_raw


def analyze_dataset_distribution(adata, celltype_column='celltype', save_dir=None, dataset_name=''):
    """
    Create comprehensive visualization and analysis of cell type distribution.
    
    Args:
        adata: AnnData object with cell type annotations
        celltype_column: Column name containing cell type labels
        save_dir: Directory to save plots (optional)
        dataset_name: Name for plot titles
    """
    
    # Get cell type counts
    if hasattr(adata.obs[celltype_column], 'cat'):
        # Categorical data
        celltype_counts = adata.obs[celltype_column].value_counts()
    else:
        # String data
        celltype_counts = pd.Series(adata.obs[celltype_column]).value_counts()
    total_cells = len(adata)
    num_cell_types = len(celltype_counts)
    
    print(f"="*60)
    print(f"DATASET DISTRIBUTION ANALYSIS: {dataset_name}")
    print(f"="*60)
    print(f"Total cells: {total_cells:,}")
    print(f"Number of cell types: {num_cell_types}")
    print(f"Average cells per type: {total_cells/num_cell_types:.1f}")
    print()
    
    # Calculate percentages
    celltype_percentages = (celltype_counts / total_cells * 100).round(2)
    
    # Create distribution DataFrame
    distribution_df = pd.DataFrame({
        'Cell Type': celltype_counts.index,
        'Count': celltype_counts.values,
        'Percentage': celltype_percentages.values
    }).sort_values('Count', ascending=False)
    
    # Print detailed statistics
    print("CELL TYPE DISTRIBUTION:")
    print("-" * 50)
    for i, row in distribution_df.iterrows():
        print(f"{row['Cell Type']:25s}: {row['Count']:6,} cells ({row['Percentage']:5.1f}%)")
    
    print("\nDISTRIBUTION STATISTICS:")
    print("-" * 30)
    print(f"Most common: {distribution_df.iloc[0]['Cell Type']} (distribution: {distribution_df.iloc[0]['Percentage']:.2f}%)")
    print(f"Least common: {distribution_df.iloc[-1]['Cell Type']} (distribution: {distribution_df.iloc[-1]['Percentage']:.2f}%)") 
    print(f"Imbalance ratio: {distribution_df.iloc[0]['Count']} / {distribution_df.iloc[-1]['Count']:.2f}%")   

    # Classify balance
    max_pct = distribution_df.iloc[0]['Percentage']
    min_pct = distribution_df.iloc[-1]['Percentage']

    if max_pct / min_pct < 3:
        balance_status = "BALANCED"
        balance_color = "green"
    elif max_pct / min_pct < 10:
        balance_status = "MODERATELY IMBALANCED"
        balance_color = "orange"
    else:
        balance_status = "HIGHLY IMBALANCED"
        balance_color = "red"

    print(f"\nBalance status: {balance_status}")
   
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Horizontal bar plot with counts
    ax1 = axes[0, 0]
    bars1 = ax1.barh(range(len(distribution_df)), distribution_df['Count'],
                     color=plt.cm.viridis(np.linspace(0, 1, len(distribution_df))))
    ax1.set_yticks(range(len(distribution_df)))
    ax1.set_yticklabels(distribution_df['Cell Type'], fontsize=9)
    ax1.set_xlabel('Number of Cells')
    ax1.set_title(f'{dataset_name} - Cell Counts', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars1, distribution_df['Count'])):
        ax1.text(bar.get_width() + max(distribution_df['Count'])*0.01, bar.get_y() + bar.get_height()/2,
                f'{count:,}', va='center', fontsize=8)
    
    # 2. Pie chart for top cell types, group others
    ax2 = axes[0, 1]
    # Show top 8 cell types, group others
    if len(distribution_df) > 8:
        top_8 = distribution_df.head(8)
        others_count = distribution_df.tail(len(distribution_df)-8)['Count'].sum()
        pie_data = list(top_8['Count']) + [others_count]
        pie_labels = list(top_8['Cell Type']) + ['Others']
    else:
        pie_data = distribution_df['Count']
        pie_labels = distribution_df['Cell Type']
    
    wedges, texts, autotexts = ax2.pie(pie_data, labels=pie_labels, autopct='%1.1f%%',
                                        startangle=90, textprops={'fontsize': 8})
    ax2.set_title(f'{dataset_name} - Distribution', fontweight='bold')
    
    # 3. Distribution histogram
    ax3 = axes[1, 0]
    bins = np.logspace(np.log10(min(distribution_df['Count'])),
                       np.log10(max(distribution_df['Count'])), 15)
    ax3.hist(distribution_df['Count'], bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_xscale('log')
    ax3.set_xlabel('Number of Cells (log scale)')
    ax3.set_ylabel('Number of Cell Types')
    ax3.set_title('Cell Count Distribution', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Cumulative percentage plot
    ax4 = axes[1, 1]
    cumulative_pct = distribution_df['Percentage'].cumsum()
    ax4.plot(range(1, len(cumulative_pct)+1), cumulative_pct, 'o-', linewidth=2, markersize=4)
    ax4.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% threshold')
    ax4.set_xlabel('Number of Top Cell Types')
    ax4.set_ylabel('Cumulative Percentage (%)')
    ax4.set_title('Cumulative Cell Type Distribution', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Add 80% line analysis
    cells_for_80pct = np.where(cumulative_pct >= 80)[0]
    if len(cells_for_80pct) > 0:
        ax4.axvline(x=cells_for_80pct[0]+1, color='red', linestyle='--', alpha=0.7)
        ax4.text(cells_for_80pct[0]+1.5, 40, f'Top {cells_for_80pct[0]+1}\ntypes = 80%',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.suptitle(f'{dataset_name} Cell Type Distribution Analysis\n'
                f'Status: {balance_status}', fontsize=14, fontweight='bold',
                color=balance_color)
    plt.tight_layout()
    
    # Save plot if directory provided
    if save_dir:
        plt.savefig(save_dir / f"dataset_distribution_{dataset_name.lower().replace(' ', '_')}.png",
                   dpi=300, bbox_inches='tight')
        print(f"\n Distribution plots saved to: {save_dir}")
    
    plt.show()
    
    # Additional analysis
    print("\n" + "="*60)
    print("CLASSIFICATION IMPLICATIONS:")
    print("-" * 40)
    
    rare_types = distribution_df[distribution_df['Percentage'] < 1.0]
    if len(rare_types) > 0:
        print(f" {len(rare_types)} rare cell types (<1%): {', '.join(rare_types['Cell Type'])}")
        print("  → These may be difficult to classify accurately")
    
    dominant_types = distribution_df[distribution_df['Percentage'] > 10.0]
    if len(dominant_types) > 0:
        print(f" {len(dominant_types)} dominant cell types (>10%): {', '.join(dominant_types['Cell Type'])}")
        print("  → Model may be biased toward these")
    
    if balance_status == "HIGHLY IMBALANCED":
        print("\n Highly imbalanced dataset:")
        print("  → Consider stratified sampling for train/validation splits")
        print("  → May need class weighting or resampling techniques")
        print("  → Evaluate using precision/recall, not just accuracy")
    elif balance_status == "MODERATELY IMBALANCED":
        print("\n Moderately imbalanced dataset:")
        print("  → Use stratified splits")
        print("  → Monitor per-class performance")
    else:
        print("\n Well-balanced dataset:")
        print("  → Random splits should work well")
        print("  → Standard accuracy metrics are reliable")
    
    return distribution_df


def quick_distribution_summary(adata, celltype_column='celltype'):
    """Quick one-liner distribution summary."""
    celltype_counts = adata.obs[celltype_column].value_counts()
    total_cells = len(adata)
    max_ratio = celltype_counts.iloc[0] / celltype_counts.iloc[-1]
    
    print(f" Dataset: {total_cells:,} cells, {len(celltype_counts)} types, "
          f"imbalance ratio: {max_ratio:.1f}:1")
    
    return celltype_counts

# =========================
# Hyperparameters setup for cell type annotation
# =========================

hyperparameter_defaults = dict(
    seed=0,
    
    dataset_path="./data/Reference.h5ad",  # Path to .h5ad file
    dataset_name="cancer_l2",  # Name for logging
    balance_type="auto",  # "balanced", "imbalanced", or "auto" #50k is imbalanced - trying auto for now
    metadata_path="./data/Ref_metadata.csv",  # metadata file
    gene_list_path="./data/Ref_gene_names.csv", # gene filtering
    
    do_train=True,
    load_model="save/scGPT_brain",
    mask_ratio=0.0,
    epochs=6,
    n_bins=51,
    MVC=False,  # Masked value prediction for cell embedding
    ecs_thres=0.0,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight=0.0,
    #lr = 0.00005,
    lr=1e-4,
    
    batch_size=8,
    layer_size=128,
    nlayers=4,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead=4,  # number of heads in nn.MultiheadAttention
    dropout=0.2,  # dropout probability
    schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
    save_eval_interval=5,
    fast_transformer=True,
    pre_norm=False,
    amp=True,  # Automatic Mixed Precision. Combines 16-bit (FP16) and 32-bit (FP32) floating point
    # FP16 reduces memory usage and speeds up computations.
    #FP32 is used when precision is critical
    include_zero_gene = False,
    freeze = False,  #freeze
    DSBN = False,  # Domain-spec batchnorm
)

run = wandb.init(
    config=hyperparameter_defaults,
    project="scGPT",
    reinit=True,
    #settings=wandb.Settings(start_method="fork"),
)
config = wandb.config
print(config)

set_seed(config.seed)

# GPU Memory and Optimization Setup

import gc
import torch

# Clear GPU cache
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Configure memory settings for Tesla T4 16GB
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:true"
torch.cuda.set_per_process_memory_fraction(0.75, device=0)

# Enable performance optimizations
torch.backends.cudnn.benchmark = True

# GPU and Flash attention setup
print(f" GPU Setup:")
print(f" Device: {torch.cuda.get_device_name()}")
print(f" Pythorch Version: {torch.__version__}")
print(f" CUDA Version: {torch.version.cuda}")
print(f" Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB total")

# Enable Flash
try:   
    if torch.backends.cuda.flash_sdp_enabled():
        torch.backends.cuda.enable_flash_sdp(True)
        print(f" Flash attention enabled")
    else:
        print(f" Flash attention not available")
except AttributeError:
    print(f" Flash attention not supported in this PyTorch version")

print(f" Memory optimization enabled! \n")


n_hvg = 1500  # number of highly variable genes
max_seq_len = n_hvg + 1
per_seq_batch_sample = False

# =========================
# Settings for input and preprocessing
# =========================
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]  # special tokens used for padding and marking sequences
mask_ratio = config["mask_ratio"]  # proportion of input tokens to mask for training
mask_value = "auto"  # for masked values, now it should always be auto

include_zero_gene = config["include_zero_gene"]  # if True, include zero genes among HVGs in the train dataset
max_seq_len = 3001
n_bins = config["n_bins"]  # Number of bins for discretizing gene expression values

# input/output representation
input_style = "binned"  # "normed_raw", "logip", or "binned"
output_style = "binned"  # "normed_raw", "logip", or "binned"


# =========================
# Settings for training
# =========================
MLM = False  # whether to use masked language modeling, currently it is always on.
CLS = True  # celltype classification objective
ADV = False  # Adversarial training for batch correction
CCE = False  # Contrastive cell embedding objective
MVC = config["MVC"]  # Masked value prediction for cell embedding
ECS = config["ecs_thres"]  # 0 #config.ecs_thres 0 to disable Elastic cell similarity objective
DAB = False  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer
INPUT_BATCH_LABELS = False  # TODO: have these help MLM and MVC, while not to classifier
input_emb_style = "continuous"  # "category" or "continuous" or "scaling"
cell_emb_style = "cls"  # "avg-pool" or "w-pool" or "cls"
adv_E_delay_epochs = 0  # delay adversarial training on encoder for a few epochs
adv_D_delay_epochs = 0
mvc_decoder_style = "inner product"  # https://github.com/bowang-lab/scGPT/blob/7301b51a72f5db.py
ecs_threshold = config["ecs_thres"]
dab_weight = config["dab_weight"]

explicit_zero_prob = MLM and include_zero_gene  # whether explicit bernoulli for zeros
do_sample_in_train = False and explicit_zero_prob  # sample the bernoulli in training

per_seq_batch_sample = False

# =========================
# Settings for optimizer
# =========================
lr = config["lr"]
# lr_ADV = 1e-3  # learning rate for discriminator, used when ADV is True
batch_size = config["batch_size"]
eval_batch_size = config["batch_size"]
epochs = config["epochs"]
schedule_interval = 1  # interval (in epochs) to update learning rate according to schedule

# =========================
# Settings for the model
# =========================
fast_transformer = config["fast_transformer"]  # Whether to use a faster Transformer implementation
fast_transformer_backend = "flash"  # "linear" or "flash"
embsize = config["layer_size"]  # embedding dimension
d_hid = config["layer_size"]  # dimension of the feedforward network in nn.TransformerEncoder
nlayers = config["nlayers"]  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = config["nhead"]  # number of heads in nn.MultiheadAttention
dropout = config["dropout"]  # dropout probability


# =========================
# Logging
# =========================
log_interval = 100  # iterations
save_eval_interval = config["save_eval_interval"]  # epochs
do_eval_scib_metrics = True
mask_value = -1
pad_value = -2
n_input_bins = n_bins

# =========================
# Validate settings
# =========================
assert input_style in ["normed_raw", "logip", "binned"]
assert output_style in ["normed_raw", "logip", "binned"]
assert input_emb_style in ["category", "continuous", "scaling"]
if input_style == "binned":
    if input_emb_style == "scaling":
        raise ValueError("input_emb_style `scaling` is not supported for binned input.")
elif input_style == "logip" or input_style == "normed_raw":
    if input_emb_style == "category":
        raise ValueError(
            "input_emb_style `category` is not supported for logip or normed_raw input."
        )

if input_emb_style == "category":
    mask_value = n_bins + 1
    pad_value = n_bins  # for padding gene expr values
    n_input_bins = n_bins + 2
else:
    mask_value = -1
    pad_value = -2
    n_input_bins = n_bins

if ADV and DAB:
    raise ValueError("ADV and DAB cannot be both True.")
DAB_separate_optim = True if DAB > 1 else False


# =========================
# dataset and save directory setup
# =========================
dataset_name = config.dataset_name
save_dir = Path(f"./save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")
logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")

# =========================
# DATA LOADER
# =========================

# Simple dataset loading and splitting
dataset_path = config.dataset_path
balance_type = config.balance_type
metadata_path = getattr(config, 'metadata_path', None)
gene_list_path = getattr(config, 'gene_list_path', None)

print(f" Dataset configuration:")
print(f"  Path: {dataset_path}")
print(f"  Balance type: {balance_type}")
print(f"  Metadata: {metadata_path}")
print(f"  Gene list: {gene_list_path}")
print(f"-"*60)

# Load dataset
adata_full, ori_batch_col, data_is_raw = load_dataset_simple(
    dataset_path,
    metadata_path=metadata_path,
    gene_list_path=gene_list_path
)

# Create smart splits
splitter = DatasetSplitter(balance_type=balance_type)
adata_train_raw, adata_val_raw, adata_test_raw, used_stratify = splitter.analyze_and_split(adata_full)

print(f" Final result: Used {'STRATIFIED' if used_stratify else 'RANDOM'} split")

# Create consistent celltype encoding for all splits
celltype_categories = adata_full.obs["celltype"].astype("category").cat.categories
global_celltype_categories = celltype_categories
num_types = len(celltype_categories)
id2type = dict(enumerate(celltype_categories))

print(f" Ready for training with {num_types} cell types")

# =========================
# PREPROCESSING: TRAIN FIRST, THEN APPLY TO VAL/TEST
# =========================

print("\n" + "="*60)
print("PREPROCESSING: TRAINING DATA ONLY")
print("="*60)

# Preprocess ONLY training data
adata_train = adata_train_raw.copy()

# Make batch columns consistent
adata_train.obs["str_batch"] = adata_train.obs[ori_batch_col].astype(str)
adata_train.obs["batch_id"] = adata_train.obs["str_batch"].astype("category").cat.codes.values

# Create consistent celltype encoding
global_celltype_categories = celltype_categories
adata_train.obs["celltype"] = adata_train.obs["celltype"].astype("category").cat.set_categories(global_celltype_categories)
adata_train.obs["celltype_id"] = adata_train.obs["celltype"].cat.codes.values

print("Training data before preprocessing: (34044, 66587)")

# Preprocess ONLY training data
preprocessor_train = Preprocessor(
    use_key="X",  # Raw counts
    filter_gene_by_counts=3,  # Filter genes with <3 counts
    filter_cell_by_counts=False,
    normalize_total=1e4,  # Normalize to 10k reads per cell
    result_normed_key="X_normed",
    log1p=data_is_raw,  # Log1p transform
    result_log1p_key="X_logip",
    subset_hvg=n_hvg,  # Select 1200 HVGs - ONLY from training data!
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=config.n_bins,  # Bin expression values
    result_binned_key="X_binned",
)

# Apply preprocessing to training data ONLY
preprocessor_train(adata_train, batch_key="str_batch" if dataset_name != "heart_cell" else None)

print(f"Training data after preprocessing: {adata_train.shape}")
print(f"Selected genes (HVGs): {len(adata_train.var_names)}")

# Store the genes selected from TRAINING data only
selected_genes_from_training = adata_train.var_names.copy()

print("\n" + "="*60)
print("APPLYING SAME PREPROCESSING TO VALIDATION/TEST")
print("="*60)


def apply_training_preprocessing_to_split(adata_split, split_name, selected_genes):
    """Apply same preprocessing as training, using training-derived parameters"""
    
    print(f"\nProcessing {split_name} set...")
    print(f"  Before gene selection: {adata_split.shape}")
    
    # Make batch columns consistent
    adata_split.obs["str_batch"] = adata_split.obs[ori_batch_col].astype(str)
    adata_split.obs["batch_id"] = adata_split.obs["str_batch"].astype("category").cat.codes.values
    
    # Apply consistent celltype encoding
    adata_split.obs["celltype"] = adata_split.obs["celltype"].astype("category").cat.set_categories(celltype_categories)
    adata_split.obs["celltype_id"] = adata_split.obs["celltype"].cat.codes.values
    
    # Use SAME genes as selected from training (no re-selection!)
    common_genes = adata_split.var_names.intersection(selected_genes)
    print(f"  Common genes with training: {len(common_genes)}/{len(selected_genes)}")
    
    adata_split = adata_split[:, common_genes].copy()
    print(f"  After gene selection: {adata_split.shape}")
    
    # Apply preprocessing (no HVG selection, no gene filtering)
    preprocessor_split = Preprocessor(
        use_key="X",
        filter_gene_by_counts=False,  # Don't filter - already have the right genes
        filter_cell_by_counts=False,
        normalize_total=1e4,  # Same normalization
        result_normed_key="X_normed",
        log1p=data_is_raw,  # Same Logip
        result_log1p_key="X_logip",
        subset_hvg=False,  # CRITICAL: Don't re-select HVGs!
        binning=config.n_bins,  # Same binning
        result_binned_key="X_binned",
    )
    
    # Apply preprocessing
    preprocessor_split(adata_split, batch_key="str_batch" if dataset_name != "heart_cell" else None)
    
    print(f"  Final {split_name} shape: {adata_split.shape}")
    
    return adata_split


# Apply to validation and test sets
adata_val = apply_training_preprocessing_to_split(adata_val_raw, "validation", selected_genes_from_training)
adata_test = apply_training_preprocessing_to_split(adata_test_raw, "test", selected_genes_from_training)

print("\n" + "="*60)
print("VERIFICATION: CONSISTENT PREPROCESSING")
print("="*60)

# Verify all splits have same genes
assert len(adata_train.var_names) == len(adata_val.var_names) == len(adata_test.var_names), "Train-Val gene mismatch!"
assert (adata_train.var_names == adata_val.var_names).all(), "Train-Val gene mismatch!"
assert (adata_train.var_names == adata_test.var_names).all(), "Train-Test gene mismatch!"

print(" All splits have consistent genes")
print(f"  Train: {adata_train.shape}, Val: {adata_val.shape}, Test: {adata_test.shape}")

# Check that all splits have required layers
required_layers = ["X_normed", "X_logip", "X_binned"]
for split_name, adata_split in [("Train", adata_train), ("Val", adata_val), ("Test", adata_test)]:
    missing_layers = [layer for layer in required_layers if layer not in adata_split.layers]
    if missing_layers:
        print(f"  {split_name} missing layers: {missing_layers}")
    else:
        print(f"  {split_name} has all required layers")

print("\n" + "="*60)
print("DATA LOADING COMPLETE - NO DATA LEAKAGE!")
print("="*60)

# Update global variables to use the training set
adata = adata_train  # This will be used for the rest of the pipeline

# Create global variables for consistent access
celltypes = global_celltype_categories
num_types = len(global_celltype_categories)
id2type = dict(enumerate(global_celltype_categories))

print(f"Number of cell types: {num_types}")
print(f"ID to type mapping: {id2type}")

# ── SAVE INFERENCE ARTIFACTS ──────────────────────────────────────────────────
# id2type is now finalized: {0: 'Astrocyte', 1: 'Endothelial', ...}
# adata_train.var_names holds the HVG gene list after preprocessing

# 1. HVG gene list
hvg_genes_l2 = adata_train.var_names.tolist()
with open(save_dir / "hvg_genes.pkl", "wb") as f:
    pickle.dump(hvg_genes_l2, f)
print(f"✓ Saved hvg_genes.pkl ({len(hvg_genes_l2)} genes)")

# 2. Label mapping (int → cell type name)
with open(save_dir / "id_to_label.json", "w") as f:
    json.dump(id2type, f, indent=2)
print(f"✓ Saved id_to_label.json: {id2type}")

# 3. Taxonomy (cell type → family, for hierarchical rejection)
taxonomy_l2 = {
    "Astrocyte":         {"family": "Glial"},
    "Microglia":         {"family": "Immune"},
    "Macrophage":        {"family": "Immune"},
    "Oligodendrocyte":   {"family": "Glial"},
    "OPC":               {"family": "Glial"},
    "Excitatory":        {"family": "Neuronal"},
    "Inhibitory":        {"family": "Neuronal"},
    "Endothelial":       {"family": "Vascular"},
    "Malignant_GBM":     {"family": "Tumor"},
    "T-Cell/Lymph.":     {"family": "Immune"},
    "Neuron":            {"family": "Neuronal"},
    "GBM":               {"family": "Tumor"},
}
# Only keep entries that are actually in your training data
taxonomy_l2 = {k: v for k, v in taxonomy_l2.items() if k in celltypes}
with open(save_dir / "taxonomy.json", "w") as f:
    json.dump(taxonomy_l2, f, indent=2)
print(f"✓ Saved taxonomy.json — classes: {list(taxonomy_l2.keys())}")
# ─────────────────────────────────────────────────────────────────────────────

# =========================
# DEVICE SETUP
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# =========================
# CLASS WEIGHTING FOR IMBALANCED DATASETS
# =========================

# Calculate class weights from training data

train_labels = adata.obs["celltype_id"].values
class_weights = compute_class_weight(
    'balanced',
    classes=np.arange(num_types),
    y=train_labels
)
class_weights_tensor = torch.FloatTensor(class_weights).to(device)

print("\n" + "="*60)
print(f"\nClass weights:")
for i, (cell_type, weight) in enumerate(zip(celltypes, class_weights)):
    count = (train_labels == i).sum()
    print(f"  {cell_type:20s}: weight={weight:.3f}, count={count:5d}")
print("=" * 60 + "\n")

# Balanced Batch Sampling

from torch.utils.data import WeightedRandomSampler

print("\n" + "="*60)
print("BALANCED BATCH SAMPLING SETUP")
print("="*60)

# Calculate sample weights for balanced sampling
train_labels = adata_train.obs["celltype_id"].values    
class_counts = np.bincount(train_labels)
class_weights_sampling = 1.0 / class_counts
sample_weights = class_weights_sampling[train_labels]

# Create weighted sampler
train_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)
print("\nSampling Strategy:")
print(f"  Total training samples: {len(sample_weights)}")
print(f" Rare classes will be oversampled to balance the dataset.")
print(f" Common classes will be undersampled accordingly.")
print("\nExpected samples per class in one epoch:")
target_per_class = len(sample_weights) / num_types
for i, cell_type in enumerate(celltypes):
    original_count = class_counts[i]
    oversample_factor = target_per_class / original_count
    print(f" {cell_type:20s}: original={original_count:5d} -> ~{int(target_per_class):5d} ({oversample_factor:.2f}x)")

print("=" * 60 + "\n")

# =========================
# TOKENIZATION AND DATA PREPARATION
# =========================

print("\n" + "="*60)
print("TOKENIZATION AND DATA PREPARATION")
print("="*60)

# Load or create vocabulary
if config.load_model is not None:
    model_dir = Path(config.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"
    
    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    
    # Check gene overlap with vocabulary (using training genes only)
    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    
    # Filter to genes in vocabulary
    genes_in_vocab = adata.var["id_in_vocab"] >= 0
    adata = adata[:, genes_in_vocab]
    
    # Apply same gene filtering to val and test
    adata_val = adata_val[:, genes_in_vocab]
    adata_test = adata_test[:, genes_in_vocab]
    
    # Load model config
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    logger.info(
        f"Resume model from {model_file}, the model args will be overridden by the "
        f"config {model_config_file}."
    )
    
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]  # Note: 'nheads' not 'nhead' in config
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]
else:
    embsize = config.layer_size
    nhead = config.nhead
    d_hid = config.layer_size
    nlayers = config.nlayers
    n_layers_cls = 3

# Create vocabulary if not loading pretrained
if config.load_model is None:
    vocab = Vocab(
        VocabPybind(adata.var["gene_name"].tolist() + special_tokens, None)
    )
vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(vocab(adata.var["gene_name"].tolist()), dtype=int)

print(f"Vocabulary size: {len(vocab)}")
print(f"Gene IDs shape: {gene_ids.shape}")

# =========================
# PREPARE DATA FOR TRAINING
# =========================

# Get the input layer
input_layer_key = {
    "normed_raw": "X_normed",
    "log1p": "X_normed",
    "binned": "X_binned",
}[input_style]

print(f"Using input layer: {input_layer_key}")

# Extract data from each split
def extract_data_from_split(adata_split, split_name):
    """Extract expression data and labels from a split"""
      
    # Get expression data
    counts = (
        adata_split.layers[input_layer_key].A
        if issparse(adata_split.layers[input_layer_key])
        else adata_split.layers[input_layer_key]
    )
    
    # Get labels
    celltype_labels = np.array(adata_split.obs["celltype_id"].tolist())
    batch_ids = np.array(adata_split.obs["batch_id"].tolist())
    
    print(f" {split_name} data summary:")
    print(f"  Expression shape: {counts.shape}")
    print(f"  Celltype labels: {len(celltype_labels)} (range: {celltype_labels.min()}-{celltype_labels.max()})")
    print(f"  Batch IDs: {len(batch_ids)} (range: {batch_ids.min()}-{batch_ids.max()})")
    print(f"-"*60)

    return counts, celltype_labels, batch_ids


# Extract data from all splits
train_data, train_celltype_labels, train_batch_labels = extract_data_from_split(adata, "Train")
valid_data, valid_celltype_labels, valid_batch_labels = extract_data_from_split(adata_val, "Val")
test_data, test_celltype_labels, test_batch_labels = extract_data_from_split(adata_test, "Test")

# Calculate batch statistics from training data only
batch_ids = train_batch_labels  # Use training batch IDs
num_batch_types = len(set(batch_ids))
print(f"Number of unique batch types (from training): {num_batch_types}")

# Validation checks
print("\n" + "="*60)
print("Data validation:")
print(f" Train feature dimensions match: {train_data.shape[1] == valid_data.shape[1] ==  test_data.shape[1]}")
print(f" All celltype labels in valid range: {max(train_celltype_labels) < num_types and max(test_celltype_labels) < num_types}")
print(f" Gene count: {len(adata.var['gene_name'])}")
print(f"-"*60)

# Store genes list
genes = adata.var["gene_name"].tolist()

print(f"\n Final data summary:")
print(f"Training: {train_data.shape} samples")
print(f"Validation: {valid_data.shape} samples")
print(f"Test: {test_data.shape} samples")
print(f"Features: {train_data.shape[1]}")
print(f"Cell types: {num_types}")
print(f"Batch types: {num_batch_types}")
print(f"-"*60)

# =========================
# TOKENIZATION
# =========================

print("\nTokenizing data...")

# Tokenize all splits
tokenized_train = tokenize_and_pad_batch(
    train_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,
    include_zero_gene=include_zero_gene,
)

tokenized_valid = tokenize_and_pad_batch(
    valid_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,
    include_zero_gene=include_zero_gene,
)

tokenized_test = tokenize_and_pad_batch(
    test_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,
    include_zero_gene=include_zero_gene,
)

logger.info(
    f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
)
logger.info(
    f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
)

print(" Tokenization complete!")
print(" No data leakage - test set completely isolated!")

# =========================
# FUNCTION DEFINITIONS
# =========================

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
        f"random masking at epoch, ratio of masked values in train: "
        f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}"
    )
    
    input_gene_ids_train, input_gene_ids_valid = (
        tokenized_train["genes"],
        tokenized_valid["genes"],
    )
    input_values_train, input_values_valid = masked_values_train, masked_values_valid
    target_values_train, target_values_valid = (
        tokenized_train["values"],
        tokenized_valid["values"],
    )
    tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
    tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()
    
    tensor_celltype_labels_train = torch.from_numpy(train_celltype_labels).long()
    tensor_celltype_labels_valid = torch.from_numpy(valid_celltype_labels).long()
    
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


# dataset
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data
    
    def __len__(self):
        return self.data["gene_ids"].shape[0]
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


# data_loader
def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
    sampler = None,  # <- NEW: Add sampler parameter
) -> DataLoader:
    
    if num_workers == 0:
        num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)
    
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
    
    # Standard dataloader with optional class balancing
    # ↓ MODIFIED: Use sampler if provided
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,  # <- Can't shuffle with sampler
        sampler=sampler,  # <- NEW: Add sampler
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader


def train(model: nn.Module, loader: DataLoader) -> Dict:  # ← CHANGED: Returns Dict now
    """
    Train the model for one epoch.
    """
    model.train()
    (
        total_loss,
        total_mse,
        total_cls,
        total_cce,
        total_mvc,
        total_ecs,
        total_dab,
        total_adv_E,
        total_adv_D,
        total_zero_log_prob,
        total_mvc_zero_log_prob,
    ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    total_error = 0.0
    
    # ===== NEW: Add these tracking variables =====
    epoch_total_loss = 0.0
    epoch_total_correct = 0
    epoch_total_samples = 0
    # =============================================
    
    start_time = time.time()
    
    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
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
                CLS=CLS,  
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
                do_sample=do_sample_in_train,
            )
            
            masked_positions = input_values.eq(mask_value)
            loss = 0.0
            metrics_to_log = {}
            
            # Initialize loss components
            loss_cls = torch.tensor(0.0).to(device)
            loss_mse = torch.tensor(0.0).to(device)
            loss_zero_log_prob = torch.tensor(0.0).to(device)
            loss_cce = torch.tensor(0.0).to(device)
            loss_mvc = torch.tensor(0.0).to(device)
            loss_ecs = torch.tensor(0.0).to(device)
            loss_dab = torch.tensor(0.0).to(device)
            loss_adv_E = torch.tensor(0.0).to(device)
            loss_adv_D = torch.tensor(0.0).to(device)
            error_rate = 0.0
            
            if MLM:
                loss_mse = criterion(output_dict["mlm_output"], target_values, masked_positions)
                loss += loss_mse
                metrics_to_log.update({"train/mse": loss_mse.item()})
                error_rate = masked_relative_error(
                    output_dict["mlm_output"], target_values, masked_positions
                )
            
            if CLS:
                loss_cls = F.cross_entropy(
                    output_dict["cls_output"],
                    celltype_labels,
                    weight=class_weights_tensor
                )
                loss += loss_cls
                metrics_to_log.update({"train/cls": loss_cls.item()})
                
                # ===== Calculate batch train accuracy =====
                with torch.no_grad():
                    train_preds = torch.argmax(output_dict["cls_output"], dim=1)
                    train_correct = (train_preds == celltype_labels).sum().item()
                    train_acc = train_correct / len(celltype_labels)
                    metrics_to_log["train/acc"] = train_acc
                    
                    # ===== NEW: Accumulate for epoch-level metrics =====
                    epoch_total_correct += train_correct
                    epoch_total_samples += len(celltype_labels)
                    # ==================================================
            
            if MVC:
                loss_mvc = criterion(output_dict["mvc_output"], target_values, masked_positions)
                loss += loss_mvc
                metrics_to_log.update({"train/mvc": loss_mvc.item()})
            
            if ECS:
                loss_ecs = output_dict["loss_ecs"]
                loss += loss_ecs
                metrics_to_log.update({"train/ecs": loss_ecs.item()})
            
            if CCE:
                loss_cce = output_dict["loss_cce"]
                loss += loss_cce
                metrics_to_log.update({"train/cce": loss_cce.item()})
            
            if DAB:
                loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
                loss += dab_weight * loss_dab
                metrics_to_log.update({"train/dab": loss_dab.item()})
            
            if explicit_zero_prob:
                loss_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["zero_logits"], target_values, masked_positions
                )
                loss += loss_zero_log_prob
                metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
                
                if MVC:
                    loss_mvc_zero_log_prob = criterion_neg_log_bernoulli(
                        output_dict["mvc_zero_logits"], target_values, masked_positions
                    )
                    loss += loss_mvc_zero_log_prob
                    metrics_to_log.update({"train/mvc_nzlp": loss_mvc_zero_log_prob.item()})
        
        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    f"can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()
        
        if 'wandb' in globals() and wandb.run is not None:
            wandb.log(metrics_to_log)
        
        # ===== NEW: Accumulate epoch loss =====
        epoch_total_loss += loss.item()
        # ======================================
        
        # logging code
        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_cls += loss_cls.item()
        total_cce += loss_cce.item()
        total_mvc += loss_mvc.item()
        total_ecs += loss_ecs.item()
        total_dab += loss_dab.item()
        total_zero_log_prob += loss_zero_log_prob.item()
        if MVC and explicit_zero_prob:
            total_mvc_zero_log_prob += loss_mvc_zero_log_prob.item()
        total_error += error_rate
        
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval if MLM else 0.0
            cur_cls = total_cls / log_interval if CLS else 0.0
            cur_cce = total_cce / log_interval if CCE else 0.0
            cur_mvc = total_mvc / log_interval if MVC else 0.0
            cur_ecs = total_ecs / log_interval if ECS else 0.0
            cur_dab = total_dab / log_interval if DAB else 0.0
            cur_zero_log_prob = (
                total_zero_log_prob / log_interval if explicit_zero_prob else 0.0
            )
            cur_mvc_zero_log_prob = (
                total_mvc_zero_log_prob / log_interval 
                if MVC and explicit_zero_prob 
                else 0.0
            )
            cur_error = total_error / log_interval
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | "
                + (f"mse {cur_mse:5.2f} | mre {cur_error:5.2f} |" if MLM else "")
                + (f"cls {cur_cls:5.2f} | " if CLS else "")
                + (f"err {cur_error:5.2f} | " if CLS else "")
                + (f"cce {cur_cce:5.2f} |" if CCE else "")
                + (f"mvc {cur_mvc:5.2f} |" if MVC else "")
                + (f"ecs {cur_ecs:5.2f} |" if ECS else "")
                + (f"dab {cur_dab:5.2f} |" if DAB else "")
                + (f"adv_E {total_adv_E:5.2f} |" if ADV else "")
                + (f"adv_D {total_adv_D:5.2f} |" if ADV else "")
                + (f"nzlp {cur_zero_log_prob:5.2f} |" if explicit_zero_prob else "")
                + (
                    f"mvc_nzlp {cur_mvc_zero_log_prob:5.2f} |"
                    if MVC and explicit_zero_prob
                    else ""
                )
            )
            total_loss = 0
            total_mse = 0
            total_cls = 0
            total_cce = 0
            total_mvc = 0
            total_ecs = 0
            total_dab = 0
            total_adv_E = 0
            total_adv_D = 0
            total_zero_log_prob = 0
            total_mvc_zero_log_prob = 0
            total_error = 0
            start_time = time.time()
    
    # ===== NEW: Calculate and return epoch-level metrics =====
    epoch_loss = epoch_total_loss / num_batches
    epoch_acc = epoch_total_correct / epoch_total_samples if epoch_total_samples > 0 else 0.0
    
    return {
        'train_loss': epoch_loss,
        'train_acc': epoch_acc
    }
    # ========================================================


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
            
            loss_dab = torch.tensor(0.0).to(device)
            if DAB:
                loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
            
            total_loss += loss.item() * len(input_gene_ids)
            accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
            total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
            total_dab += loss_dab.item() * len(input_gene_ids) if DAB else 0.0
            total_num += len(input_gene_ids)
            preds = output_values.argmax(1).cpu().numpy()
            predictions.append(preds)
    
    if return_raw:
        return np.concatenate(predictions, axis=0)
    
    return total_loss / total_num, total_error / total_num

# =========================
# HIERARCHICAL REJECTION ANALYSIS
# =========================

class HierarchicalRejectClassifier:
    """
    Hierarchical reject system with dynamic cell type mapping.
    Works with any brain cell dataset by inferring hierarchy from cell type names.
    """
    
    def __init__(self, cell_types: List[str] = None):
        """
        Args:
            cell_types: List of cell type names from the dataset (e.g., from adata.obs['CellType'])
        """
        if cell_types is None:
            # Default: use all known cell types from training
            cell_types = self._get_all_known_cell_types()
        
        self.cell_type_mapping = self._build_dynamic_mapping(cell_types)
    
    def _get_all_known_cell_types(self):
        """Return all cell types the model was trained on."""
        return [
            # Old dataset (21 types)
            'TCells.memory', 'NK.Cells', 'TCells.helper', 'TCells.activated',
            'Plasmacells', 'Fibroblasts', 'SC.Like', 'M2.Macrophages',
            'Monocytes', 'M1.Macrophages', 'pDCs', 'Microglia.Inflamatory',
            'Microglia.Protective', 'VascularEndothel', 'BCells', 'Pericytes',
            'Oligodendrocytes', 'Neurons', 'OPCs', 'Astrocytes', 'SMC',
            # New dataset (8 types) - add
            'Neuron', 'GBM', 'Oligodendrocyte', 'Microglia',
            'Macrophage', 'Astrocyte', 'Endothelial', 'T-Cell/Lymph.'
        ]
    
    def _build_dynamic_mapping(self, cell_types: List[str]) -> Dict:
        """
        Automatically map cell types to family/category based on keywords.
        """
        
        # Define keyword-based rules for hierarchy
        hierarchy_rules = {
            # Brain cells
            'Neuron': ('Neural cells', 'Brain cells'),
            'Oligodendrocyte': ('Glial cells', 'Brain cells'),
            'Astrocyte': ('Glial cells', 'Brain cells'),
            'OPC': ('Glial precursors', 'Brain cells'),
            'Microglia': ('Microglia', 'Brain cells'),
            
            # Malignant cells
            'GBM': ('Brain tumors', 'Malignant cells'),
            'Glioma': ('Brain tumors', 'Malignant cells'),
            'Tumor': ('Tumors', 'Malignant cells'),
            'Cancer': ('Cancer cells', 'Malignant cells'),
            'Malignant': ('Malignant cells', 'Malignant cells'),
            
            # Immune cells
            'TCell': ('T lymphocytes', 'Immune cells'),
            'T-Cell': ('T lymphocytes', 'Immune cells'),
            'BCell': ('B lymphocytes', 'Immune cells'),
            'NK': ('Natural killer cells', 'Immune cells'),
            'Macrophage': ('Macrophages', 'Immune cells'),
            'M1': ('Macrophages', 'Immune cells'),
            'M2': ('Macrophages', 'Immune cells'),
            'Monocyte': ('Monocytes', 'Immune cells'),
            'Dendritic': ('Dendritic cells', 'Immune cells'),
            'pDC': ('Dendritic cells', 'Immune cells'),
            'Plasma': ('B lymphocytes', 'Immune cells'),
            'Lymph': ('Lymphocytes', 'Immune cells'),
            
            # Vascular cells
            'Endothel': ('Endothelial cells', 'Vascular cells'),
            'Pericyte': ('Vascular support cells', 'Vascular cells'),
            'SMC': ('Vascular support cells', 'Vascular cells'),
            
            # Stromal cells
            'Fibroblast': ('Stromal cells', 'Stromal cells'),
            'SC.Like': ('Stem cells', 'Stem cells'),
            'Stem': ('Stem cells', 'Stem cells'),
        }
        
        mapping = {}
        for idx, cell_type in enumerate(cell_types):
            family, category = self._classify_cell_type(cell_type, hierarchy_rules)
            mapping[idx] = {
                'name': cell_type,
                'family': family,
                'category': category
            }
        
        return mapping
    
    def _classify_cell_type(self, cell_type: str, rules: Dict) -> Tuple[str, str]:
        """
        Match cell type name to hierarchy using keyword rules.
        """
        cell_type_lower = cell_type.lower()
        
        # Check each rule
        for keyword, (family, category) in rules.items():
            if keyword.lower() in cell_type_lower:
                return family, category
        
        # Default: Unknown
        return 'Unclassified', 'Unknown'
    
    def predict_with_hierarchy(self, model_output: torch.Tensor) -> List[Dict]:
        probabilities = F.softmax(model_output, dim=1)
        confidences, predicted_classes = torch.max(probabilities, dim=1)
        
        results = []
        for i, (confidence, predicted_class) in enumerate(zip(confidences, predicted_classes)):
            conf_val = confidence.item()
            class_idx = predicted_class.item()
            
            cell_info = self.cell_type_mapping.get(class_idx, {
                'name': f'Unknown_class_{class_idx}',
                'family': 'Unknown',
                'category': 'Unknown'
            })
            
            category_confidence = self._calculate_category_confidence(probabilities[i], cell_info)
            family_confidence = self._calculate_family_confidence(probabilities[i], cell_info)
            
            prediction_result = {
                'cell_id': i,
                'confidence': conf_val,
                'raw_prediction': cell_info['name'],
                'decision_level': None,
                'final_prediction': None,
                'reasoning': None
            }
            
            if conf_val >= 0.80:  # High confidence
                prediction_result.update({
                    'decision_level': 0,
                    'final_prediction': f"{cell_info['name']} ({conf_val:.0%} confident)",
                    'reasoning': f"High confidence ({conf_val:.1%}) - specific classification"
                })
            elif family_confidence >= 0.65:  # Family level
                prediction_result.update({
                    'decision_level': 1,
                    'final_prediction': f"{cell_info['family']} ({family_confidence:.0%} confident)",
                    'reasoning': f"Moderate confidence ({family_confidence:.1%}) - family level classification"
                })
            elif category_confidence >= 0.65:  # Family level
                prediction_result.update({
                    'decision_level': 2,
                    'final_prediction': f"{cell_info['category']} ({category_confidence:.0%} confident)",
                    'reasoning': f"Partial rejection to category level ({category_confidence:.1%}) - category level classification"
                })
            else:
                prediction_result.update({
                    'decision_level': 3,
                    'final_prediction': "Rejected",
                    'reasoning': f"Low confidence ({conf_val:.1%}) - prediction rejected"
                })
            results.append(prediction_result)

        return results
    
    def _calculate_category_confidence(self, probabilities, target_category):
        category_prob_sum = 0.0
        for class_idx, prob in enumerate(probabilities):
            if class_idx in self.cell_type_mapping:
                if self.cell_type_mapping[class_idx]['category'] == target_category:
                    category_prob_sum += prob.item()
        return category_prob_sum
    
    def _calculate_family_confidence(self, probabilities, target_category): 
        family_prob_sum = 0.0
        for class_idx, prob in enumerate(probabilities):
            if class_idx in self.cell_type_mapping:
                if self.cell_type_mapping[class_idx]['family'] == target_category:
                    family_prob_sum += prob.item()
        return family_prob_sum
    
    def evaluate_hierarchical_performance(self, model_output: torch.Tensor, true_labels: torch.Tensor) -> Dict:
        predictions = self.predict_with_hierarchy(model_output)
        level_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # specific, family, broad, rejected
        correct_at_level = {0: 0, 1: 0, 2:0, 3: 0}

        for i, pred in enumerate(predictions):
            true_class = true_labels[i].item()
            true_info = self.cell_type_mapping.get(true_class, {})
            level = pred['decision_level']
            level_counts[level] += 1
            final_pred = pred['final_prediction']

            if level == 0:
                clean_pred = pred['raw_prediction']
                if clean_pred == true_info.get('name'):
                    correct_at_level[0] += 1
            elif level == 1:
                clean_family = final_pred.split('(')[0].strip() if '(' in final_pred else final_pred
                if clean_family == true_info.get('family'):
                    correct_at_level[1] += 1
            elif level == 2:
                clean_category = final_pred.split('(')[0].strip() if '(' in final_pred else final_pred
                if clean_category == true_info.get('category'):
                    correct_at_level[2] += 1
            # elif level == 3: rejected predictions are never "correct"
        
        total_cells = len(predictions)
        results = {
            'total_cells': total_cells,
            'level_distribution': {
                'specific_predictions': level_counts[0],
                'family_predictions': level_counts[1],
                'broad_predictions': level_counts[2],
                'rejected_predictions': level_counts[3], 
                'specific_pct': level_counts[0] / total_cells * 100,
                'family_pct': level_counts[1] / total_cells * 100,
                'broad_pct': level_counts[2] / total_cells * 100,
                'rejected_pct': level_counts[3] / total_cells * 100,
            },
            'accuracy_by_level': {
                'specific_accuracy': correct_at_level[0] / max(level_counts[0], 1) * 100,
                'family_accuracy': correct_at_level[1] / max(level_counts[1], 1) * 100,
                'broad_accuracy': correct_at_level[2] / max(level_counts[2], 1) * 100,
            },
            'overall_hierarchical_accuracy': sum(correct_at_level.values()) / total_cells,
        }
        
        return results
    
    def print_sample_results(self, predictions: List[Dict], n_samples: int = 10):
        print(f"\n=== Sample Hierarchical Predictions (first {n_samples}) ===")
        for i, pred in enumerate(predictions[:n_samples]):
            level_map = {0: 'Specific', 1: 'Family', 2: 'Broad', 3: 'Rejected'}
            level_name = level_map[pred['decision_level']]
            print(f"Cell {pred['cell_id']:3d}: {pred['final_prediction']:20s} "
                  f"[{level_name}] - {pred['reasoning']}")


def integrate_hierarchical_with_evaluation(model, test_loader, device):
    hierarchical_classifier = HierarchicalRejectClassifier()
    model.eval()
    all_logits, all_labels, all_predictions_standard = [], [], []
    total_error, total_loss = 0.0, 0.0
    
    with torch.no_grad():
        for batch_data in test_loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            
            with autocast(): 
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                    CLS=True
                )
            
            logits = output_dict["cls_output"]
            loss_cls = criterion_cls(logits, celltype_labels)
            total_loss += loss_cls.item()
            
            error_rate = 1 - (
                (logits.argmax(1) == celltype_labels).sum().item()
                 )/ celltype_labels.size(0)
            total_error += error_rate
            
            all_logits.append(logits.cpu())
            all_labels.extend(celltype_labels.cpu().numpy())
            standard_preds = logits.argmax(1)
            all_predictions_standard.extend(standard_preds.cpu().numpy())
    
    num_batches = len(test_loader)
    val_loss = total_loss / num_batches
    val_err = total_error / num_batches
    all_logits_tensor = torch.cat(all_logits, dim=0)
    all_labels_tensor = torch.tensor(all_labels)

    # Convert FP16 to FP32 for hierarchical classifier
    all_logits_tensor = all_logits_tensor.float()
    
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
    print(f"Specific predictions: {dist['specific_pct']:.1f}%")
    print(f"Family predictions:   {dist['family_pct']:.1f}%")
    print(f"Broad predictions:    {dist['broad_pct']:.1f}%")
    
    acc = hierarchical_results['accuracy_by_level']
    print("\n--- Accuracy by Confidence Level ---")
    print(f"Specific accuracy:  {acc['specific_accuracy']:.1f}%")
    print(f"Family accuracy:    {acc['family_accuracy']:.1f}%")
    print(f"Broad accuracy:     {acc['broad_accuracy']:.1f}%")
    
    hierarchical_classifier.print_sample_results(hierarchical_predictions, n_samples=15)
    analyze_confidence_by_cell_type(hierarchical_predictions, all_labels)
    #analyze_confidence_by_cell_type(hierarchical_predictions, all_labels, hierarchical_classifier)
    
    return val_loss, val_err, hierarchical_results, hierarchical_predictions


#def analyze_confidence_by_cell_type(predictions, true_labels, hierarchical_classifier, cell_types):
def analyze_confidence_by_cell_type(predictions, true_labels):
    """
    Analyze confidence and rejection patterns by cell type.
    
    Args:
        predictions: List of prediction dictionaries from hierarchical classifier
        true_labels: Tensor of true labels
        hierarchical_classifier: The HierarchicalRejectClassifier instance
    """

    confidence_by_type = {}
    decision_level_by_type = {}

    for pred, true_label in zip(predictions, true_labels):
        cell_type = pred ['raw_prediction']
        if cell_type not in confidence_by_type:
            confidence_by_type[cell_type] = []
            decision_level_by_type[cell_type] = []
        confidence_by_type[cell_type].append(pred['confidence'])
        decision_level_by_type[cell_type].append(pred['decision_level'])
    
    print("\n" + "="*60)
    print("CONFIDENCE ANALYSIS BY CELL TYPE")
    print("="*60)

    avg_confidence_by_type = {}
    high_conf_pct_by_type = {}
    for cell_type, confidences in confidence_by_type.items():
        avg_conf = np.mean(confidences)
        high_conf_count = sum(1 for c in confidences if c >=0.85)
        high_conf_pct = high_conf_count / len(confidences)*100
        avg_confidence_by_type[cell_type] = avg_conf
        high_conf_pct_by_type[cell_type] = high_conf_pct

    sorted_by_confidence = sorted(avg_confidence_by_type.items(), key=lambda x: x[1], reverse=True)
    print("Cell Type Rankings (by average confidence):")
    print("-" * 50)
    for i, (cell_type, avg_conf) in enumerate(sorted_by_confidence[:10]):
        high_pct = high_conf_pct_by_type[cell_type]
        print(f"{i+1:2d}. {cell_type:20s}: {avg_conf:.1%} avg conf, {high_pct:.0f}% high conf")
    
    print("\nLowest confidence cell types:")
    print("-" * 30)
    for cell_type, avg_conf in sorted_by_confidence[-5:]:
        high_pct = high_conf_pct_by_type[cell_type]
        print(f"{cell_type:20s}: {avg_conf:.1%} avg conf, {high_pct:.0f}% high conf")

# Model Creation 

if config.load_model is not None:
    n_layers_cls = model_configs.get("n_layers_cls", 3) #use from loaded config
else:
    n_layers_cls = 3 # default

ntokens = len(vocab) #size of vocab
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
    #use_gradient_checkpointing=True,
    fast_transformer_backend=fast_transformer_backend,
    pre_norm=config["pre_norm"],
)

if config["load_model"] is not None:
    try:
        model.load_state_dict(torch.load(model_file))
        logger.info(f"Loading all model params from {model_file}")
    except:
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            logger.info(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

pre_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

# Freeze all pre-decoder weights
for name, para in model.named_parameters():
    print("-"*20)
    print(f"name: {name}")
    if config["freeze"] and "encoder" in name and "transformer_encoder" not in name:
        print(f"freezing weights for: {name}")
        para.requires_grad = False

post_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

logger.info(f"Total Pre freeze Params {(pre_freeze_param_count )}")
logger.info(f"Total Post freeze Params {(post_freeze_param_count )}")

# Move model to device
model.to(device)
scaler = GradScaler()
print(f" Model created successfully with {post_freeze_param_count:,} trainable parameters")

# Initialize optimizers and criteria
criterion = masked_mse_loss
criterion_cls = nn.CrossEntropyLoss()
criterion_dab = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=lr, eps=1e-4 if config["amp"] else 1e-8 #original lr= 1e-4
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, schedule_interval, gamma=config["schedule_ratio"]
)
scaler = torch.cuda.amp.GradScaler(enabled=config["amp"])

print(" Optimizers and criteria initialized")

print(f"schedule_interval: {schedule_interval}")
print(f"Optimizer initial LR: {optimizer.param_groups[0]['lr']}")

# ===========================
# TEST FUNCTION
# ===========================

def test(model: nn.Module) -> tuple:
    
    print(f"Testing on test set...")
    print(f"Test set shape: {test_data.shape}")
    print(f"Test labels: {len(test_celltype_labels)}")
    
    # Verify no data leakage
    assert test_data.shape[1] == train_data.shape[1], f"Feature dimension mismatch!"
    print(f" Feature dimensions match: {test_data.shape[1]}")
    
    # Use the ALREADY PREPROCESSED test data
    # The test_data was already:
    # 1. Filtered to same genes as training
    # 2. Normalized using same parameters 
    # 3. Binned using same parameters
    # 4. NO re-computation of any statistics
    
    # Tokenize test data using SAME gene_ids and vocab as training
    input_values_test = random_mask_value(
        tokenized_test["values"], 
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )

    test_data_pt = {
        "gene_ids": tokenized_test["genes"],
        "values": input_values_test,
        "target_values": tokenized_test["values"],
        "batch_labels": torch.from_numpy(test_batch_labels).long(),
        "celltype_labels": torch.from_numpy(test_celltype_labels).long(),
    }

    test_loader = DataLoader(
        dataset=SeqDataset(test_data_pt),
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=min(len(os.sched_getaffinity(0)), eval_batch_size // 2),
        pin_memory=True,
    )

    # Evaluate model
    model.eval()
    predictions = evaluate(
        model,
        loader=test_loader,
        return_raw=True,
    )

    # Compute metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(test_celltype_labels, predictions)
    precision = precision_score(test_celltype_labels, predictions, average="macro", zero_division=0)
    recall = recall_score(test_celltype_labels, predictions, average="macro", zero_division=0)
    macro_f1 = f1_score(test_celltype_labels, predictions, average="macro", zero_division=0)

    logger.info(
        f"TEST RESULTS - "
        f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, "
        f"Recall: {recall:.3f}, Macro F1: {macro_f1:.3f}"
    )

    results = {
        "test/accuracy": accuracy,
        "test/precision": precision,
        "test/recall": recall,
        "test/macro_f1": macro_f1,
    }

    return predictions, test_celltype_labels, results


def extract_test_embeddings(model: nn.Module):
    """Extract embeddings from test set for visualization"""
    
    model.eval()
    cell_embeddings = []

    # Get embeddings in batches
    batch_size_embed = 32
    with torch.no_grad():
        with autocast():
            for i in range(0, len(tokenized_test["genes"]), batch_size_embed):
                end_idx = min(i + batch_size_embed, len(tokenized_test["genes"]))
                
                batch_gene_ids = torch.tensor(tokenized_test["genes"][i:end_idx]).to(device)
                batch_values = torch.tensor(tokenized_test["values"][i:end_idx]).to(device)
                src_key_padding_mask = batch_gene_ids.eq(vocab[pad_token])
                
                output_dict = model(
                    batch_gene_ids,
                    batch_values,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=True,
                )
                
                # Extract cell embeddings (keep as tensor)
                if "cell_emb" in output_dict:
                    embeddings = output_dict["cell_emb"].cpu()  
                else:
                    embeddings = output_dict["cls_output"].cpu()  
                
                cell_embeddings.append(embeddings)

    # Concatenate torch tensors first, then convert to numpy
    cell_embeddings = torch.cat(cell_embeddings, dim=0).numpy()  # ← Changed
    print(f"Extracted test cell embeddings shape: {cell_embeddings.shape}")
    
    return cell_embeddings


print("Train distribution:")
print(adata_train.obs['celltype'].value_counts())
print("\nVal distribution:")
print(adata_val.obs['celltype'].value_counts())

# ========================================
# TRAINING PREPARATION
# ========================================

print("\n" + "="*60)
print("TRAINING PREPARATION")
print("="*60)

# Initialize training history tracking
training_history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'learning_rates': [],
    'epoch_times': [],
}
print(" Training history tracking initialized")

# ========================================
# TRAINING LOOP
# ========================================

training_history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'learning_rates': [],
    'epoch_times': [],
    'hierarchical_acc': [],
}

print(" Training history RESET")
print(f"Starting with {len(training_history['val_loss'])} entries")  # Should be 0!

best_val_loss = float("inf")
best_avg_bio = 0.0
best_model = None
best_model_epoch = 0

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    
    # ============================================================
    # PREPARE DATA WITH NEW RANDOM MASKING FOR THIS EPOCH
    # ============================================================
    print(f"\nEpoch {epoch}: Preparing data with random masking...")
    train_data_pt, valid_data_pt = prepare_data(sort_seq_batch=per_seq_batch_sample)
    
    # ============================================================
    # CREATE DATALOADERS FOR THIS EPOCH
    # ============================================================
    
    # Training loader (with balanced sampling)
    train_loader = prepare_dataloader(
        train_data_pt,
        batch_size=batch_size,
        shuffle=False,
        sampler=train_sampler,
        drop_last=True
    )
    
    # Validation loader (natural distribution)
    valid_loader = prepare_dataloader(
        valid_data_pt,
        batch_size=eval_batch_size,
        shuffle=False,
        intra_domain_shuffle=False,
        drop_last=False,
    )
    
    print(f"  ✓ Train loader: {len(train_loader)} batches")
    print(f"  ✓ Valid loader: {len(valid_loader)} batches")
    
    # ============================================================
    # VERIFY BATCH COMPOSITION (EPOCH 1 ONLY)
    # ============================================================
    
    if epoch == 1:
        print("\n" + "="*60)
        print(" VERIFYING BALANCED BATCH COMPOSITION")
        print("="*60)
        
        batch_compositions = []
        temp_iterator = iter(train_loader)
        
        for batch_idx in range(min(5, len(train_loader))):
            try:
                batch_data = next(temp_iterator)
                celltype_labels = batch_data["celltype_labels"]
                
                unique_labels, counts = torch.unique(celltype_labels, return_counts=True)
                batch_compositions.append((unique_labels, counts))
                
                print(f"\n Batch {batch_idx + 1}:")
                for label_idx, count in zip(unique_labels.cpu().numpy(), counts.cpu().numpy()):
                    print(f"  {celltypes[label_idx]:<20}: {count:2d} cells")
                    
            except StopIteration:
                break
        
        # Summary statistics
        print("\n" + "-"*60)
        print(" BATCH BALANCE SUMMARY:")
        
        all_class_appearances = {i: 0 for i in range(num_types)}
        for unique_labels, counts in batch_compositions:
            for label in unique_labels.cpu().numpy():
                all_class_appearances[label] += 1
        
        print(f"\nClass appearance frequency (out of {len(batch_compositions)} batches):")
        for i, cell_type in enumerate(celltypes):
            appearances = all_class_appearances[i]
            print(f"  {cell_type:<20}: {appearances}/{len(batch_compositions)} batches "
                  f"({appearances/len(batch_compositions)*100:.0f}%)")
        
        print("="*60 + "\n")
    
    # ============================================================
    # TRAINING PHASE
    # ============================================================
    
    if config["do_train"]:
        train_metrics = train(model, loader=train_loader)
    
    # ============================================================
    # VALIDATION PHASE
    # ============================================================
    
    val_loss, val_err, hierarchical_results, predictions = integrate_hierarchical_with_evaluation(
        model, valid_loader, device
    )
    
    # Calculate validation accuracy from error rate
    val_acc = 1.0 - val_err

    # EXTRACT HIERARCHICAL ACCURACY FROM RESULTS
    hierarchical_acc = hierarchical_results['overall_hierarchical_accuracy']

    # Get current learning rate
    current_lr = scheduler.get_last_lr()[0]
    
    # Calculate elapsed time
    elapsed = time.time() - epoch_start_time

    # After validation in training loop
    print(f"Epoch {epoch} - Validation")
    print(f"  Standard Accuracy: {val_acc:.3f}")
    print(f"  Hierarchical Accuracy: {hierarchical_acc:.3f}")
    
    # ============================================================
    # TRACK METRICS
    # ============================================================
    
    training_history['train_loss'].append(train_metrics['train_loss']) 
    training_history['train_acc'].append(train_metrics['train_acc'])
    training_history['val_loss'].append(val_loss)
    training_history['val_acc'].append(val_acc)
    training_history['hierarchical_acc'].append(hierarchical_acc)
    training_history['learning_rates'].append(current_lr)
    training_history['epoch_times'].append(elapsed)
        
    # ============================================================
    # VERIFY FULL EPOCH BALANCE (EPOCH 1 ONLY)
    # ============================================================
    
    if epoch == 1:
        print("\n" + "="*60)
        print(" FULL EPOCH CLASS EXPOSURE VERIFICATION")
        print("="*60)
            
        total_seen = {i: 0 for i in range(num_types)}
            
        # Recreate train_loader to iterate through it again
        train_loader_check = prepare_dataloader(
            train_data_pt,
            batch_size=batch_size,
            shuffle=False,
            sampler=train_sampler,
            drop_last=True
        )
        
        for batch_data in train_loader_check:
            labels = batch_data["celltype_labels"]
            for label in labels:
                total_seen[label.item()] += 1
        
        del train_loader_check
            
        print(f"\nTotal samples processed: {sum(total_seen.values()):,}")
        print(f"\nPer-class exposure:")
        print(f"  {'Cell Type':<20} {'Count':>8} {'Target':>8} {'Ratio':>8}")
        print("  " + "-"*46)
            
        target_per_class = sum(total_seen.values()) / num_types
        for i, cell_type in enumerate(celltypes):
            count = total_seen[i]
            ratio = count / target_per_class
            print(f"  {cell_type:<20} {count:>8,} {int(target_per_class):>8,} {ratio:>7.2f}×")
            
        print("\n If ratios are close to 1.0, sampling is balanced over full epoch")
        print("="*60 + "\n")
    
    # ============================================================
    # LOGGING & CHECKPOINTING
    # ============================================================
    
    # Print epoch results
    logger.info("-" * 89)
    logger.info(
        f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
        f"valid loss/mse {val_loss:5.4f} | err {val_err:5.4f} | acc {val_acc:5.4f}"
    )
    logger.info("-" * 89)
    
    # WandB logging
    if 'wandb' in globals() and wandb.run is not None:
        wandb.log({
            "epoch": epoch,
            "train/time_per_epoch": elapsed,
            "val/loss": val_loss,
            "val/error_rate": val_err,
            "val/accuracy": val_acc,
            "train/learning_rate": current_lr,
            "train/is_best_epoch": val_loss < best_val_loss,
        })
        
        if hierarchical_results:
            for level, metrics in hierarchical_results.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        wandb.log({f"hierarchical/{level}_{metric_name}": value})
    
    # Print detailed results
    if epoch % 1 == 0:
        print(f"\nEpoch {epoch} Detailed Results:")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Acc:  {val_acc:.3f}")
        print(f"  LR: {current_lr:.2e}")
        print(f"  Time: {elapsed:.1f}s")
        
        if 'specific' in hierarchical_results:
            print(f"\n  Hierarchical Results:")
            for level, metrics in hierarchical_results.items():
                if isinstance(metrics, dict) and 'accuracy' in metrics:
                    print(f"    {level.capitalize():<12}: {metrics['accuracy']:.3f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        best_model_epoch = epoch
        best_val_acc = val_acc
        torch.save(best_model.state_dict(), save_dir / "best_model.pt")
        logger.info(f"✓ Best model saved with loss {best_val_loss:5.4f} and acc {best_val_acc:5.4f}")
    
    scheduler.step()
    
    # Save checkpoint
    if epoch % save_eval_interval == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_val_loss': best_val_loss,
            'training_history': training_history,
        }, save_dir / f"checkpoint_epoch_{epoch}.pt")
    
    # Clean up to free memory
    del train_data_pt, valid_data_pt, train_loader, valid_loader
    gc.collect()

# ========================================
# TRAINING COMPLETED
# ========================================

print("\n" + "="*60)
print("TRAINING COMPLETED")
print("="*60)
print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_model_epoch}")
print(f"Best validation acc:  {best_val_acc:.3f} at epoch {best_model_epoch}")
print("="*60)

# VISUALIZE TRAINING HISTORY
# ================================================================================

print("\n" + "="*80)
print("TRAINING VISUALIZATIONS")
print("="*80)

def plot_training_history_level2(history, save_dir):
    """
    Plot comprehensive training history for Level 2.
    Similar to Level 1 but adapted for Level 2's multi-class setting.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History - Level 2 (Cell Type Classification)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # ============================================
    # 1. LOSS CURVES
    # ============================================
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', 
             linewidth=2, markersize=6, alpha=0.8)
    ax1.plot(epochs, history['val_loss'], color='#FF8C00', marker='s', label='Val Loss', 
             linewidth=2, markersize=6, alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Mark best epoch
    best_epoch_idx = np.argmin(history['val_loss'])
    ax1.axvline(x=best_epoch_idx+1, color='green', linestyle='--', 
                alpha=0.5, linewidth=2, label='Best Epoch')
    ax1.legend(fontsize=11, loc='best')
    
    # ============================================
    # 2. ACCURACY CURVES
    # ============================================
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Acc', 
             linewidth=2, markersize=6, alpha=0.8)
    ax2.plot(epochs, history['val_acc'], color='#FF8C00', marker='s', label='Val Acc', 
             linewidth=2, markersize=6, alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim([0, 1.05])
    
    # Mark best validation accuracy
    ax2.axvline(x=best_epoch_idx+1, color='green', linestyle='--', 
                alpha=0.5, linewidth=2)
    best_val_acc = max(history['val_acc'])
    ax2.axhline(y=best_val_acc, color='green', linestyle=':', alpha=0.3, linewidth=2)
    ax2.text(0.5, best_val_acc + 0.01, f'Best: {best_val_acc:.3f}', 
             fontsize=10, color='green', fontweight='bold', va='bottom')
    
    # ============================================
    # 3. LEARNING RATE SCHEDULE
    # ============================================
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['learning_rates'], 'g-o', linewidth=2, 
             markersize=6, alpha=0.8)
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Add initial and final LR annotations
    initial_lr = history['learning_rates'][0]
    final_lr = history['learning_rates'][-1]
    ax3.text(1, initial_lr, f'{initial_lr:.2e}', fontsize=9, 
             color='green', fontweight='bold', va='bottom')
    ax3.text(len(epochs), final_lr, f'{final_lr:.2e}', fontsize=9, 
             color='red', fontweight='bold', va='top')
    
    # ============================================
    # 4. OVERFITTING ANALYSIS
    # ============================================
    ax4 = axes[1, 1]
    loss_gap = np.array(history['val_loss']) - np.array(history['train_loss'])
    acc_gap = np.array(history['train_acc']) - np.array(history['val_acc'])
    
    ax4_twin = ax4.twinx()
    line1 = ax4.plot(epochs, loss_gap, 'r-o', label='Loss Gap (Val-Train)', 
                     linewidth=2, markersize=6, alpha=0.8)
    line2 = ax4_twin.plot(epochs, acc_gap, 'b-s', label='Acc Gap (Train-Val)', 
                          linewidth=2, markersize=6, alpha=0.8)
    
    ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Loss Gap', fontsize=12, color='r', fontweight='bold')
    ax4_twin.set_ylabel('Accuracy Gap', fontsize=12, color='b', fontweight='bold')
    ax4.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
    ax4.tick_params(axis='y', labelcolor='r')
    ax4_twin.tick_params(axis='y', labelcolor='b')
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    # Reference lines at 0
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax4_twin.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left', fontsize=11)
    
    plt.tight_layout()
    
    # Save figure
    save_path = save_dir / "training_curves.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Training curves saved to {save_path}")
    
    plt.show()
    
    # ============================================
    # TRAINING SUMMARY STATISTICS
    # ============================================
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total epochs: {len(epochs)}")
    print(f"Total training time: {sum(history['epoch_times']):.1f}s ({sum(history['epoch_times'])/60:.1f} min)")
    print(f"Average time per epoch: {np.mean(history['epoch_times']):.1f}s")
    print(f"\nBest validation loss: {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss'])+1})")
    print(f"Best validation accuracy: {max(history['val_acc']):.4f} (Epoch {np.argmax(history['val_acc'])+1})")
    print(f"\nFinal metrics:")
    print(f"  Train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Val loss:   {history['val_loss'][-1]:.4f}")
    print(f"  Train acc:  {history['train_acc'][-1]:.3f}")
    print(f"  Val acc:    {history['val_acc'][-1]:.3f}")
    print(f"  Train-val loss gap: {history['val_loss'][-1] - history['train_loss'][-1]:.4f}")
    print(f"  Train-val acc gap:  {history['train_acc'][-1] - history['val_acc'][-1]:.3f}")
    print("="*60)
    
    return fig

# Generate the plots
fig = plot_training_history_level2(training_history, save_dir)

# Save training history
import pickle
with open(save_dir / "training_history.pkl", 'wb') as f:
    pickle.dump(training_history, f)
print(f"\n✓ Training history saved to {save_dir / 'training_history.pkl'}")

# CONFUSION MATRIX FUNCTIONS

def create_auto_abbreviations(cell_types, max_length=8):
    """Automatically create abbreviations for any cell type names."""
    abbreviations = {}
    
    for cell_type in cell_types:
        if '.' in cell_type:
            parts = cell_type.split('.')
            if len(parts) == 2:
                abbrev = f"{parts[0]}-{parts[1][:3]}"
            else:
                abbrev = cell_type[:max_length]
        elif len(cell_type) <= max_length:
            abbrev = cell_type
        else:
            if cell_type.count(' ') > 0:
                words = cell_type.split()
                abbrev = ''.join(word[0].upper() for word in words)
                if len(abbrev) < 3:
                    abbrev = words[0][:4] + words[-1][:2] if len(words) > 1 else cell_type[:max_length]
            else:
                abbrev = cell_type[:max_length]
        
        abbreviations[cell_type] = abbrev[:max_length]
    
    # Check for duplicates and fix them
    seen_abbrevs = {}
    for cell_type, abbrev in abbreviations.items():
        if abbrev in seen_abbrevs:
            counter = 2
            new_abbrev = f"{abbrev[:-1]}{counter}"
            while new_abbrev in seen_abbrevs.values():
                counter += 1
                new_abbrev = f"{abbrev[:-1]}{counter}"
            abbreviations[cell_type] = new_abbrev
        seen_abbrevs[cell_type] = abbreviations[cell_type]
    
    return abbreviations

def create_clean_confusion_matrix(cm_df, save_dir, use_abbreviations=True):
    """Create a clean, readable confusion matrix for ANY dataset."""
    
    cm_display = cm_df.copy()
    
    if use_abbreviations and len(cm_df.index) > 10:
        cell_types = list(cm_df.index)
        abbreviations = create_auto_abbreviations(cell_types)
        
        new_index = [abbreviations[name] for name in cm_df.index]
        new_columns = [abbreviations[name] for name in cm_df.columns]
        
        cm_display.index = new_index
        cm_display.columns = new_columns
        
        print("\nCell Type Abbreviations Used:")
        print("-" * 40)
        for orig, abbrev in abbreviations.items():
            print(f"{abbrev:8s} = {orig}")
        print()
    
    n_classes = len(cm_df)
    fig_size = max(10, min(20, n_classes * 0.7))
    
    plt.figure(figsize=(fig_size, fig_size-1))
    
    annot_size = max(6, min(10, 120 // n_classes))
    
    ax = sns.heatmap(
        cm_display, 
        annot=True, 
        fmt=".2f", 
        cmap="Blues",
        cbar_kws={'label': 'Normalized Frequency', 'shrink': 0.8},
        annot_kws={'size': annot_size},
        square=True,
        linewidths=0.5,
        linecolor='white'
    )
    
    plt.title("Confusion Matrix - Model Performance", fontsize=16, pad=25, fontweight='bold')
    plt.ylabel("True Cell Type", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Cell Type", fontsize=14, fontweight='bold')
    
    avg_label_length = np.mean([len(str(label)) for label in cm_display.columns])
    rotation_angle = 90 if avg_label_length > 6 else 45
    
    plt.xticks(rotation=rotation_angle, ha='right' if rotation_angle==45 else 'center', fontsize=max(8, 12-n_classes//10))
    plt.yticks(rotation=0, fontsize=max(8, 12-n_classes//10))
    
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix_clean.png", dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    return cm_display

def create_focused_confusion_matrix(cm_df, save_dir, focus_threshold=0.05):
    """Create a focused view showing only significant confusions."""
    
    n_classes = len(cm_df)
    auto_threshold = max(0.02, min(0.10, 1.0 / n_classes))
    threshold = min(focus_threshold, auto_threshold)
    
    mask = np.zeros_like(cm_df, dtype=bool)
    
    for i in range(len(cm_df)):
        for j in range(len(cm_df.columns)):
            if i != j and cm_df.iloc[i, j] < threshold:
                mask[i, j] = True
    
    fig_size = max(10, min(20, n_classes * 0.7))
    plt.figure(figsize=(fig_size, fig_size-1))
    
    ax = sns.heatmap(
        cm_df, 
        mask=mask,
        annot=True, 
        fmt=".3f", 
        cmap="Blues",
        cbar_kws={'label': 'Normalized Frequency', 'shrink': 0.8},
        annot_kws={'size': max(6, min(10, 120 // n_classes))},
        square=True,
        linewidths=0.5,
        linecolor='white',
        vmin=0, vmax=1
    )
    
    plt.title(f"Confusion Matrix - Major Confusions Only (>{threshold:.1%})", 
              fontsize=16, pad=25, fontweight='bold')
    plt.ylabel("True Cell Type", fontsize=14, fontweight='bold') 
    plt.xlabel("Predicted Cell Type", fontsize=14, fontweight='bold')
    
    avg_label_length = np.mean([len(str(label)) for label in cm_df.columns])
    rotation_angle = 90 if avg_label_length > 6 else 45
    
    plt.xticks(rotation=rotation_angle, ha='right' if rotation_angle==45 else 'center', 
               fontsize=max(8, 12-n_classes//10))
    plt.yticks(rotation=0, fontsize=max(8, 12-n_classes//10))
    
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix_focused.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')

def analyze_major_confusions(cm_df, auto_threshold=True):
    """Identify and report major confusions - adapts to any dataset."""
    n_classes = len(cm_df)
    
    if auto_threshold:
        threshold = max(0.05, 2.0 / n_classes)  
    else:
        threshold = 0.1
    
    print(f"\n{'='*60}")
    print(f"MAJOR CLASSIFICATION CONFUSIONS (>{threshold:.1%})")
    print(f"{'='*60}")
    
    confusions = []
    
    for i, true_type in enumerate(cm_df.index):
        for j, pred_type in enumerate(cm_df.columns):
            if i != j and cm_df.iloc[i, j] > threshold:
                severity = 'HIGH' if cm_df.iloc[i, j] > (4.0/n_classes) else 'MODERATE'
                confusions.append({
                    'true': true_type,
                    'predicted': pred_type,
                    'rate': cm_df.iloc[i, j],
                    'severity': severity
                })
    
    if confusions:
        confusions.sort(key=lambda x: x['rate'], reverse=True)
        
        print(f"Found {len(confusions)} major confusions:")
        print("-" * 60)
        
        for conf in confusions[:10]:
            true_name = str(conf['true'])[:20]
            pred_name = str(conf['predicted'])[:20]
            print(f"{conf['severity']:8s} | {true_name:20s} → {pred_name:20s} ({conf['rate']:.1%})")
        
        if len(confusions) > 10:
            print(f"... and {len(confusions)-10} more")
    else:
        print(f" No major confusions found!")

def create_all_confusion_visualizations(cm_df, save_dir):
    """Create all confusion matrix visualizations - works with ANY dataset."""
    print(f"Creating confusion matrix visualizations for {len(cm_df)} classes...")
    
    # 1. Clean version
    cm_display = create_clean_confusion_matrix(cm_df, save_dir)
    print(" Created clean confusion matrix")
    
    # 2. Focused version
    create_focused_confusion_matrix(cm_df, save_dir)
    print(" Created focused confusion matrix")
    
    # 3. Analysis
    analyze_major_confusions(cm_df)
    
    # 4. Per-class accuracy ranking
    diagonal_accuracy = {cm_df.index[i]: cm_df.iloc[i, i] for i in range(len(cm_df))}
    sorted_accuracy = sorted(diagonal_accuracy.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'='*60}")
    print("PER-CLASS ACCURACY RANKING")
    print(f"{'='*60}")
    
    n_show = min(5, len(sorted_accuracy))
    
    print("Top Performers:")
    for i, (cell_type, acc) in enumerate(sorted_accuracy[:n_show]):
        type_name = str(cell_type)[:25]
        print(f"{i+1:2d}. {type_name:25s}: {acc:.1%}")
    
    print("\nNeed Improvement:")
    for i, (cell_type, acc) in enumerate(sorted_accuracy[-n_show:]):
        rank = len(sorted_accuracy) - n_show + 1 + i
        type_name = str(cell_type)[:25]
        print(f"{rank:2d}. {type_name:25s}: {acc:.1%}")
    
    plt.show()
    return cm_display

# ===Other viz===
def create_performance_analysis_level2(test_labels, test_predictions, celltypes, save_dir):
    """
    Create comprehensive performance analysis visualizations.
    """
    from sklearn.metrics import classification_report, precision_recall_fscore_support
    
    # Get per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels, test_predictions, average=None, zero_division=0
    )
    
    # Calculate macro and micro averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        test_labels, test_predictions, average='macro', zero_division=0
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        test_labels, test_predictions, average='micro', zero_division=0
    )
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # ============================================
    # 1. PER-CLASS PERFORMANCE BAR CHART
    # ============================================
    ax1 = axes[0]
    
    # Sort by F1-score
    sorted_indices = np.argsort(f1)
    sorted_celltypes = [celltypes[i] for i in sorted_indices]
    sorted_precision = precision[sorted_indices]
    sorted_recall = recall[sorted_indices]
    sorted_f1 = f1[sorted_indices]
    
    x = np.arange(len(celltypes))
    width = 0.25
    
    ax1.barh(x - width, sorted_precision, width, label='Precision', color='#1f77b4', alpha=0.8)
    ax1.barh(x, sorted_recall, width, label='Recall', color='#ff7f0e', alpha=0.8)
    ax1.barh(x + width, sorted_f1, width, label='F1-Score', color='#2ca02c', alpha=0.8)
    
    ax1.set_yticks(x)
    ax1.set_yticklabels(sorted_celltypes, fontsize=9)
    ax1.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Per-Class Performance', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.axvline(x=np.mean(f1), color='red', linestyle='--', alpha=0.5, label=f'Mean F1: {np.mean(f1):.3f}')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_xlim([0, 1.05])
    
    # ============================================
    # 2. MACRO VS MICRO COMPARISON
    # ============================================
    ax2 = axes[1]
    
    metrics = ['Precision', 'Recall', 'F1-Score']
    macro_scores = [macro_precision, macro_recall, macro_f1]
    micro_scores = [micro_precision, micro_recall, micro_f1]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, macro_scores, width, label='Macro Average', 
                     color='#1f77b4', alpha=0.8)
    bars2 = ax2.bar(x + width/2, micro_scores, width, label='Micro Average', 
                     color='#ff7f0e', alpha=0.8)
    
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('Macro vs Micro Averages', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1.05])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # ============================================
    # 3. PERFORMANCE VS FREQUENCY SCATTER WITH F1 COLORBAR
    # ============================================
    ax3 = axes[2]

    # Create scatter plot with continuous colormap
    scatter = ax3.scatter(
        support, 
        f1, 
        c=f1,  # Color by F1 score
        cmap='RdYlGn',  # Red-Yellow-Green gradient
        s=100, 
        alpha=0.7, 
        edgecolors='black',
        vmin=0,
        vmax=1
    )

    #  F1-Score colorbar
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('F1-Score', fontsize=11, fontweight='bold')

    #  Annotate ALL classes with their names
    for i, (class_name, supp, f1_val) in enumerate(zip(celltypes, support, f1)):
        ax3.annotate(
            class_name,
            (supp, f1_val),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.8,
            fontweight='bold' if f1_val < 0.6 else 'normal'  # Bold for worst performers
        )

    ax3.set_xlabel('Class Support (samples)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax3.set_title('Performance vs Frequency', fontsize=14, fontweight='bold')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(save_dir / "performance_analysis.png", dpi=300, bbox_inches='tight')
    print(f"✓ Performance analysis saved to {save_dir / 'performance_analysis.png'}")
    plt.show()
    
    # ============================================
    # PRINT WORST PERFORMERS
    # ============================================
    print("\n" + "="*60)
    print("WORST PERFORMING CLASSES (Bottom 3 by F1-Score)")
    print("="*60)
    worst_indices = np.argsort(f1)[:3]
    for idx in worst_indices:
        print(f"{celltypes[idx]:20s}: P={precision[idx]:.3f}, R={recall[idx]:.3f}, "
              f"F1={f1[idx]:.3f}, Support={support[idx]}")
    print("="*60)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'cell_type': celltypes,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': support
    }).sort_values('f1_score')
    
    metrics_df.to_csv(save_dir / 'per_class_metrics.csv', index=False)
    print(f"✓ Per-class metrics saved to {save_dir / 'per_class_metrics.csv'}")


# ============================================================
# TEST EVALUATION
# ============================================================

print("="*60)
print("EVALUATING ON TEST SET WITH BEST MODEL")
print("="*60)

# Load best model
best_model_path = save_dir / "best_model.pt"
model.load_state_dict(torch.load(best_model_path, map_location=device))
print("✓ Best model loaded")

# ============================================================
# PREPARE TEST DATA & LOADER
# ============================================================

input_values_test = random_mask_value(
    tokenized_test["values"], 
    mask_ratio=mask_ratio,
    mask_value=mask_value,
    pad_value=pad_value,
)

test_data_pt = {
    "gene_ids": tokenized_test["genes"],
    "values": input_values_test,
    "target_values": tokenized_test["values"],
    "batch_labels": torch.from_numpy(test_batch_labels).long(),
    "celltype_labels": torch.from_numpy(test_celltype_labels).long(),
}

test_loader = DataLoader(
    dataset=SeqDataset(test_data_pt),
    batch_size=eval_batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=min(len(os.sched_getaffinity(0)), eval_batch_size // 2),
    pin_memory=True,
)
print(f"✓ Test loader created: {len(test_loader)} batches")

# ============================================================
# BASIC EVALUATION
# ============================================================

model.eval()
test_predictions_basic = evaluate(model, loader=test_loader, return_raw=True)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(test_celltype_labels, test_predictions_basic)
precision = precision_score(test_celltype_labels, test_predictions_basic, average="macro", zero_division=0)
recall = recall_score(test_celltype_labels, test_predictions_basic, average="macro", zero_division=0)
macro_f1 = f1_score(test_celltype_labels, test_predictions_basic, average="macro", zero_division=0)

print(f"\nBasic Test Performance:")
print(f"  Accuracy: {accuracy:.3f}")
print(f"  Precision: {precision:.3f}")
print(f"  Recall: {recall:.3f}")
print(f"  Macro F1: {macro_f1:.3f}")

# ============================================================
# HIERARCHICAL EVALUATION
# ============================================================

test_loss, test_err, test_hierarchical_results, test_predictions_dict = integrate_hierarchical_with_evaluation(
    model, test_loader, device
)

test_acc = 1.0 - test_err
test_hierarchical_acc = test_hierarchical_results['overall_hierarchical_accuracy']

print(f"  Test Accuracy: {test_acc:.3f}")
print(f"  Test Hierarchical Accuracy: {test_hierarchical_acc:.3f}")

print("\n" + "="*60)
print("HIERARCHICAL CLASSIFICATION RESULTS")
print("="*60)
print(test_hierarchical_results)

# ============================================================================
# COMPREHENSIVE PERFORMANCE SUMMARY - LEVEL 2
# ============================================================================

print("\n" + "="*80)
print("PERFORMANCE ANALYSIS SUMMARY - LEVEL 2")
print("="*80)

# Use the variables from your code
test_labels = test_celltype_labels 
test_preds = test_predictions_basic  

# Generate full classification report
from sklearn.metrics import classification_report

report = classification_report(
    test_labels, 
    test_preds, 
    target_names=celltypes,  # Your 8 class names
    output_dict=True,
    zero_division=0
)

# Extract macro averages
macro_precision = report['macro avg']['precision']
macro_recall = report['macro avg']['recall']
macro_f1 = report['macro avg']['f1-score']

# Extract weighted averages
weighted_precision = report['weighted avg']['precision']
weighted_recall = report['weighted avg']['recall']
weighted_f1 = report['weighted avg']['f1-score']

# Calculate overall accuracy (already have it, but recalculate for consistency)
overall_accuracy = (test_labels == test_preds).mean()

# Print summary
print(f"Overall Test Accuracy: {overall_accuracy:.1%}")

print(f"\nMacro Averages (treats all 8 classes equally):")
print(f"  Precision: {macro_precision:.3f}")
print(f"  Recall:    {macro_recall:.3f}")
print(f"  F1-Score:  {macro_f1:.3f}")

print(f"\nWeighted Averages (weighted by class support):")
print(f"  Precision: {weighted_precision:.3f}")
print(f"  Recall:    {weighted_recall:.3f}")
print(f"  F1-Score:  {weighted_f1:.3f}")

# Analyze class imbalance impact
print(f"\nClass Imbalance Impact:")
gap = weighted_f1 - macro_f1
if gap > 0.05:
    print(f"   Weighted-Macro gap: {gap:.3f} (model biased toward frequent classes)")
elif gap < -0.05:
    print(f"   Weighted-Macro gap: {gap:.3f} (model underperforms on frequent classes)")
else:
    print(f"  Weighted-Macro gap: {gap:.3f} (balanced performance)")

# Hierarchical accuracy comparison
print(f"\nHierarchical Classification:")
print(f"  Standard Accuracy:     {test_acc:.1%}")
print(f"  Hierarchical Accuracy: {test_hierarchical_acc:.1%}")
boost = test_hierarchical_acc - test_acc
print(f"  Hierarchical Boost:    {boost:+.1%}")

print("="*80)

print("\n✓ Comprehensive performance analysis complete!")

create_performance_analysis_level2(
    test_celltype_labels, 
    test_predictions_basic, 
    celltypes,
    save_dir
)

# ============================================================
# VISUALIZATIONS (use basic predictions)
# ============================================================

print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

# Extract embeddings
print("Extracting embeddings for UMAP...")
test_embeddings = extract_test_embeddings(model)

# Create test AnnData for visualization
adata_test_vis = adata_test.copy()
adata_test_vis.obs["predictions"] = [id2type[p] for p in test_predictions_basic]
adata_test_vis.obsm["X_scgpt"] = test_embeddings

# Compute UMAP
print("Computing UMAP...")
sc.pp.neighbors(adata_test_vis, use_rep="X_scgpt", n_neighbors=15)
sc.tl.umap(adata_test_vis)

# Create color palette
palette_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
while len(palette_colors) < len(celltypes):
    palette_colors.extend(plt.rcParams["axes.prop_cycle"].by_key()["color"])
palette_ = {c: palette_colors[i] for i, c in enumerate(celltypes)}

# Plot UMAP
print("Creating UMAP plots...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 16), dpi=150)

sc.pl.umap(adata_test_vis, color="celltype", palette=palette_, ax=ax1, show=False, title="True Cell Types")
sc.pl.umap(adata_test_vis, color="predictions", palette=palette_, ax=ax2, show=False, title="Predicted Cell Types")

plt.suptitle("Test Results", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(save_dir / "test_results.png", dpi=300, bbox_inches='tight')
plt.show()


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("Creating confusion matrix...")
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_celltype_labels, test_predictions_basic)
cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

cm_df = pd.DataFrame(
    cm_normalized,
    index=[id2type[i] for i in sorted(set(test_celltype_labels))],
    columns=[id2type[i] for i in sorted(set(test_predictions_basic))]
)

cm_display = create_all_confusion_visualizations(cm_df, save_dir)

# Create comprehensive performance analysis
print("\n" + "="*60)
print("COMPREHENSIVE PERFORMANCE ANALYSIS")
print("="*60)

create_performance_analysis_level2(
    test_celltype_labels, 
    test_predictions_basic,
    celltypes,
    save_dir
)

"""if needed, you can add more visualizations here, such as:""
print("\n" + "="*80)
print("COMPREHENSIVE PERFORMANCE ANALYSIS - LEVEL 2")
print("="*80)

print("\n1. Calculating per-class metrics...")
# [Copy from L3 - per_class_performance.png]

print("\n2. Computing macro vs micro averages...")
# [Copy from L3 - macro_vs_micro.png]

print("\n3. Analyzing performance vs class frequency...")
# [Copy from L3 - performance_vs_frequency.png with names]

print("\n4. Identifying worst performers...")
# Bottom 3 (not 5)

print("\n5. F1-Score Heatmap...")
# New heatmap

print("\n✓ Comprehensive performance analysis complete!")
"""
# ============================================================
# SAVE RESULTS
# ============================================================

print("\nSaving results...")
test_save_dict = {
    "test_predictions": test_predictions_basic,
    "test_labels": test_celltype_labels,
    "test_accuracy": accuracy,
    "test_loss": test_loss,
    "hierarchical_results": test_hierarchical_results,
    "id_maps": id2type,
    "test_embeddings": test_embeddings,
}

with open(save_dir / "test_results.pkl", "wb") as f:
    pickle.dump(test_save_dict, f)

print(f"✓ All results saved to: {save_dir}")
print("="*60)
print(" TEST EVALUATION COMPLETE")
print("="*60)