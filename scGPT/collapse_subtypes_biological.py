# ============================================================================
# COLLAPSE 296 SUBTYPES → 22 BIOLOGICALLY MEANINGFUL GROUPS (WITH RANGES)
# ============================================================================
# Run this BEFORE training to create new h5ad with collapsed labels
# ============================================================================

import scanpy as sc
import pandas as pd
import numpy as np

def collapse_to_biological_groups(adata):
    """
    Collapse 296 Allen Atlas computational clusters into 22 biologically meaningful groups.
    
    Strategy:
    - Group similar numbered clusters (e.g., Oligo_40-43 → Oligo_early_mature (40-43))
    - Preserve biological distinctions (OPC vs mature vs COP)
    - Keep vascular subtypes (already biological)
    """
    
    def map_subtype_to_group(subtype):
        """Map individual subtype to biological group with range labels"""
        subtype_str = str(subtype)
        
        # ====================================================================
        # OLIGODENDROCYTES (5 groups)
        # ====================================================================
        if subtype_str.startswith('Oligo_'):
            # Extract number
            try:
                num = int(subtype_str.split('_')[1])
                if 40 <= num <= 43:
                    return 'Oligo_early_mature (40-43)'
                elif 44 <= num <= 47:
                    return 'Oligo_mid_mature (44-47)'
                elif 48 <= num <= 50:
                    return 'Oligo_late_mature (48-50)'
                else:
                    return 'Oligo_other'
            except:
                return 'Oligo_other'
        
        elif subtype_str.startswith('OPC_'):
            return 'OPC'  # Keep separate (precursor cells)
        
        elif subtype_str.startswith('COP_'):
            return 'COP'  # Keep separate (committed precursors)
        
        # ====================================================================
        # ASTROCYTES (4 groups)
        # ====================================================================
        elif subtype_str.startswith('Astro_'):
            try:
                num = int(subtype_str.split('_')[1])
                if 52 <= num <= 56:
                    return 'Astro_protoplasmic (52-56)'
                elif 57 <= num <= 60:
                    return 'Astro_fibrous (57-60)'
                elif 61 <= num <= 64:
                    return 'Astro_reactive (61-64)'
                else:
                    return 'Astro_other'
            except:
                return 'Astro_other'
        
        elif subtype_str.startswith('Bgl_'):
            return 'Bergmann_glia'  # Keep separate (specialized astrocyte)
        
        # ====================================================================
        # MICROGLIA (2 groups)
        # ====================================================================
        elif subtype_str.startswith('Mgl_'):
            try:
                num = int(subtype_str.split('_')[1])
                if 4 <= num <= 7:
                    return 'Microglia_homeostatic (4-7)'
                elif 8 <= num <= 12:
                    return 'Microglia_activated (8-12)'
                else:
                    return 'Microglia_other'
            except:
                return 'Microglia_other'
        
        # ====================================================================
        # VASCULAR ENDOTHELIAL (6 groups - keep as is, already biological)
        # ====================================================================
        elif subtype_str.startswith('VendC_'):
            return 'Endothelial_capillary'
        
        elif subtype_str.startswith('VendAC_'):
            return 'Endothelial_arterial_capillary'
        
        elif subtype_str.startswith('VendVC_'):
            return 'Endothelial_venous_capillary'
        
        elif subtype_str.startswith('VendA_'):
            return 'Endothelial_arterial'
        
        elif subtype_str.startswith('VendV_'):
            return 'Endothelial_venous'
        
        elif subtype_str.startswith('VendPLVAP_'):
            return 'Endothelial_specialized'
        
        # ====================================================================
        # OTHER VASCULAR (2 groups)
        # ====================================================================
        elif subtype_str.startswith('Per_'):
            return 'Pericytes'  # Keep as single group
        
        elif subtype_str.startswith('Vsmc_'):
            return 'Vascular_SMC'  # Keep as single group
        
        # ====================================================================
        # EPITHELIAL (2 groups)
        # ====================================================================
        elif subtype_str.startswith('Epen_'):
            return 'Ependymal'  # Keep as single group
        
        elif subtype_str.startswith('Chrp_'):
            return 'Choroid_plexus'  # Keep as single group
        
        # ====================================================================
        # STROMAL (1 group)
        # ====================================================================
        elif subtype_str.startswith('Fbl_'):
            return 'Fibroblasts'  # Keep as single group
        
        # ====================================================================
        # UNKNOWN
        # ====================================================================
        else:
            return 'Unknown'
    
    # Apply mapping
    print("Mapping 296 subtypes to biological groups...")
    
    # Don't create subtype_original if it already exists
    if 'subtype_original' not in adata.obs.columns:
        adata.obs['subtype_original'] = adata.obs['subtype'].copy()  # Backup original
    
    adata.obs['subtype'] = adata.obs['subtype'].apply(map_subtype_to_group)
    
    # Convert to categorical
    adata.obs['subtype'] = adata.obs['subtype'].astype('category')
    
    # Print summary
    print("\n" + "="*80)
    print("BIOLOGICAL GROUPING SUMMARY")
    print("="*80)
    print(f"Original subtypes: 296")
    print(f"Biological groups: {adata.obs['subtype'].nunique()}")
    print(f"\nGroup distribution:")
    
    group_counts = adata.obs['subtype'].value_counts().sort_values(ascending=False)
    for group, count in group_counts.items():
        print(f"  {group:35s}: {count:>8,} cells")
    
    print("="*80)
    
    return adata


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    print("Loading original dataset...")
    
    # Use the PROCESSED file (has celltype and subtype columns)
    adata_full = sc.read_h5ad("./data/adata_processed_L2_L3.h5ad")
    
    print(f"Original shape: {adata_full.shape}")
    print(f"Original subtypes: {adata_full.obs['subtype'].nunique()}")
    
    # Collapse subtypes
    adata_collapsed = collapse_to_biological_groups(adata_full)
    
    # Save collapsed version
    output_path = "./data/Nonneuron_collapsed.h5ad"
    print(f"\nSaving collapsed dataset to: {output_path}")
    adata_collapsed.write_h5ad(output_path)
    
    print("\n✓ Done! Use this file for training:")
    print(f"  dataset_path = '{output_path}'")
    
    # Validation: Show some examples of mapping
    print("\n" + "="*80)
    print("EXAMPLE MAPPINGS")
    print("="*80)
    examples = adata_collapsed.obs[['subtype_original', 'subtype']].drop_duplicates().head(30)
    for idx, row in examples.iterrows():
        print(f"  {row['subtype_original']:30s} → {row['subtype']}")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. In your training notebook config, update:")
    print("   dataset_path='./data/Nonneuron_collapsed.h5ad'")
    print("   metadata_path=None")
    print("   gene_list_path=None")
    print("")
    print("2. Delete old processed files:")
    print("   rm ./save/adata_*_level3_processed.h5ad")
    print("")
    print("3. Update _classify_cell_type function in notebook")
    print("")
    print("4. Run training from start!")
    print("="*80)