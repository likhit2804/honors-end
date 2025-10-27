import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import itertools
from collections import defaultdict

# --- CHANGED ---
# Updated signature to accept paths for all 5 data types.
# Removed synthetic data generation.
def load_omics( somatic_path, cnv_path, mirna_path, clinical_path):
    """
    Loads real gene, somatic, cnv, mirna, and clinical data from CSVs.
    """
    # Load real data
  
    somatic_df = pd.read_csv(somatic_path, index_col=0)
    cnv_df = pd.read_csv(cnv_path, index_col=0)
    # --- NEW ---
    mirna_df = pd.read_csv(mirna_path, index_col=0)
    clinical_df = pd.read_csv(clinical_path, index_col=0)



    # Normalize real data
    # Transpose so that rows are patients (samples) and columns are features
    
    
    
    scaler_somatic = StandardScaler()
    somatic = torch.tensor(scaler_somatic.fit_transform(somatic_df.T), dtype=torch.float) 
    
    scaler_cnv = StandardScaler()
    cnv = torch.tensor(scaler_cnv.fit_transform(cnv_df.T), dtype=torch.float)

    # --- NEW: Process mirna and clinical data ---
    scaler_mirna = StandardScaler()
    mirna = torch.tensor(scaler_mirna.fit_transform(mirna_df.T), dtype=torch.float)
    
    scaler_clinical = StandardScaler()
    # Clinical data might be categorical; for this model, we assume it's all numeric.
    # If you have categorical data (e.g., text), it must be one-hot encoded first.
    clinical = torch.tensor(scaler_clinical.fit_transform(clinical_df.T), dtype=torch.float)
    
    print(f"Loaded real data (patients x features):")
   
    print(f"  Somatic: {somatic.shape}")
    print(f"  CNV: {cnv.shape}")
    print(f"  miRNA Expression: {mirna.shape}")
    print(f"  Clinical: {clinical.shape}")

    # Note: If you have labels (e.g., patient subtype), you should load them here
    # from the clinical file (or another file) and return them as well.
    # For now, we only return the features.
    
    return  somatic, cnv, mirna, clinical

def build_pathway_graph(pathway_path):
    df = pd.read_csv(pathway_path)
    hyperedges = defaultdict(list)
    for _, row in df.iterrows():
        hyperedges[row["Pathway_ID"]].append(row["Entity_ID"])

    entities = sorted(set(df["Entity_ID"]))
    entity_to_idx = {e: i for i, e in enumerate(entities)}

    edges = []
    for members in hyperedges.values():
        for a, b in itertools.combinations(members, 2):
            edges.append((entity_to_idx[a], entity_to_idx[b]))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index, entity_to_idx