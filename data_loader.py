# In data_loader.py

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import itertools
from collections import defaultdict

def load_omics(mirna_path, somatic_path, cnv_path, clinical_path):
    """
    Loads real mirna, somatic, cnv, and clinical data from CSVs.
    """
    # Load all data files
    # index_col=0 assumes the first column is the feature/patient ID
    mirna_df = pd.read_csv(mirna_path,index_col=0)
    somatic_df = pd.read_csv(somatic_path)
    cnv_df = pd.read_csv(cnv_path)
    
    # Load clinical data - 'index_col=0' is critical
    clinical_df = pd.read_csv(clinical_path, index_col=0)

    # Normalize real data
    scaler = StandardScaler()
    
    # Transpose omics data (assuming features are rows, patients are columns)
    mirna = torch.tensor(scaler.fit_transform(mirna_df.T), dtype=torch.float)
    
    scaler_somatic = StandardScaler()
    somatic = torch.tensor(scaler_somatic.fit_transform(somatic_df.T), dtype=torch.float) 
    
    scaler_cnv = StandardScaler()
    cnv = torch.tensor(scaler_cnv.fit_transform(cnv_df.T), dtype=torch.float)
    
    # Scale clinical data (assuming patients are rows, features are columns)
    # NO .T on this one - this was the source of your first error
    scaler_clinical = StandardScaler()
    clinical = torch.tensor(scaler_clinical.fit_transform(clinical_df), dtype=torch.float)

    print(f"Loaded real data (patients x features):")
    print(f"  miRNA Expression: {mirna_df.isnull().sum()}")
    print(f"  Somatic: {somatic_df.isnull().sum()}")
    print(f"  CNV: {cnv_df.isnull().sum()}")
    print(f"  Clinical: {clinical_df.isnull().sum()}")

    # Return the four loaded tensors
    return mirna, somatic, cnv, clinical

# --- (build_pathway_graph function stays the same) ---
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