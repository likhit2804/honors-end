import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import itertools
from collections import defaultdict

def load_omics(gene_path, somatic_path, cnv_path, num_clinical_features=10, num_mirna_features=500):
    """
    Loads real gene, somatic, and cnv data from CSVs.
    Generates synthetic mirna and clinical data.
    """
    # Load real data
    gene_df = pd.read_csv(gene_path, index_col=0)
    somatic_df = pd.read_csv(somatic_path, index_col=0)
    cnv_df = pd.read_csv(cnv_path, index_col=0)

    # Assuming patients are columns, features are rows
    num_patients = gene_df.shape[1] 

    # Normalize real data
    scaler = StandardScaler()
    # Transpose so that rows are patients (samples) and columns are features
    gene = torch.tensor(scaler.fit_transform(gene_df.T), dtype=torch.float)
    
    # Apply standard scaling to somatic and CNV data
    # (Assuming they are also features x patients)
    scaler_somatic = StandardScaler()
    somatic = torch.tensor(scaler_somatic.fit_transform(somatic_df.T), dtype=torch.float) 
    
    scaler_cnv = StandardScaler()
    cnv = torch.tensor(scaler_cnv.fit_transform(cnv_df.T), dtype=torch.float)

    # Generate synthetic data for missing omics
    mirna = torch.randn(num_patients, num_mirna_features, dtype=torch.float)
    clinical = torch.randn(num_patients, num_clinical_features, dtype=torch.float)
    
    print(f"Loaded real data (patients x features):")
    print(f"  Gene Expression: {gene.shape}")
    print(f"  Somatic: {somatic.shape}")
    print(f"  CNV: {cnv.shape}")
    print(f"Generated synthetic data (patients x features):")
    print(f"  miRNA Expression: {mirna.shape}")
    print(f"  Clinical: {clinical.shape}")

    return gene, somatic, cnv, mirna, clinical

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