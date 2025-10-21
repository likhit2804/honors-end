import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

def load_omics(gene_path, mirna_path, clinical_path):
    gene_df = pd.read_csv(gene_path, index_col=0)
    mirna_df = pd.read_csv(mirna_path, index_col=0)
    clinical_df = pd.read_csv(clinical_path, index_col=0)

    # Normalize
    scaler = StandardScaler()
    gene = torch.tensor(scaler.fit_transform(gene_df.T), dtype=torch.float)
    mirna = torch.tensor(scaler.fit_transform(mirna_df.T), dtype=torch.float)
    clinical = torch.tensor(clinical_df.values, dtype=torch.float)

    return gene, mirna, clinical

import itertools

from collections import defaultdict

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
