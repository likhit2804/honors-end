import torch
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_patient_graph(features, k=5):
    """(This function is no longer used but kept for reference)"""
    sim = cosine_similarity(features)
    edges = []
    for i in range(sim.shape[0]):
        topk = np.argsort(sim[i])[-k:]
        for j in topk:
            if i != j:
                edges.append((i, j))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

def build_synthetic_patient_graph(num_patients, k=5):
    """
    Generates a synthetic random graph (PSN).
    Each node connects to 'k' other random nodes.
    """
    edges = []
    for i in range(num_patients):
        possible_neighbors = [j for j in range(num_patients) if j != i]
        if len(possible_neighbors) <= k:
            neighbors = possible_neighbors
        else:
            neighbors = random.sample(possible_neighbors, k)
        
        for j in neighbors:
            edges.append((i, j))
            
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    print(f"Built synthetic patient graph with {num_patients} nodes.")
    return edge_index

def build_synthetic_pathway_graph(num_nodes=500, num_edges=2000):
    """
    Generates a synthetic random graph (pathway graph).
    """
    edges = set()
    while len(edges) < num_edges:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u != v:
            edges.add((u, v))
            
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    print(f"Built synthetic pathway graph with {num_nodes} nodes and {num_edges} edges.")
    return edge_index, num_nodes