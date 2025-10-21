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
    sim = cosine_similarity(features)
    edges = []
    for i in range(sim.shape[0]):
        topk = np.argsort(sim[i])[-k:]
        for j in topk:
            if i != j:
                edges.append((i, j))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index
