import torch
import torch.nn.functional as F
from models.hypergat import HyperGAT
from models.h_gat import HeteroGAT
from models.fusion_mlp import MLPClassifier
from data_loader import load_omics
from data_loader import build_pathway_graph
from utils import set_seed, build_patient_graph
import yaml

def train():
    cfg = yaml.safe_load(open("configs.yaml"))
    set_seed(cfg["seed"])

    gene, mirna, clinical = load_omics(**cfg["data_paths"])
    edge_index_pathway, entity_to_idx = build_pathway_graph(cfg["data_paths"]["pathway_db"])

    # Initialize models
    hypergat = HyperGAT(128, 64)
    hgat = HeteroGAT(64, 32)
    mlp = MLPClassifier(64+32, 128)

    optimizer = torch.optim.Adam(
        list(hypergat.parameters()) + list(hgat.parameters()) + list(mlp.parameters()),
        lr=cfg["train_params"]["lr"]
    )

    # Toy data fusion
    gene_embed = hypergat(torch.randn(len(entity_to_idx), 128), edge_index_pathway)
    patient_embed = hgat(torch.randn(gene.size(0), 64), build_patient_graph(gene.numpy()))

    fused = torch.cat([gene_embed.mean(0).expand(patient_embed.size(0), -1), patient_embed], dim=1)
    labels = torch.randint(0, 5, (fused.size(0),))

    for epoch in range(cfg["train_params"]["epochs"]):
        optimizer.zero_grad()
        out = mlp(fused)
        loss = F.cross_entropy(out, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()
