import torch
import torch.nn.functional as F
from models.hypergat import HyperGAT
from models.h_gat import HeteroGAT
from models.fusion_mlp import MLPClassifier
from data_loader import load_omics
from data_loader import build_pathway_graph
# Import the new function
from utils import set_seed, build_patient_graph, build_synthetic_patient_graph
import yaml

def train():
    cfg = yaml.safe_load(open("configs.yaml"))
    set_seed(cfg["seed"])

    # --- Updated Data Loading ---
    # These paths are now read from the updated configs.yaml
    data_paths = cfg["data_paths"]
    
    # Load real gene, somatic, cnv data; generate synthetic mirna, clinical
    gene, somatic, cnv, mirna, clinical = load_omics(
        gene_path=data_paths["gene_expr"],
        somatic_path=data_paths["somatic_expr"],
        cnv_path=data_paths["cnv_expr"]
    )
    
    # Get patient count from your real gene data
    num_patients = gene.size(0)
    
    # Build pathway graph (still uses pathway_db file)
    edge_index_pathway, entity_to_idx = build_pathway_graph(data_paths["pathway_db"])
    # --- End Updated Data Loading ---

    # Initialize models (parameters loaded from config)
    hypergat = HyperGAT(cfg["model_params"]["hypergat"]["in_channels"], 
                        cfg["model_params"]["hypergat"]["out_channels"], 
                        heads=cfg["model_params"]["hypergat"]["heads"])
    hgat = HeteroGAT(cfg["model_params"]["hgat"]["in_channels"], 
                      cfg["model_params"]["hgat"]["out_channels"], 
                      heads=cfg["model_params"]["hgat"]["heads"])
    
    mlp_in_dim = cfg["model_params"]["hypergat"]["out_channels"] + \
                 cfg["model_params"]["hgat"]["out_channels"]
    
    mlp = MLPClassifier(mlp_in_dim, 
                        cfg["model_params"]["mlp"]["hidden_dim"], 
                        dropout=cfg["model_params"]["mlp"]["dropout"])

    optimizer = torch.optim.Adam(
        list(hypergat.parameters()) + list(hgat.parameters()) + list(mlp.parameters()),
        lr=cfg["train_params"]["lr"]
    )

    # --- Toy data fusion (with Synthetic PSN) ---
    
    # 1. Get pathway/gene embeddings (same as before)
    pathway_node_features = torch.randn(len(entity_to_idx), cfg["model_params"]["hypergat"]["in_channels"])
    gene_embed = hypergat(pathway_node_features, edge_index_pathway)

    # 2. Get patient embeddings (USING SYNTHETIC PSN)
    
    # 2a. Generate synthetic patient graph structure
    # This fulfills the "synthetic data for psn only" request
    patient_edge_index = build_synthetic_patient_graph(num_patients, k=5)
    
    # 2b. Generate synthetic features for patient nodes
    # This was already synthetic, which also fits the request
    patient_features = torch.randn(num_patients, cfg["model_params"]["hgat"]["in_channels"])
    
    # 2c. Pass synthetic graph and synthetic features to hgat (PSN model)
    patient_embed = hgat(patient_features, patient_edge_index)

    # 3. Fuse embeddings (same as before)
    fused = torch.cat([gene_embed.mean(0).expand(patient_embed.size(0), -1), patient_embed], dim=1)
    
    # 4. Generate toy labels (same as before)
    labels = torch.randint(0, 5, (fused.size(0),)) # Assuming 5 classes for MLP output

    # Training loop
    for epoch in range(cfg["train_params"]["epochs"]):
        optimizer.zero_grad()
        out = mlp(fused)
        loss = F.cross_entropy(out, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()