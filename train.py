import torch
import torch.nn as nn # --- NEW ---
import torch.nn.functional as F
from models.hypergat import HyperGAT
from models.h_gat import HeteroGAT
from models.fusion_mlp import MLPClassifier
from data_loader import load_omics
from data_loader import build_pathway_graph
# --- CHANGED ---
# Import the function for building the graph from real features
from utils import set_seed, build_patient_graph
import yaml

def train():
    cfg = yaml.safe_load(open("configs.yaml"))
    set_seed(cfg["seed"])

    # In train.py (lines 20-30)

    # --- Updated Data Loading ---
    # These paths are now read from the updated configs.yaml
    data_paths = cfg["data_paths"]
    
    # Load all 4 real data types
    mirna, somatic, cnv, clinical = load_omics(
        mirna_path=data_paths["mirna_expr"],
        somatic_path=data_paths["somatic_expr"],
        cnv_path=data_paths["cnv_expr"],
        clinical_path=data_paths["clinical_data"]
    )
    
    # Get patient count from your real data
    num_patients = mirna.size(0)
    
    # Build pathway graph
    edge_index_pathway, entity_to_idx = build_pathway_graph(data_paths["pathway_db"])
    # --- End Updated Data Loading ---
    # --- End Updated Data Loading ---

    # Initialize models
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

    # --- NEW: Feature Projection Layer ---
    # Concatenate all omics features to create the "Feature Vector"
    omics_features = torch.cat([ somatic, cnv, mirna, clinical], dim=1)
    
    total_feature_dim = omics_features.shape[1]
    hgat_in_channels = cfg["model_params"]["hgat"]["in_channels"]
    
    # This layer maps the large concatenated feature vector
    # to the input dimension expected by the H-GAT model.
    feature_projector = nn.Linear(total_feature_dim, hgat_in_channels)
    # --- End New Feature Projection ---

    optimizer = torch.optim.Adam(
        list(hypergat.parameters()) + 
        list(hgat.parameters()) + 
        list(mlp.parameters()) +
        list(feature_projector.parameters()), # --- NEW: Add projector to optimizer
        lr=float(cfg["train_params"]["lr"])
    )

    # --- Data fusion (USING REAL PSN) ---
    
    # 1. Get pathway/gene embeddings (same as before)
    # These features are learnable embeddings for the entities in the pathway graph
    pathway_node_features = torch.randn(len(entity_to_idx), cfg["model_params"]["hypergat"]["in_channels"])
    gene_embed = hypergat(pathway_node_features, edge_index_pathway)

    # 2. Get patient embeddings (USING REAL DATA AND PSN)
    
    # 2a. Project omics data to H-GAT input dimension
    # This 'patient_features_projected' is the 's1_featvec' from your diagram
    patient_features_projected = feature_projector(omics_features)
    
    # 2b. Build real patient graph structure from projected features
    # (Using .detach() as graph structure is not trained via backprop)
    patient_edge_index = build_patient_graph(
        patient_features_projected.detach().cpu().numpy(), 
        k=cfg["train_params"]["psn_k"] # Assumes you add 'psn_k' to configs.yaml
    )
    print(f"Built real patient graph with {num_patients} nodes: {patient_edge_index.shape}")
    
    # 2c. Pass real features and real graph to hgat (PSN model)
    patient_embed = hgat(patient_features_projected, patient_edge_index)

    # 3. Fuse embeddings (same as before)
    fused = torch.cat([gene_embed.mean(0).expand(patient_embed.size(0), -1), patient_embed], dim=1)
    
    # 4. Generate toy labels (same as before)
    # --- NOTE ---
    # For a real implementation, you would load real labels (e.g., from clinical_df)
    # in load_omics and use them here instead of torch.randint
    labels = torch.randint(0, 5, (fused.size(0),)) # Assuming 5 classes

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