# In train.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np # Import numpy
import sys # For exiting cleanly

# Assuming models are in './models/' directory relative to train.py
# Make sure these paths are correct relative to where you run the script
try:
    from models.hypergat import HyperGAT
    from models.h_gat import HeteroGAT
    from models.fusion_mlp import MLPClassifier
except ImportError as e:
    print(f"Error importing models: {e}")
    print("Ensure models (hypergat.py, h_gat.py, fusion_mlp.py) exist in a 'models' subdirectory.")
    sys.exit(1) # Exit if models can't be imported

# Assuming data_loader and utils are in the same directory or accessible
try:
    from data_loader import load_omics, build_pathway_graph
    from utils import set_seed, build_patient_graph # Assuming build_patient_graph is correctly defined
except ImportError as e:
    print(f"Error importing helpers: {e}")
    print("Ensure data_loader.py and utils.py are in the same directory or Python path.")
    sys.exit(1) # Exit if helpers can't be imported

def train():
    # --- Load Configuration ---
    try:
        cfg = yaml.safe_load(open("configs.yaml"))
        print("✅ Configs loaded successfully.")
    except FileNotFoundError:
        print("❌ Error: configs.yaml not found in the current directory.")
        return
    except Exception as e:
        print(f"❌ Error loading configs.yaml: {e}")
        return

    # --- Setup Seed and Device ---
    seed = cfg.get("seed", 42)
    set_seed(seed)
    print(f"🌱 Seed set to: {seed}")
    # Force CPU as requested in the input code
    device = torch.device("cpu")
    print(f"⚙️ Running on device: {device}")


    # --- Load Data ---
    print("\n--- Loading & Processing Data ---")
    data_paths = cfg.get("data_paths", {})
    try:
        # Capture the returned labels AND encoder (6 values total)
        mirna, somatic, cnv, clinical, labels, label_encoder = load_omics(
            mirna_path=data_paths.get("mirna_expr"),
            somatic_path=data_paths.get("somatic_expr"),
            cnv_path=data_paths.get("cnv_expr"),
            clinical_path=data_paths.get("clinical_data")
        )
        print("✅ Data loaded and processed.")
    except Exception as e:
        print(f"❌ FATAL ERROR during data loading: {e}")
        return # Stop execution

    num_patients = mirna.size(0)
    print(f"Number of patients loaded and aligned: {num_patients}")
    if num_patients == 0:
        print("❌ Error: No patients loaded after alignment/filtering. Exiting.")
        return

    # --- Build Pathway Graph ---
    try:
         pathway_db_path = data_paths.get("pathway_db")
         if not pathway_db_path:
             raise ValueError("pathway_db path missing in configs.yaml")
         edge_index_pathway, entity_to_idx = build_pathway_graph(pathway_db_path)
         num_pathway_nodes = len(entity_to_idx)
         if num_pathway_nodes == 0:
              print("⚠️ Warning: Pathway graph has 0 nodes. HyperGAT path will be skipped.")
         print("✅ Pathway graph built.")
    except Exception as e:
         print(f"❌ FATAL ERROR building pathway graph: {e}")
         return


    # --- Move Data to Device ---
    print(f"Moving data to device ({device})...")
    try:
        mirna = mirna.to(device)
        somatic = somatic.to(device)
        cnv = cnv.to(device)
        clinical = clinical.to(device) # May have 0 features, shape (N, 0)
        labels = labels.to(device)
        edge_index_pathway = edge_index_pathway.to(device)
        print("✅ Data moved to device.")
    except Exception as e:
        print(f"❌ Error moving data to device {device}: {e}")
        return

    # --- Initialize Models ---
    print("\n--- Initializing Models ---")
    model_params = cfg.get("model_params", {})
    try:
        # HyperGAT
        hypergat_cfg = model_params.get("hypergat", {})
        hypergat_in = hypergat_cfg.get("in_channels", 128) # Default if missing
        hypergat_out = hypergat_cfg.get("out_channels", 64)
        hypergat_heads = hypergat_cfg.get("heads", 4)
        hypergat = HyperGAT(hypergat_in, hypergat_out, heads=hypergat_heads).to(device)
        print(f"  HyperGAT initialized: In={hypergat_in}, Out={hypergat_out}, Heads={hypergat_heads}")

        # Feature Projector
        hgat_cfg = model_params.get("hgat", {})
        hgat_in_channels_config = hgat_cfg.get("in_channels", 64) # Default if missing
        total_feature_dim = mirna.shape[1] + somatic.shape[1] + cnv.shape[1] + clinical.shape[1]
        print(f"  Total concatenated feature dimension: {total_feature_dim}")
        if total_feature_dim == 0:
             print("⚠️ Warning: Total feature dimension is 0. Check loaded data files and processing.")
             # Consider raising an error if no features are loaded
        feature_projector = nn.Linear(total_feature_dim, hgat_in_channels_config).to(device)
        print(f"  Feature Projector initialized: In={total_feature_dim}, Out={hgat_in_channels_config}")

        # HGAT
        hgat_out = hgat_cfg.get("out_channels", 32)
        hgat_heads = hgat_cfg.get("heads", 2)
        hgat = HeteroGAT(hgat_in_channels_config, hgat_out, heads=hgat_heads).to(device)
        print(f"  HGAT initialized: In={hgat_in_channels_config}, Out={hgat_out}, Heads={hgat_heads}")

        # MLP
        mlp_cfg = model_params.get("mlp", {})
        # Assuming mean aggregation for HyperGAT output and single head output for HGAT's final layer
        mlp_in_dim = hypergat_out + hgat_out
        num_classes = len(torch.unique(labels))
        print(f"  Number of unique classes found: {num_classes}")
        if num_classes <= 1:
            print("⚠️ Warning: <= 1 class found. Check label processing. Setting num_classes=2 for MLP.")
            num_classes = 2 # Adjust as needed

        mlp_hidden = mlp_cfg.get("hidden_dim", 128)
        mlp_dropout = mlp_cfg.get("dropout", 0.3)
        mlp = MLPClassifier(
            mlp_in_dim,
            mlp_hidden,
            out_dim=num_classes,
            dropout=mlp_dropout
        ).to(device)
        print(f"  MLP initialized: In={mlp_in_dim}, Hidden={mlp_hidden}, Out={num_classes}, Dropout={mlp_dropout}")
        print("✅ Models initialized successfully.")

    except KeyError as e:
        print(f"❌ Error initializing models: Missing parameter in configs.yaml - {e}")
        return
    except Exception as e:
        print(f"❌ Error initializing models: {e}")
        return

    # --- Optimizer ---
    print("\n--- Setting up Optimizer ---")
    train_params = cfg.get("train_params", {})
    try:
        optimizer_name = train_params.get("optimizer", "adam").lower()
        lr = float(train_params.get("lr", 1e-3))

        params_to_optimize = list(hypergat.parameters()) + \
                             list(hgat.parameters()) + \
                             list(mlp.parameters()) + \
                             list(feature_projector.parameters())

        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
        # Add elif for other optimizers if needed (e.g., SGD)
        else:
            print(f"⚠️ Warning: Unknown optimizer '{optimizer_name}'. Defaulting to Adam.")
            optimizer = torch.optim.Adam(params_to_optimize, lr=lr)

        print(f"✅ Optimizer: {optimizer_name.capitalize()}, Learning Rate: {lr}")
    except Exception as e:
        print(f"❌ Error setting up optimizer: {e}")
        return


    # --- Static Inputs ---
    # Pathway node features (Randomly initialized)
    pathway_node_features = torch.randn(
        num_pathway_nodes, # Use actual number of nodes from graph building
        hypergat_in,
        device=device
    ) if num_pathway_nodes > 0 else torch.empty((0, hypergat_in), device=device) # Handle empty graph case

    # Build patient graph (computed ONCE)
    print("\n--- Building Patient Similarity Network ---")
    # Concatenate features on the correct device for projection
    omics_features_dev = torch.cat([somatic, cnv, mirna, clinical], dim=1).to(device)
    with torch.no_grad():
        initial_patient_features_dev = feature_projector(omics_features_dev)
        # Move projected features to CPU for numpy/build_patient_graph if necessary
        initial_patient_features_cpu_np = initial_patient_features_dev.cpu().numpy()

    try:
        psn_k = train_params.get("psn_k", 5)
        # Ensure utils.build_patient_graph accepts numpy and returns torch tensor
        patient_edge_index = build_patient_graph(
            initial_patient_features_cpu_np,
            k=psn_k
        ).to(device) # Move edge index back to target device
        print(f"✅ Built patient graph with {num_patients} nodes and {patient_edge_index.shape[1]} edges using k={psn_k}.")
    except Exception as e:
        print(f"❌ Error building patient graph: {e}")
        return

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    epochs = train_params.get("epochs", 50)

    hypergat.train()
    hgat.train()
    mlp.train()
    feature_projector.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        # --- FORWARD PASS ---
        try:
            # 1. Get pathway/gene embeddings (if graph exists)
            if num_pathway_nodes > 0 and edge_index_pathway.numel() > 0 :
                gene_embed = hypergat(pathway_node_features, edge_index_pathway)
                # Aggregate gene embeddings (using mean) and expand
                gene_embed_aggregated = gene_embed.mean(dim=0, keepdim=True).expand(num_patients, -1)
            else:
                 # Zero tensor if no pathway graph
                 gene_embed_aggregated = torch.zeros((num_patients, hypergat_out), device=device)

            # 2. Get patient embeddings
            patient_features_projected = feature_projector(omics_features_dev) # Use pre-computed features
            patient_embed = hgat(patient_features_projected, patient_edge_index)

            # 3. Fuse embeddings
            fused = torch.cat([gene_embed_aggregated, patient_embed], dim=1)

            # 4. Get classification output
            out = mlp(fused)
        except Exception as e:
            print(f"\n❌ Error during forward pass epoch {epoch+1}: {e}")
            # Optionally add traceback: import traceback; traceback.print_exc()
            continue # Skip to next epoch

        # --- END FORWARD PASS ---

        # Calculate loss using REAL labels
        try:
            loss = F.cross_entropy(out, labels)
        except Exception as e:
            print(f"\n❌ Error calculating loss epoch {epoch+1}: {e}")
            print(f"   Output shape: {out.shape}, Labels shape: {labels.shape}")
            print(f"   Labels min: {labels.min()}, Labels max: {labels.max()}, Num Classes (MLP): {num_classes}")
            continue # Skip epoch if loss fails

        # Backward pass and optimizer step
        try:
            loss.backward()
            optimizer.step()
        except Exception as e:
            print(f"\n❌ Error during backward pass/optimizer step epoch {epoch+1}: {e}")
            continue # Skip epoch


        # Calculate accuracy (on training data)
        with torch.no_grad():
            _, predicted = torch.max(out, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / num_patients

        print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.4f}")

    print("--- Training Finished ---")

    # --- Check Predictions (Optional Evaluation on Training Data) ---
    print("\n--- Evaluating Final Model on Training Data ---")
    hypergat.eval() # Set models to evaluation mode
    hgat.eval()
    mlp.eval()
    feature_projector.eval()

    with torch.no_grad(): # Disable gradient calculations for inference
        try:
            # Re-run forward pass
            if num_pathway_nodes > 0 and edge_index_pathway.numel() > 0:
                 gene_embed = hypergat(pathway_node_features, edge_index_pathway)
                 gene_embed_aggregated = gene_embed.mean(dim=0, keepdim=True).expand(num_patients, -1)
            else:
                 gene_embed_aggregated = torch.zeros((num_patients, hypergat_out), device=device)

            patient_features_projected = feature_projector(omics_features_dev)
            patient_embed = hgat(patient_features_projected, patient_edge_index)
            fused = torch.cat([gene_embed_aggregated, patient_embed], dim=1)
            final_out = mlp(fused)

            _, final_predicted = torch.max(final_out, 1)

            # Move predictions and labels to CPU for printing/analysis
            final_predicted_cpu = final_predicted.cpu().numpy()
            labels_cpu = labels.cpu().numpy()

            # Print overall final accuracy on training data
            final_correct = (final_predicted == labels).sum().item()
            final_accuracy = final_correct / num_patients
            print(f"\n✅ Final Overall Accuracy on Training Set: {final_accuracy:.4f} ({final_correct}/{num_patients})")


            # Print predictions for the first few patients using the label encoder
            print("\n--- Sample Predictions (First 10 Patients) ---")
            print("Patient Index | Predicted Label Index | True Label Index | Predicted Name | True Name")
            print("-" * 75)
            for i in range(min(10, num_patients)):
                pred_idx = final_predicted_cpu[i]
                true_idx = labels_cpu[i]
                # Use the label_encoder returned by load_omics to get names
                try:
                    pred_name = label_encoder.classes_[pred_idx]
                    true_name = label_encoder.classes_[true_idx]
                except IndexError: # Handle case where index might be out of bounds if something went wrong
                    pred_name = "ERROR"
                    true_name = "ERROR"
                print(f" {i: <12} | {pred_idx: <20} | {true_idx: <15} | {pred_name: <14} | {true_name: <9}")

        except Exception as e:
            print(f"❌ Error during final prediction check: {e}")
            # Optionally add traceback: import traceback; traceback.print_exc()

    # --- END Prediction Check ---


if __name__ == "__main__":
    train()