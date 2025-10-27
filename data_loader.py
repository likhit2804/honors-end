# In data_loader.py

import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import itertools
from collections import defaultdict

def standardize_patient_id(pid):
    """Helper function to potentially standardize patient IDs."""
    # Example: Convert to uppercase, replace dots with hyphens, strip whitespace
    return str(pid).upper().replace('.', '-').strip()

def load_omics(mirna_path, somatic_path, cnv_path, clinical_path):
    """
    Loads real mirna, somatic, cnv, clinical data. Aligns patients robustly.
    Processes one-hot encoded 'PAM50Call_RNAseq_*' columns (or fallback) into labels.
    Returns features, labels, and the label encoder.
    """
    try:
        # Load all data files
        print("Loading data files...")
        mirna_df = pd.read_csv(mirna_path, index_col=0) #
        somatic_df = pd.read_csv(somatic_path, index_col=0) #
        cnv_df = pd.read_csv(cnv_path, index_col=0) #
        clinical_df = pd.read_csv(clinical_path, index_col=0) #
        print("Data files loaded.")

        # --- Standardize IDs (Optional) ---
        # print("Standardizing patient IDs (if uncommented)...")
        # clinical_df.index = clinical_df.index.map(standardize_patient_id)
        # mirna_df.columns = mirna_df.columns.map(standardize_patient_id)
        # somatic_df.columns = somatic_df.columns.map(standardize_patient_id)
        # cnv_df.columns = cnv_df.columns.map(standardize_patient_id)
        # print("Standardization complete (if uncommented).")
        # --- End Standardization ---

        # --- Process Labels ---
        print("Processing labels...")
        pam50_cols = [
            'PAM50Call_RNAseq_Her2', 'PAM50Call_RNAseq_LumA',
            'PAM50Call_RNAseq_LumB', 'PAM50Call_RNAseq_Normal',
            'PAM50Call_RNAseq_Basal' # Add Basal just in case
        ] #
        present_pam50_cols = [col for col in pam50_cols if col in clinical_df.columns] #
        label_column_name_single = 'PAM50Call_RNAseq'
        label_source_for_drop = []

        if present_pam50_cols:
            print(f"Found one-hot PAM50 columns: {present_pam50_cols}") #
            pam50_df = clinical_df[present_pam50_cols] #
            # Handle potential multiple '1's or all '0's per row before idxmax
            valid_one_hot = pam50_df.sum(axis=1) == 1
            if not valid_one_hot.all():
                 invalid_count = (~valid_one_hot).sum()
                 print(f"Warning: {invalid_count} rows do not have exactly one PAM50 label set. Dropping these rows.")
                 clinical_df = clinical_df[valid_one_hot]
                 pam50_df = pam50_df[valid_one_hot] # Re-filter pam50_df

            if pam50_df.empty:
                raise ValueError("No valid one-hot PAM50 labels found after filtering.")

            single_col_labels = pam50_df.idxmax(axis=1).str.replace('PAM50Call_RNAseq_', '', regex=False) #
            labels_df = pd.DataFrame({'PAM50_Subtype': single_col_labels}, index=pam50_df.index) # Use index from pam50_df now
            label_source_for_drop = present_pam50_cols #

        elif label_column_name_single in clinical_df.columns:
            print(f"Warning: Using single label column '{label_column_name_single}'.") #
            original_len = len(clinical_df)
            clinical_df.dropna(subset=[label_column_name_single], inplace=True) #
            if len(clinical_df) < original_len:
                print(f"Dropped {original_len - len(clinical_df)} patients due to missing labels.")
            if clinical_df.empty:
                 raise ValueError("No patients remaining after dropping missing labels from single column.")
            labels_df = clinical_df[[label_column_name_single]].copy() #
            label_source_for_drop = [label_column_name_single] #
        else:
            raise ValueError("No suitable PAM50 label column found ('PAM50Call_RNAseq_*' or 'PAM50Call_RNAseq').") #

        # --- Robust Patient Alignment ---
        print("\n--- Aligning Patients ---")
        # Use index from clinical_df AFTER label processing/filtering
        clinical_ids_with_labels = clinical_df.index
        mirna_ids = mirna_df.columns
        somatic_ids = somatic_df.columns
        cnv_ids = cnv_df.columns

        common_ids_list = list(set(clinical_ids_with_labels) & set(mirna_ids) & set(somatic_ids) & set(cnv_ids)) #
        common_ids = pd.Index(sorted(common_ids_list)) #

        if len(common_ids) == 0:
            print("Clinical Index (with labels) Sample:", list(clinical_ids_with_labels[:5])) #
            print("miRNA Columns Sample:", list(mirna_ids[:5])) #
            print("Somatic Columns Sample:", list(somatic_ids[:5])) #
            print("CNV Columns Sample:", list(cnv_ids[:5])) #
            raise ValueError("No common patient IDs found across all data files after label processing. Check ID formats and file contents.") #

        print(f"Found {len(common_ids)} common patients across all files.") #

        # Filter all dataframes to keep only common patients IN THE SAME ORDER
        clinical_df_aligned = clinical_df.loc[common_ids] #
        labels_df_aligned = labels_df.loc[common_ids] # Filter labels_df created earlier #
        mirna_df_aligned = mirna_df.loc[:, common_ids] #
        somatic_df_aligned = somatic_df.loc[:, common_ids] #
        cnv_df_aligned = cnv_df.loc[:, common_ids] #
        print("Alignment complete.")
        # --- End Alignment ---

        # --- Encode Final Labels ---
        print("Encoding labels...")
        label_col_to_encode = labels_df_aligned.columns[0]
        encoder = LabelEncoder() #
        numerical_labels = encoder.fit_transform(labels_df_aligned[label_col_to_encode]) #
        labels = torch.tensor(numerical_labels, dtype=torch.long) #

        print("\n--- Label Encoding (Post-Alignment) ---")
        for i, class_name in enumerate(encoder.classes_): #
            print(f"  Class '{class_name}' mapped to label {i}") #
        print(f"Total labels encoded: {len(labels)}")
        # --- End Label Encoding ---

        # --- Normalize Features (AFTER Alignment) ---
        print("Normalizing features...")
        scaler_mirna = StandardScaler() #
        mirna = torch.tensor(scaler_mirna.fit_transform(mirna_df_aligned.T), dtype=torch.float) #

        scaler_somatic = StandardScaler() #
        somatic = torch.tensor(scaler_somatic.fit_transform(somatic_df_aligned.T), dtype=torch.float) #

        scaler_cnv = StandardScaler() #
        cnv = torch.tensor(scaler_cnv.fit_transform(cnv_df_aligned.T), dtype=torch.float) #

        # Prepare clinical features (drop original label columns)
        clinical_features_df = clinical_df_aligned.drop(columns=label_source_for_drop, errors='ignore') #

        scaler_clinical = StandardScaler() #
        try:
            numeric_cols = clinical_features_df.select_dtypes(include=np.number).columns #
            if len(numeric_cols) < len(clinical_features_df.columns):
                non_numeric_cols = clinical_features_df.select_dtypes(exclude=np.number).columns #
                print(f"Warning: Dropping non-numeric clinical feature columns: {list(non_numeric_cols)}") #
                clinical_features_df = clinical_features_df[numeric_cols] #

            if not clinical_features_df.empty:
                 clinical_features_scaled = scaler_clinical.fit_transform(clinical_features_df) #
                 clinical = torch.tensor(clinical_features_scaled, dtype=torch.float) #
            else:
                 print("Warning: No numeric clinical features remain. Creating zero tensor.") #
                 clinical = torch.zeros((len(common_ids), 0), dtype=torch.float) #

        except ValueError as e:
            print(f"\nError scaling clinical data: {e}. Check for non-numeric data.") #
            clinical = torch.zeros((len(common_ids), len(clinical_features_df.columns)), dtype=torch.float) #
        print("Normalization complete.")

        # Final check of shapes
        num_patients = len(common_ids) #
        if not (mirna.shape[0] == num_patients and somatic.shape[0] == num_patients and cnv.shape[0] == num_patients and clinical.shape[0] == num_patients and labels.shape[0] == num_patients): #
            print("\n--- SHAPE MISMATCH POST-ALIGNMENT ---")
            print(f"Expected patients: {num_patients}")
            print(f"Labels shape: {labels.shape}")
            print(f"miRNA shape: {mirna.shape}")
            print(f"Somatic shape: {somatic.shape}")
            print(f"CNV shape: {cnv.shape}")
            print(f"Clinical shape: {clinical.shape}")
            raise RuntimeError("Data shape mismatch.") #

        print(f"\nLoaded and aligned real data ({num_patients} patients):") #
        print(f"  miRNA Features: {mirna.shape[1]}") #
        print(f"  Somatic Features: {somatic.shape[1]}") #
        print(f"  CNV Features: {cnv.shape[1]}") #
        print(f"  Clinical Features: {clinical.shape[1]}") #
        print(f"  Labels shape: {labels.shape}") #

        # ****** CORRECTED RETURN STATEMENT ******
        return mirna, somatic, cnv, clinical, labels, encoder # Now returns 6 items

    except FileNotFoundError as e:
        print(f"Error loading data: File not found - {e}") #
        raise
    except ValueError as e:
         print(f"Data Processing Error: {e}")
         raise
    except Exception as e:
        print(f"An unexpected error occurred during data loading/processing: {e}") #
        raise


# --- build_pathway_graph function ---
# (No changes needed here based on the error)
def build_pathway_graph(pathway_path):
    print("Building pathway graph...")
    try:
        df = pd.read_csv(pathway_path) #
        hyperedges = defaultdict(list) #
        for _, row in df.iterrows(): #
            # Basic check for valid entity ID format if needed
            if pd.notna(row["Entity_ID"]) and row["Entity_ID"] != '':
                 hyperedges[row["Pathway_ID"]].append(row["Entity_ID"]) #

        entities_in_pathways = set(e for members in hyperedges.values() for e in members)
        if not entities_in_pathways:
             print("Warning: No entities found in pathway definitions.")
             return torch.empty((2, 0), dtype=torch.long), {} # Return empty graph and map

        entities = sorted(list(entities_in_pathways)) #
        entity_to_idx = {e: i for i, e in enumerate(entities)} #

        edges = [] #
        for members in hyperedges.values(): #
            valid_members = [m for m in members if m in entity_to_idx]
            for a, b in itertools.combinations(valid_members, 2): #
                edges.append((entity_to_idx[a], entity_to_idx[b])) #

        if not edges:
            print("Warning: No edges were created for the pathway graph.") #
            edge_index = torch.empty((2, 0), dtype=torch.long) #
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() #

        print(f"Built pathway graph (via clique expansion) with {len(entity_to_idx)} nodes and {edge_index.shape[1]} edges.") #
        return edge_index, entity_to_idx #
    except FileNotFoundError:
         print(f"Error: Pathway file not found at {pathway_path}")
         raise
    except Exception as e:
         print(f"Error building pathway graph: {e}")
         raise