import torch
import shap

def extract_attention_weights(model):
    """
    In-Hoc path:
    Extracts learned attention from HyperGAT for GSEA.
    """
    att = []
    # --- FIX: Access the stored attention attributes ---
    if hasattr(model, "alpha1") and model.alpha1 is not None:
        att.append(model.alpha1)
    if hasattr(model, "alpha2") and model.alpha2 is not None:
        att.append(model.alpha2)
    # --- End Fix ---

    # This will now succeed, assuming a forward pass has been run
    return torch.cat(att, dim=0)


def explain_with_shap(model, inputs):
    """
    Post-Hoc path:
    Model-agnostic explanation of MLP outputs.
    """
    explainer = shap.DeepExplainer(model, inputs)
    shap_values = explainer.shap_values(inputs)
    return shap_values