import torch

def extract_attention_weights(model):
    """
    In-Hoc path:
    Extracts learned attention from HyperGAT for GSEA.
    """
    att = []
    for layer in model.modules():
        if hasattr(layer, "alpha"):
            att.append(layer.alpha)
    return torch.cat(att, dim=0)
import shap

def explain_with_shap(model, inputs):
    """
    Post-Hoc path:
    Model-agnostic explanation of MLP outputs.
    """
    explainer = shap.DeepExplainer(model, inputs)
    shap_values = explainer.shap_values(inputs)
    return shap_values
