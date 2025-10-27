# Multi-Omics Hypergraph Pipeline


Files:
- `configs.py` - experiment config
- `data_loader.py` - dataset & preprocessing utilities
- `models/hypergat.py` - Hypergraph Attention Network for pathway hypergraph
- `models/h_gat.py` - Heterogeneous GAT for patient similarity network
- `models/fusion_mlp.py` - fusion + classifier
- `train.py` - training loop, evaluation, checkpointing
- `utils.py` - helpers (metrics, seed, device)
- `explainers.py` - in-hoc attention extraction and post-hoc SHAP/LIME hooks (skeleton)


Usage:
```bash
python train.py --config configs.py