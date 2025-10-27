import torch
from torch import nn
from torch_geometric.nn import GATConv

class HyperGAT(nn.Module):
    """
    Processes pathway hypergraph for gene/miRNA embeddings.
    """
    def __init__(self, in_channels, out_channels, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_channels, out_channels, heads=heads)
        self.gat2 = GATConv(out_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = torch.relu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x
