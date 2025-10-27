import torch
from torch import nn
from torch_geometric.nn import GATConv

class HeteroGAT(nn.Module):
    """
    Patient Similarity Network with heterophilic edges.
    """
    def __init__(self, in_channels, out_channels, heads=2):
        super().__init__()
        self.layer1 = GATConv(in_channels, out_channels, heads=heads)
        self.layer2 = GATConv(out_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = torch.relu(self.layer1(x, edge_index))
        x = self.layer2(x, edge_index)
        return x
