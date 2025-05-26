import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GATNet(nn.Module):
    def __init__(self, in_channels_hrtf=129, in_channels_pos=15, hidden_channels=64, out_channels=129,
                 edge_dim_hrtf=1, edge_dim_pos=1, heads=8, dropout=0.0):
        super(GATNet, self).__init__()

        # GAT for HRTF graph
        self.gat1_hrtf = GATConv(in_channels_hrtf, hidden_channels, heads=heads, dropout=dropout,
                                 edge_dim=edge_dim_hrtf)
        self.gat2_hrtf = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout,
                                 edge_dim=edge_dim_hrtf)

        # GAT for Position graph
        self.gat1_pos = GATConv(in_channels_pos, hidden_channels, heads=heads, dropout=dropout, edge_dim=edge_dim_pos)
        self.gat2_pos = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout,
                                edge_dim=edge_dim_pos)

        self.fc = nn.Sequential(
            nn.Linear(out_channels*2, 129)
        )

    def forward(self, data_hrtf, data_pos):
        # Process HRTF graph
        x_hrtf, edge_index_hrtf, edge_attr_hrtf = data_hrtf.x, data_hrtf.edge_index, data_hrtf.edge_attr
        x_hrtf = self.gat1_hrtf(x_hrtf, edge_index_hrtf, edge_attr_hrtf)
        x_hrtf = F.elu(x_hrtf)
        x_hrtf = self.gat2_hrtf(x_hrtf, edge_index_hrtf, edge_attr_hrtf)

        # Process Position graph
        x_pos, edge_index_pos, edge_attr_pos = data_pos.x, data_pos.edge_index, data_pos.edge_attr
        x_pos = self.gat1_pos(x_pos, edge_index_pos, edge_attr_pos)
        x_pos = F.elu(x_pos)
        x_pos = self.gat2_pos(x_pos, edge_index_pos, edge_attr_pos)

        batch = data_hrtf.batch
        batch_size = data_hrtf.num_graphs
        num_nodes = batch.size(0)
        node_counts = torch.bincount(batch)
        last_node_indices = torch.cumsum(node_counts, dim=0) - 1
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[last_node_indices] = True
        x_hrtf_last = x_hrtf[mask]
        x_pos_last = x_pos[mask]
        x = torch.cat([x_hrtf_last, x_pos_last], dim=-1)
        x = self.fc(x)

        return x.unsqueeze(1)
