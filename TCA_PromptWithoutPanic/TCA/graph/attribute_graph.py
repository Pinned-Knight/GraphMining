import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """Single Graph Attention layer implemented with edge lists."""

    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
        super().__init__()
        if out_dim % num_heads != 0:
            raise ValueError("out_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.zeros(num_heads, 2 * self.head_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)

    def forward(self, h, edge_index):
        """
        h: (N, in_dim)
        edge_index: (2, E) with src->dst edges
        """
        n_nodes = h.size(0)
        wh = self.W(h).view(n_nodes, self.num_heads, self.head_dim)

        if edge_index.numel() == 0:
            return wh.reshape(n_nodes, -1)

        src, dst = edge_index
        alpha_input = torch.cat([wh[src], wh[dst]], dim=-1)
        e = F.leaky_relu((alpha_input * self.a).sum(-1), negative_slope=0.2)

        out = torch.zeros_like(wh)
        for node in range(n_nodes):
            node_mask = dst == node
            if not node_mask.any():
                continue
            scores = e[node_mask]
            weights = torch.softmax(scores, dim=0)
            weights = self.dropout(weights)
            out[node] = (weights.unsqueeze(-1) * wh[src[node_mask]]).sum(dim=0)

        return out.reshape(n_nodes, -1)


class AttributeGAT(nn.Module):
    """L-layer GAT for attribute refinement."""

    def __init__(self, clip_dim=512, hidden_dim=512, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        in_dim = clip_dim

        for layer_idx in range(num_layers):
            out_dim = clip_dim if layer_idx == num_layers - 1 else hidden_dim
            self.layers.append(GATLayer(in_dim, out_dim, num_heads=num_heads, dropout=dropout))
            in_dim = out_dim

        self.residual_proj = nn.Linear(clip_dim, clip_dim, bias=False)
        nn.init.eye_(self.residual_proj.weight)

    def forward(self, h, edge_index):
        residual = h
        for layer in self.layers:
            h = F.elu(layer(h, edge_index))
        return F.normalize(h + self.residual_proj(residual), dim=-1)
