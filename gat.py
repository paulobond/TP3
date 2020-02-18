import torch
import torch.nn.functional as F
from torch import nn


class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        z = self.fc(h)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs))


class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads, num_layers=2):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads))
        out_prev = hidden_dim*num_heads

        for i in range(num_layers-2):
            self.layers.append(MultiHeadGATLayer(g, out_prev, hidden_dim, num_heads))
            out_prev = hidden_dim * num_heads

        self.layers.append(MultiHeadGATLayer(g, out_prev, out_dim, 1))

    def forward(self, h):
        for layer in self.layers[:-1]:
            h = layer(h)
            h = F.elu(h)
        h = self.layers[-1](h)
        return h

    def set_g(self, g):
        for layer in self.layers:
            for head in layer.heads:
                head.g = g
