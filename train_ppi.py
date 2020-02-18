import argparse
from os import path

import dgl.function as fn
import numpy as np
import torch
import torch as th
import torch.nn.functional as F
from dgl import batch
from dgl.data.ppi import LegacyPPIDataset
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import edge_softmax, GATConv
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

MODEL_STATE_FILE = path.join(path.dirname(path.abspath(__file__)), "model_state.pth")


# CODE FROM DGL

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """Return input"""
        return x


class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if in_feats != out_feats:
                self.res_fc = nn.Linear(in_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feat):
        r"""Compute graph attention network layer.
        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.
        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        """
        graph = graph.local_var()
        h = self.feat_drop(feat)
        feat = self.fc(h).view(-1, self._num_heads, self._out_feats)
        el = (feat * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat * self.attn_r).sum(dim=-1).unsqueeze(-1)
        graph.ndata.update({'ft': feat, 'el': el, 'er': er})
        # compute edge attention
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))
        # compute softmax
        graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
        # message passing
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.ndata['ft']
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h).view(h.shape[0], -1, self._out_feats)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst


class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):

        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l - 1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits

    def set_g(self, g):
        self.g = g


class BasicGraphModel(nn.Module):

    def __init__(self, g, n_layers, input_size, hidden_size, output_size, nonlinearity):
        super().__init__()

        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(input_size, hidden_size, activation=nonlinearity))
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(hidden_size, hidden_size, activation=nonlinearity))
        self.layers.append(GraphConv(hidden_size, output_size))

    def forward(self, inputs):
        outputs = inputs
        for i, layer in enumerate(self.layers):
            outputs = layer(self.g, outputs)
        return outputs


def main(args):
    # create the dataset
    train_dataset, test_dataset = LegacyPPIDataset(mode="train"), LegacyPPIDataset(mode="test")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    n_features, n_classes = train_dataset.features.shape[1], train_dataset.labels.shape[1]

    # create the model, loss function and optimizer
    device = torch.device("cpu" if args.gpu < 0 else "cuda:" + str(args.gpu))
    print(f"Using device: {device}")

    if args.model == 'gat':
        print(f"Using model GAT")
        print(f"Number of heads: {args.num_heads}")
        print(f"Hidden dim: {args.hidden_dim}")
        print(f"Save location {MODEL_STATE_FILE}")

        model = GAT(g=train_dataset.graph,
                    num_layers=args.num_layers,
                    in_dim=n_features,
                    num_hidden=args.hidden_dim,
                    num_classes=n_classes,
                    heads=[args.num_heads] * args.num_layers,
                    activation=None,
                    feat_drop=args.feat_drop,
                    attn_drop=args.att_drop,
                    negative_slope=0.2,
                    residual=True).to(device)
    else:
        print(f"Using base model (GCN)")
        model = BasicGraphModel(g=train_dataset.graph, n_layers=2, input_size=n_features,
                                hidden_size=256, output_size=n_classes, nonlinearity=F.elu).to(device)

    loss_fcn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # train and test
    if args.mode == "train":
        train(model, loss_fcn, device, optimizer, train_dataloader, test_dataset)
        torch.save(model.state_dict(), MODEL_STATE_FILE)
    model.load_state_dict(torch.load(MODEL_STATE_FILE))
    return test(model, loss_fcn, device, test_dataloader)


def train(model, loss_fcn, device, optimizer, train_dataloader, test_dataset):
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch, data in enumerate(train_dataloader):
            subgraph, features, labels = data
            features = features.to(device)
            labels = labels.to(device)

            model.set_g(subgraph)
            #model.g = subgraph

            logits = model(features.float())
            loss = loss_fcn(logits, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        loss_data = np.array(losses).mean()
        print("Epoch {:05d} | Loss: {:.4f}".format(epoch + 1, loss_data))

        if epoch % 5 == 0 or epoch in [141, 143, 147, 152]:
            scores = []
            for batch, test_data in enumerate(test_dataset):
                subgraph, features, labels = test_data
                features = torch.tensor(features).to(device)
                labels = torch.tensor(labels).to(device)
                score, _ = evaluate(features.float(), model, subgraph, labels.float(), loss_fcn)
                scores.append(score)
            print("F1-Score: {:.4f} ".format(np.array(scores).mean()))


def test(model, loss_fcn, device, test_dataloader):
    test_scores = []
    for batch, test_data in enumerate(test_dataloader):
        subgraph, features, labels = test_data
        features = features.to(device)
        labels = labels.to(device)
        test_scores.append(evaluate(features, model, subgraph, labels.float(), loss_fcn)[0])
    mean_scores = np.array(test_scores).mean()
    print("F1-Score: {:.4f}".format(np.array(test_scores).mean()))
    return mean_scores


def evaluate(features, model, subgraph, labels, loss_fcn):
    with torch.no_grad():
        model.eval()
        model.set_g(subgraph)
        output = model(features.float())
        loss_data = loss_fcn(output, labels.float())
        predict = np.where(output.data.cpu().numpy() >= 0.5, 1, 0)
        score = f1_score(labels.data.cpu().numpy(), predict, average="micro")
        return score, loss_data.item()


def collate_fn(sample):
    graphs, features, labels = map(list, zip(*sample))
    graph = batch(graphs)
    features = torch.from_numpy(np.concatenate(features))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, features, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  choices=["train", "test"], default="train")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--model",  choices=["base", "gat"], default="gat")
    parser.add_argument("--hidden_dim", type=int, default=200)
    parser.add_argument("--num_heads", type=int, default=10)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--feat_drop", type=float, default=0.0)
    parser.add_argument("--att_drop", type=float, default=0.0)

    args = parser.parse_args()
    main(args)
