import argparse
from os import path

import numpy as np
import torch
import torch.nn.functional as F
from dgl import batch
from dgl.data.ppi import LegacyPPIDataset
from dgl.nn.pytorch import GraphConv
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.utils.data import DataLoader
from gat import GAT
from gat2 import GAT2

MODEL_STATE_FILE = path.join(path.dirname(path.abspath(__file__)), "model_state.pth")


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

        # model = GAT(g=train_dataset.graph,
        #             in_dim=n_features,
        #             hidden_dim=args.hidden_dim,
        #             out_dim=n_classes,
        #             num_heads=args.num_heads,
        #             num_layers=args.num_layers).to(device)

        model = GAT2(g=train_dataset.graph,
                     num_layers=args.num_layers,
                     in_dim=n_features,
                     num_hidden=args.hidden_dim,
                     num_classes=n_classes,
                     heads=[args.num_heads] * args.num_layers,
                     activation=None,
                     feat_drop=args.feat_drop,
                     attn_drop=args.att_drop,
                     negative_slope=0.2,
                     residual=False
                     ).to(device)
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


# def train(model, loss_fcn, device, optimizer, train_dataloader, test_dataset):
#     for epoch in range(args.epochs):
#         model.train()
#         losses = []
#         for batch, data in enumerate(train_dataloader):
#             subgraph, features, labels = data
#             features = features.to(device)
#             labels = labels.to(device)
#             model.set_g(subgraph)
#             logits = model(features.float())
#             loss = loss_fcn(logits, labels.float())
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             losses.append(loss.item())
#         loss_data = np.array(losses).mean()
#         print("Epoch {:05d} | Loss: {:.4f}".format(epoch + 1, loss_data))
#
#         if epoch % 5 == 0:
#             scores = []
#             for batch, test_data in enumerate(test_dataset):
#                 subgraph, features, labels = test_data
#                 features = torch.tensor(features).to(device)
#                 labels = torch.tensor(labels).to(device)
#                 score, _ = evaluate(features.float(), model, subgraph, labels.float(), loss_fcn)
#                 scores.append(score)
#             print("F1-Score: {:.4f} ".format(np.array(scores).mean()))


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
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--model",  choices=["base", "gat"], default="gat")
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--feat_drop", type=float, default=0.0)
    parser.add_argument("--att_drop", type=float, default=0.0)



    args = parser.parse_args()
    main(args)
