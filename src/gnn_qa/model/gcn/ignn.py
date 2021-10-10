"""Wraps the influence graph under a Pytorch-geom wrapper
"""
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, GCNConv
from pytorch_lightning.core.lightning import LightningModule




class InfluenceGraphGNN(LightningModule):
    def __init__(self, num_in_features: int, num_out_features: int, num_relations: int, rgcn: bool = False):
        super(InfluenceGraphGNN, self).__init__()
        self.rgcn = rgcn
        if rgcn:
            self.conv1 = RGCNConv(num_in_features, num_out_features, num_relations=num_relations)
            self.conv2 = RGCNConv(num_out_features, num_out_features, num_relations=num_relations)
            self.conv3 = RGCNConv(num_out_features, num_out_features, num_relations=num_relations)
        else:
            self.conv1 = GCNConv(num_in_features, num_out_features)
            self.conv2 = GCNConv(num_out_features, num_out_features)

    def forward(self, data):
        if self.rgcn:
            return self.forward_rgcn(data)
        else:
            return self.forward_gcn(data)

    def forward_rgcn(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_index = edge_index.to(self.device)
        x = self.conv1(x, edge_index=edge_index, edge_type=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index=edge_index, edge_type=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index=edge_index, edge_type=edge_attr)
        return x

    def forward_gcn(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_index = edge_index.to(self.device)
        x = self.conv1(x, edge_index=edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index=edge_index)
        return x

if __name__ == "__main__":
    graph_str = "Food is taken in the form of food  - hurts - X : The man eats less food than he needs | Z : A person is at a party and eats more  - helps - X : The man eats less food than he needs | X : The man eats less food than he needs  - hurts - W : The man will feel hungry | X : The man eats less food than he needs  - helps - Y : The man won't eat enough | U : more food is taken in  - hurts - Y : The man won't eat enough | W : The man will feel hungry  - hurts - L : LESS food being digested | W : The man will feel hungry  - helps - M : MORE food being digested? | Y : The man won't eat enough  - hurts - M : MORE food being digested? | Y : The man won't eat enough  - helps - L : LESS food being digested."