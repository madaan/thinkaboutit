
import torch
import torch.nn as nn
from src.gnn_qa.model.moe.data import InfluenceGraphNNData
from src.gnn_qa.model.lib.moe import GatingNetwork, ExpertModel

class GraphExpert(nn.Module):
    def __init__(self, num_layers: int, input_size: int, hidden_size: int, output_size: int,
    k: int = 2):
        super().__init__()
        
        self.experts = nn.ModuleList([ExpertModel(num_layers, input_size, hidden_size,
                                            output_size)
                                for i in range(InfluenceGraphNNData.num_nodes_per_graph)])
        
        self.gating = GatingNetwork(hidden_size * InfluenceGraphNNData.num_nodes_per_graph,
         InfluenceGraphNNData.num_nodes_per_graph)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.k = k

    def forward(self, graphs):
        #  graphs: B x InfluenceGraphNNData.num_nodes_per_graph x 768
    
        bsz, num_nodes, hsz = graphs.shape

        outputs = [self.experts[i](graphs[:, i]) for i in range(num_nodes)]
        outputs = torch.cat(outputs, dim=-1).view(bsz, hsz, num_nodes)

        # only keep the results from top k experts.
        experts_logits = self.gating(graphs.contiguous().view(bsz, -1))
        # zeroes = torch.zeros_like(experts_logits)
        # ones = torch.ones_like(experts_logits)
        # zeroes.scatter_(1, index=experts_logits.topk(self.k, dim=-1).indices, src=ones)
        # experts_logits = torch.mul(experts_logits, zeroes)

        expert_probs = self.softmax(experts_logits).unsqueeze(1)  # B x 1 x num_nodes
        combined_output = expert_probs.mul(outputs)
        
        return combined_output.sum(dim=-1).squeeze(1), expert_probs  # B x hsz