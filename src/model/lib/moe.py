"""MOE
Taken from https://raw.githubusercontent.com/madaan/CLTL/master/src/model/moe.py
Author:
    Tanmay Parekh <tparekh@cs.cmu.edu>
Εdited by:
    Aman Madaan <amadaan@cs.cmu.edu>


Returns:
    [type] -- [description]
"""
import torch
from torch import nn


class ExpertModel(nn.Module):
    """Expert Model specifically designated for a particular task/subtask
    Fits into a mixture of experts

    Input: Extracted Features (B x S x H)
    Output: Out Features (B x S x O)

    B: Batch Size
    S: Sequence Length
    H: Hidden Size (Extracted Features)
    O: Output Size
    """

    def __init__(self, num_layers, input_size, hidden_size, output_size, activation='relu',
                 final_activation='relu', dropout=0.0, layer_norm=False):
        '''
        Basic Paper Implementation:
            Multi-layer perceptron. Layers can have varied sizes and dropouts

            Input:
                num_layers:             Number of Layers [int >= 0]
                input_size:             Size of Input Features [int > 0]
                hidden_size:            Sizes of individual layer(s) [int > 0] (all layers of same size) OR
                                        list[int > 0] (for varied sizes) with len(list) == num_layers
                output_size:            Size of Output [int > 0]
                activation:             Activation to be used after each layer(s) [categorical = {'tanh', 'relu', 'softmax', 'logsoftmax}]
                final_activation:       Activation to be used at output layer [categorical = {'tanh', 'relu', 'softmax', 'logsoftmax}]
                                        (Vary this for tagging and classification tasks)
                dropout:                Dropout for layer(s) [int > 0] (all layers of same size) OR
                                        list[int > 0] (for varied sizes) with len(list) == num_layers + 1
                layer_norm:             Layer Normalization after each layer [bool]
        '''
        super(ExpertModel, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.final_activation = final_activation
        self.dropout = dropout
        self.use_layer_norm = layer_norm

        # Input Format Checks
        assert isinstance(
            self.num_layers, int) and self.num_layers >= 0, "Number of Layers should be a non-negative integer; Got {}".format(self.num_layers)
        assert isinstance(
            self.input_size, int) and self.input_size > 0, "Input size should be a positive integer; Got {}".format(self.input_size)
        assert isinstance(self.output_size, int) and self.output_size > 0, "Output size should be a positive integer; Got {}".format(
            self.output_size)
        assert isinstance(self.activation, str) and self.activation in [
            'tanh', 'relu', 'softmax', 'logsoftmax'], "Activation should be string and amongst ['tanh', 'relu', 'softmax', 'logsoftmax']; Got {}".format(activation)
        assert isinstance(self.final_activation, str) and self.final_activation in [
            'tanh', 'relu', 'softmax', 'logsoftmax'], "Final activation should be string and amongst ['tanh', 'relu', 'softmax']; Got {}".format(final_activation)
        assert isinstance(self.use_layer_norm, bool), "Layer Norm should be a boolean; Got {}".format(
            self.use_layer_norm)

        if isinstance(self.hidden_size, int):
            assert self.hidden_size > 0, "Hidden size should be a positive integer; Got {}".format(
                self.hidden_size)
        elif isinstance(self.hidden_size, list):
            assert len(self.hidden_size) == num_layers, "Number of layers don't match length of hidden sizes; Num Layers: {} and Length of Hidden Sizes: {}".format(
                self.num_layers, len(self.hidden_size))
            assert all(isinstance(h, int) and h > 0 for h in self.hidden_size), "All sizes in hidden size vector are not positive integers; Got {}".format(
                " ".join([str(h) for h in self.hidden_size]))
        else:
            assert False, "Got non-integer or list type for hidden size"

        if isinstance(self.dropout, float):
            assert self.dropout >= 0, "Dropout should be positive; Got {}".format(
                self.dropout)
        elif isinstance(self.dropout, list):
            assert len(self.dropout) == num_layers + \
                1, "Number of layers + 1 don't match length of dropout; Num Layers: {} and Length of Dropout: {}".format(
                    self.num_layers, len(self.dropout))
            assert all(isinstance(d, float) and d >= 0 for d in self.dropout), "All sizes in dropout vector are not positive; Got {}".format(
                " ".join([str(d) for d in self.dropout]))
        else:
            assert False, "Got non-float or list type for dropout"

        # Model

        # Basic Paper Implementation

        # Activation
        if self.activation == 'relu':
            activation_layer = nn.ReLU()
        elif self.activation == 'tanh':
            activation_layer = nn.Tanh()
        elif self.activation == 'softmax':
            activation_layer = nn.Softmax(dim=-1)
        elif self.activation == 'logsoftmax':
            activation_layer = nn.LogSoftmax(dim=-1)
        else:
            raise Exception(f"Unknown activation {self.activation}")

        # Final Activation
        if self.final_activation == 'relu':
            final_activation_layer = nn.ReLU()
        elif self.final_activation == 'tanh':
            final_activation_layer = nn.Tanh()
        elif self.final_activation == 'softmax':
            final_activation_layer = nn.Softmax(dim=-1)
        elif self.final_activation == 'logsoftmax':
            final_activation_layer = nn.LogSoftmax(dim=-1)
        else:
            raise Exception(
                f"Unknown final activation {self.final_activation}")

        self.model = nn.Sequential()

        # Initial Dropout
        if isinstance(self.dropout, float):
            self.model.add_module("expert_dropout_{}".format(
                0), nn.Dropout(p=self.dropout))
        elif isinstance(self.dropout, list):
            self.model.add_module("expert_dropout_{}".format(
                0), nn.Dropout(p=self.dropout[0]))

        for i in range(self.num_layers):

            if isinstance(self.hidden_size, int):

                # Linear
                if i == 0:
                    self.model.add_module("expert_linear_{}".format(
                        i), nn.Linear(self.input_size, self.hidden_size))
                else:
                    self.model.add_module("expert_linear_{}".format(
                        i), nn.Linear(self.hidden_size, self.hidden_size))

                # Layer Norm
                if self.use_layer_norm:
                    self.model.add_module("expert_layerNorm_linear_{}".format(
                        i), nn.LayerNorm(self.hidden_size))

            else:

                # Linear
                if i == 0:
                    self.model.add_module("expert_linear_{}".format(
                        i), nn.Linear(self.input_size, self.hidden_size[i]))
                else:
                    self.model.add_module("expert_linear_{}".format(
                        i), nn.Linear(self.hidden_size[i-1], self.hidden_size[i]))

                # Layer Norm
                if self.use_layer_norm:
                    self.model.add_module("expert_layerNorm_linear_{}".format(
                        i), nn.LayerNorm(self.hidden_size[i]))

            # Activation
            self.model.add_module(
                "expert_activation_{}".format(i), activation_layer)

            # Dropout
            if isinstance(self.dropout, float):
                self.model.add_module("expert_dropout_{}".format(
                    i+1), nn.Dropout(p=self.dropout))
            elif isinstance(self.dropout, list):
                self.model.add_module("expert_dropout_{}".format(
                    i+1), nn.Dropout(p=self.dropout[i]))

        # Final Linear
        if self.num_layers == 0:
            self.model.add_module("expert_final_linear", nn.Linear(
                self.input_size, self.output_size))
        else:
            last_hidden_size = self.hidden_size if isinstance(
                self.hidden_size, int) else self.hidden_size[-1]
            self.model.add_module("expert_final_linear", nn.Linear(
                last_hidden_size, self.output_size))

        # Final Layer Norm
        if self.use_layer_norm:
            self.model.add_module(
                "expert_layerNorm_final_linear", nn.LayerNorm(self.output_size))

        # Final Activation
        self.model.add_module("expert_final_activation",
                              final_activation_layer)

    def forward(self, inputs):
        '''
        Transform the input to output via MLP

        Input: Extracted Features (B x S x H)
        Output: Out Features (B x S x O)
        '''

        return self.model(inputs)


class MixtureOfExperts(nn.Module):
    '''
    Combination of individual expert models with a gating mechanism to
    combine the experts in a weighted manner

    Input: Extracted Features (B x S x H) OR (B x H)
    Output: Out Features (B x S x O) OR (B x O), Gate Outputs (B x S x E) OR (B x E)

    B: Batch Size
    S: Sequence Length
    H: Hidden Size (Extracted Features)
    O: Output Size
    E: The number of experts
    '''

    def __init__(self, num_layers, input_size, hidden_size, output_size, num_experts,
                 activation='relu', final_activation='relu', dropout=0.0, layer_norm=False,
                 gating_input_detach=True):
        '''
        Basic Paper Implementation:
            Multi-layer perceptron. Layers can have varied sizes and dropouts

            Input:
                num_layers:             Number of Layers [int >= 0]
                input_size:             Size of Input Features [int > 0]
                hidden_size:            Sizes of individual layer(s) [int > 0] (all layers of same size) OR
                                        list[int > 0] (for varied sizes) with len(list) == num_layers
                output_size:            Size of Output [int > 0]
                num_experts:            Number of expert models to use [int > 0]
                activation:             Activation to be used after each layer(s) [categorical = {'tanh', 'relu', 'softmax', 'logsoftmax}]
                final_activation:       Activation to be used at output layer [categorical = {'tanh', 'relu', 'softmax', 'logsoftmax}]
                                        (Vary this for tagging and classification tasks)
                dropout:                Dropout for layer(s) [int > 0] (all layers of same size) OR
                                        list[int > 0] (for varied sizes) with len(list) == num_layers + 1
                layer_norm:             Layer Normalization after each layer [bool]
                gating_input_detach:    Whether to allow gradients to pass back from the gating network [bool]
        '''
        super(MixtureOfExperts, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.activation = activation
        self.final_activation = final_activation
        self.dropout = dropout
        self.use_layer_norm = layer_norm
        self.gating_input_detach = gating_input_detach

        # Input Format Checking Done in individual models

        # Model

        # Basic Paper Implementation

        self.experts = nn.ModuleList([ExpertModel(self.num_layers, self.input_size, self.hidden_size,
                                                  self.output_size, self.activation, self.final_activation,
                                                  self.dropout, self.use_layer_norm)
                                      for i in range(self.num_experts)])
        self.gating = GatingNetwork(self.input_size, self.num_experts)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, inputs):
        '''
        Transform input to a convex combination of outputs of individual expert models
        combined in a weighted manner by the gating network

        Input: Extracted Features (B x S x H) OR (B x H)
        Output: Out Features (B x S x O) OR (B x O), Gate Outputs (B x S x E) OR (B x E)
        '''

        expert_outputs = torch.cat([e(inputs).unsqueeze(
            dim=-1) for e in self.experts], dim=-1)       # B x S x O x E
        if self.gating_input_detach:
            # B x S x E
            logits = self.gating(inputs.detach())
        else:
            # B x S x E
            logits = self.gating(inputs)

        # convert logits to probs        
        probs = self.softmax(logits)

        if len(list(inputs.size())) == 3:
            combined_output = probs.unsqueeze(dim=2).mul(
                expert_outputs)                                    # B x S x O x E
        elif len(list(inputs.size())) == 2:
            combined_output = probs.unsqueeze(dim=1).mul(
                expert_outputs)                                    # B x O x E
            
        combined_output = combined_output.sum(dim=-1)               # B x S x O

        return combined_output, probs


class GatingNetwork(nn.Module):
    '''
    Model to decide weights to be given to individual expert model

    Input: Extracted Features (B x S x H)
    Output: Model Weights (B x S x E)

    B: Batch Size
    S: Sequence Length
    H: Hidden Size (Extracted Features)
    E: Number of experts
    '''

    def __init__(self, input_size, num_experts):
        '''
        Basic Paper Implementation:
            Simple Linear Transformation with softmax

        Inputs:
            input_size:             Size of Input Features [int > 0]
            num_experts:            Number of experts [int > 0]
        '''
        super(GatingNetwork, self).__init__()

        self.input_size = input_size
        self.num_experts = num_experts

        # Input Format Checks
        assert isinstance(
            self.input_size, int) and self.input_size > 0, "Input size should be a positive integer; Got {}".format(self.input_size)
        assert isinstance(self.num_experts, int) and self.num_experts > 0, "Number of experts should be a positive integer; Got {}".format(
            self.num_experts)

        # Model

        # Basic Paper Implementation

        self.model = nn.Sequential()

        # Linear
        self.model.add_module("gating_linear", nn.Linear(
            self.input_size, self.num_experts))

    def forward(self, inputs):
        '''
        Based on input, assign probability score to each expert

        Input: Extracted Features (B x S x H)
        Output: Model Weights (B x S x E)
        '''

        return self.model(inputs)


if __name__ == "__main__":

    ########################
    # Testing Expert Model
    ########################

    # Test 1 - Testing zero-layer MLP
    m1 = ExpertModel(0, 8, 6, 4)
    i1 = torch.rand(2, 10, 8)
    assert list(m1(i1).size()) == [2, 10, 4], "Shape Testing Failed"

    # Test 2 - Testing vectored hidden size and dropout
    m2 = ExpertModel(2, 4, [6, 8], 4, dropout=[0.2, 0.4, 0.2])
    i2 = torch.rand(4, 2, 4)
    assert list(m2(i2).size()) == [
        4, 2, 4], "Hidden Vector and Dropout Vector Testing Failed"

    # Test 3 - Testing Layer Norm
    m3 = ExpertModel(1, 4, 6, 2, layer_norm=True)
    i3 = torch.rand(4, 2, 4)
    assert list(m3(i3).size()) == [4, 2, 2], "Layer Norm Testing Failed"

    # Test 4 - Testing activations
    m4 = ExpertModel(0, 4, 6, 2, activation='relu')
    i4 = torch.rand(4, 2, 4)
    assert list(m4(i4).size()) == [4, 2, 2], "Activation Testing Failed"

    m4 = ExpertModel(0, 4, 6, 2, activation='tanh')
    i4 = torch.rand(4, 2, 4)
    assert list(m4(i4).size()) == [4, 2, 2], "Activation Testing Failed"

    m4 = ExpertModel(0, 4, 6, 2, activation='softmax')
    i4 = torch.rand(4, 2, 4)
    assert list(m4(i4).size()) == [4, 2, 2], "Activation Testing Failed"

    m4 = ExpertModel(0, 4, 6, 2, activation='logsoftmax')
    i4 = torch.rand(4, 2, 4)
    assert list(m4(i4).size()) == [4, 2, 2], "Activation Testing Failed"

    # Test 5 - Testing final activation
    m5 = ExpertModel(0, 4, 6, 2, final_activation='relu')
    i5 = torch.rand(4, 2, 4)
    assert list(m5(i5).size()) == [4, 2, 2], "Final activation Testing Failed"

    m5 = ExpertModel(0, 4, 6, 2, final_activation='tanh')
    i5 = torch.rand(4, 2, 4)
    assert list(m5(i5).size()) == [4, 2, 2], "Final activation Testing Failed"

    m5 = ExpertModel(0, 4, 6, 2, final_activation='softmax')
    i5 = torch.rand(4, 2, 4)
    assert list(m5(i5).size()) == [4, 2, 2], "Final activation Testing Failed"

    m5 = ExpertModel(0, 4, 6, 2, final_activation='logsoftmax')
    i5 = torch.rand(4, 2, 4)
    assert list(m5(i5).size()) == [4, 2, 2], "Final activation Testing Failed"

    ########################
    # Testing Gating Network
    ########################

    # Test 1 - Testing expert shapes and probability sums
    g1 = GatingNetwork(8, 3)
    i1 = torch.rand(4, 12, 8)
    assert list(g1(i1).size()) == [4, 12, 3], "Expert Gate Shapes Test Failed"
    assert torch.allclose(g1(i1).sum(dim=-1), torch.ones(4, 12)
                          ), "Probabilities of expert gates don't add to 1"

    ########################
    # Testing Mixture of Experts
    ########################

    # Test 1 - Testing shape of outputs and gate probs
    num_layers, input_size, hidden_size, output_size, num_experts = 2, 3, 4, 8, 5
    batch_sz, seq_len, input_hsz = 6, 10, 3
    moe1 = MixtureOfExperts(num_layers, input_size,
                            hidden_size, output_size, num_experts)
    i1 = torch.rand(batch_sz, seq_len, input_hsz)
    o1, g1 = moe1(i1)
    assert list(o1.size()) == [batch_sz, seq_len,
                               output_size], "MoE Output Shape Test Failed"
    assert list(g1.size()) == [batch_sz, seq_len,
                               num_experts], "MoE Gates Shape Test Failed"
    assert torch.allclose(g1.sum(dim=-1), torch.ones(batch_sz, seq_len)
                          ), "Probabilities of expert gates don't add to 1"

    # Test 2 - Testing shape of outputs and gate probs for 2-dim inputs
    num_layers, input_size, hidden_size, output_size, num_experts = 2, 3, 4, 8, 5
    batch_sz, input_hsz = 6, 3
    moe1 = MixtureOfExperts(num_layers, input_size,
                            hidden_size, output_size, num_experts)
    i1 = torch.rand(batch_sz, input_hsz)
    o1, g1 = moe1(i1)
    assert list(o1.size()) == [batch_sz,
                               output_size], "MoE Output Shape Test Failed"
    assert list(g1.size()) == [batch_sz,
                               num_experts], "MoE Gates Shape Test Failed"
    assert torch.allclose(g1.sum(dim=-1), torch.ones(batch_sz)
                          ), "Probabilities of expert gates don't add to 1"