import torch.nn as nn
import torch
import math

class CustomRobertaClassificationHeadConcat(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class CustomRobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class MultiheadedAttention(nn.Module):
  """Multiheaded attention. The is NOT self attention, the query belongs to one type,
  and the keys/values belong to another.
  """
  def __init__(self, h_dim: int, kqv_dim: int, n_heads: int):
    super(MultiheadedAttention, self).__init__()

    self.h_dim = h_dim
    self.kqv_dim = kqv_dim
    self.n_heads = n_heads
    self.w_k = nn.Linear(h_dim, kqv_dim * n_heads, bias=False)
    self.w_v = nn.Linear(h_dim, kqv_dim * n_heads, bias=False)
    self.w_q = nn.Linear(h_dim, kqv_dim * n_heads, bias=False)
    self.w_out = nn.Linear(kqv_dim * n_heads, h_dim)
    
  
  def forward(self, KV: torch.Tensor, Q: torch.Tensor):
    """Multiheaded self-attention

    Args:
        KV (torch.Tensor): [B x seq_len x h_dim]
        Q (torch.Tensor): [B x h_dim]
    """

    bsz, seq_len, h_dim = KV.shape
    # step 1: Apply the K, V, Q transforms
    K = self.w_k(KV).view(bsz, seq_len, self.n_heads, self.kqv_dim)
    V = self.w_v(KV).view(bsz, seq_len, self.n_heads, self.kqv_dim)
    Q = self.w_q(Q).view(bsz, self.n_heads, self.kqv_dim)

    # step 2: Get the attention scores
    attn = self.get_attention(Q=Q, K=K) # [(B x n_heads) x 1 x seq_len]
    
    # step 3: Get the values
    V = V.transpose(1, 2)
    V = V.reshape(bsz * self.n_heads, seq_len, self.kqv_dim)  # [(B x n_heads) x seq_len x kqv_input_dim]
    values = torch.bmm(attn, V)  # [(B x n_heads) x 1 x kqv_input_dim]
    values = values.reshape(bsz, self.n_heads * self.kqv_dim)
    output = self.w_out(values)

    attn = attn.view(bsz, self.n_heads, 1, seq_len).sum(dim=1) / self.n_heads
    return attn, output


  def get_attention(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Returns the attention scores.

    Args:
        Q (torch.Tensor): [B x n_heads x kqv_input_dim]
        K (torch.Tensor): [bsz, seq_len, self.n_heads, self.kqv_input_dim]
    Returns:
      Attention scores of dim [B x 1 x seq_len]
    """
    bsz, seq_len, _, _ = K.shape

    K = K.transpose(1, 2)  # [(B x n_heads x seq_len x kqv_input_dim)]
    K = K.reshape(bsz * self.n_heads, seq_len, self.kqv_dim)  # [(B x n_heads) x seq_len x kqv_input_dim)]

    Q = Q.unsqueeze(2)  # B x n_heads x 1 x kqv_input_dim
    Q = Q.reshape(bsz * self.n_heads, 1, self.kqv_dim) # [(B x n_heads) x 1 x kqv_input_dim]
    attn = torch.bmm(Q, K.transpose(1, 2))  # [(B x n_heads) x 1 x seq_len]
    attn = attn / math.sqrt(self.kqv_dim)   # [(B x n_heads) x 1 x seq_len]
    attn = torch.softmax(attn, dim=-1)  # [(B x n_heads) x 1 x seq_len]
    return attn


#  from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      """Adds positional encoding to x.
      x should be of the size: B x seq_len x d_model

      Args:
          x (torch.Tensor): [The input tensor.]

      Returns:
          torch.Tensor: [The output tensor with the positional embeddings added.]
      """
      x = x + self.pe[:x.size(0), :]
      return self.dropout(x)