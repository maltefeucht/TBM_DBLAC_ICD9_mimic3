import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math


class LabelAttention(nn.Module):
    '''
    Label attention classifier: Uses pretrained word2vec label embeddings for every ICD_9-code, applies them to the input embedding representation of an encoder architecture
    and computes a vector-label representation for every ICD-9 code using the self-attention concept (i.e. every word ofinput seauence length N is assigned an attention score (0-1) for every preditec ICD-9 code)
    Parameters
    ----------
    word embedding matrix E: tensor (B, N, d_e)
    label embedding matrix V: tensor (B, M, d_a)
    Returns
    -------
    label-attention matrix C: tensor ( B x M x d_e)
    """
    '''

    def __init__(self, hidden_size, label_embed_size, dropout_rate):
        super(LabelAttention, self).__init__()
        self.l1 = nn.Linear(hidden_size, label_embed_size)
        self.tnh = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate)
        self.label_embed_size_sqrt = math.sqrt(label_embed_size)

    def forward(self, hidden, label_embeds, attn_mask=None):
        # output_1: B x N x d_e -> B x N x d_a
        output_1 = self.l1(hidden)
        output_1 = self.dropout(output_1)
        # output_2: (B x N x d_a) x (d_a x M) -> B x N x M
        assert output_1.size(2) == label_embeds.weight.t().size(0), "label_embed_size must match the dimension of the embedded labels"
        output_2 = th.matmul(output_1, label_embeds.weight.t())
        # output_2: keep dim B x N x M but divide output_2 by sqrt. label_embed_size
        output_3 = th.div(output_2, self.label_embed_size_sqrt)
        output_3 = output_3.type_as(hidden)

        # Masked fill to avoid softmaxing over padded words
        if attn_mask is not None:
            output_3 = output_3.masked_fill(attn_mask == 0, -1e9)

        # attn_weights: B x N x M -> B x M x N
        attn_weights = F.softmax(output_3, dim=1).transpose(1, 2)

        # weighted_output: (B x M x N) @ (B x N x d_e) -> B x M x d_e
        weighted_output = attn_weights @ hidden
        return weighted_output, attn_weights


