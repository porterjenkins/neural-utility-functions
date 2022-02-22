# source: https://github.com/rixwew/pytorch-fm/blob/master/torchfm/model/dfm.py

import torch

from baselines.fm_layers import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron
from baselines.fm_layers import FeaturesSparseEmbedding, FeaturesSparseLinear


class DeepFM(torch.nn.Module):
    """
    A pytorch implementation of DeepFM.
    Reference:
        H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, use_logit=False):
        super().__init__()
        self.use_logit = use_logit
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=False)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, users, items):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """

        embed_x = self.embedding(items)
        x_linear = self.linear(items).unsqueeze(dim=-1)
        x_fm = self.fm(embed_x).unsqueeze(dim=-1)
        x_mlp = self.mlp(embed_x.view(-1, self.embed_output_dim))

        if items.ndim == 3:
            x_mlp = x_mlp.view(items.shape[0], items.shape[1], -1)

        output = x_linear + x_fm + x_mlp

        if self.use_logit:
            output = torch.sigmoid(output)

        return output