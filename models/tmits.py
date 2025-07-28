from math import sqrt

import torch
from torch import nn

from models.attention import Attention
from models.UD import UpDimensional
from models.transformer import TransformerEncoderWrapper


class T_MITS(nn.Module):
    def __init__(
        self,
        n_var_embs: int,
        dim_demog: int = 2,
        dim_embed: int = 104,
        n_layers: int = 2,
        n_heads: int = 8,
        dim_ff: int = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        n_quantiles: int = 3,
    ):
        """Transformer for Multivariate Irregular Time Series

        Parameters
        ----------
        n_var_embs : int
            How many different variables will be in the input (for the variable embedding module)
        dim_demog : int, optional
            dimension of the demographic input, by default 2
        dim_embed : int, optional
            size of the val/time/var embeddings, by default 104
        n_layers : int, optional
            Number of Transformer encoder layers, by default 2
        n_heads : int, optional
            Number of attention heads, by default 8
        dim_ff : int, optional
            Dimension of the feedforward layers in the Transformer (if None, is set to `2*dim_embed`), by default None
        dropout : float, optional
            Probability to drop values, by default 0.0
        activation : str, optional
            Activation function of the Transformer layers, by default "gelu"
        n_quantiles: int, optional
            How many different quantiles to predict, by default 3
        """
        super().__init__()
        n_var_embs += 1  # Add 1 for the time token

        # Store info for easy access
        self.n_var_embs = n_var_embs
        self.dim_demog = dim_demog
        self.dim_embed = dim_embed
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.n_quantiles = n_quantiles

        if self.dim_demog is not None:
            self.lin_1_demog = nn.Linear(
                in_features=dim_demog, out_features=3 * dim_embed
            )
            self.lin_2_demog = nn.Linear(
                in_features=3 * dim_embed, out_features=3 * dim_embed
            )
            self.lin_3_demog = nn.Linear(
                in_features=3 * dim_embed, out_features=dim_embed
            )
            self.tanh_demog = nn.Tanh()

        self.var_embedding = nn.Embedding(
            num_embeddings=n_var_embs, embedding_dim=dim_embed
        )
        self.ud_values = UpDimensional(
            hid_units=int(sqrt(dim_embed)), output_dim=dim_embed
        )
        self.ud_times = UpDimensional(
            hid_units=int(sqrt(dim_embed)), output_dim=dim_embed
        )

        self.dense_agg = nn.Linear(
            in_features=3 * dim_embed, out_features=3 * dim_embed
        )

        self.trans = TransformerEncoderWrapper(
            3 * dim_embed,
            n_layers,
            n_heads,
            dim_ff,
            dropout,
            activation,
        )
        self.attn = Attention(3 * dim_embed, hid_dim=6 * dim_embed)

        if self.dim_demog is not None:
            self.quantile_outputs = nn.ModuleList(
                [
                    nn.Linear(in_features=4 * dim_embed, out_features=1)
                    for _ in range(n_quantiles)
                ]
            )
        else:
            self.quantile_outputs = nn.ModuleList(
                [
                    nn.Linear(in_features=3 * dim_embed, out_features=1)
                    for _ in range(n_quantiles)
                ]
            )

    def forward(
        self,
        demog: torch.Tensor,
        values: torch.Tensor,
        times: torch.Tensor,
        variables: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        """
        demog : [batch_size, dim_demog]
        values, times, variables, mask : [batch_size, seq_len]

        Ouput: [batch_size, n_quantiles]

        """
        # Embed the static vector, if it exists
        if self.dim_demog is not None:
            demo_emb = self.lin_3_demog(
                self.tanh_demog(
                    self.lin_2_demog(self.tanh_demog(self.lin_1_demog(demog)))
                )
            )

        # Embed each element of the (val, time, var) triplet
        var_emb = self.var_embedding(variables.long())
        values_emb = self.ud_values(values)
        times_emb = self.ud_times(times)

        # stack the embeddings
        combined_emb = torch.concat((var_emb, values_emb, times_emb), dim=2)
        # turn the stack of embeddings into a single dim_embed vec
        aggregated_emb = self.dense_agg(combined_emb)

        contextual_emb = self.trans(aggregated_emb, src_key_padding_mask=mask)

        # to reduce the embedding dimension, do a weighted sum with the attn weights
        attn_weights = self.attn(contextual_emb, mask=mask)
        fused_emb = torch.sum(contextual_emb * attn_weights, dim=-2)

        # add demog
        if self.dim_demog is not None:
            concat = torch.concat((fused_emb, demo_emb), dim=1)
        else:
            # Conserve dimensionality even when no demog
            concat = fused_emb

        # return quantile predictions
        return torch.cat([qo(concat) for qo in self.quantile_outputs], dim=1).squeeze()
