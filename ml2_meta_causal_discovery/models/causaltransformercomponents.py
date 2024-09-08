import torch
import torch.nn as nn
from typing import Callable, Optional
from torch import Tensor
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import math
from torch.nn.init import xavier_uniform_
import copy


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, 1, max_len, d_model)
        pe[0, 0, :, 0::2] = torch.sin(position * div_term)
        pe[0, 0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, sample_size, seq_len, 1]``
        """
        pos = self.pe[:, :, :x.size(2), :]
        # tile
        # shape [batch_size, sample_size, seq_len, d_model]
        pos = pos.repeat(x.size(0), x.size(1), 1, 1)
        return self.dropout(pos)


def build_mlp(dim_in, dim_hid, dim_out, depth):
    if dim_in == dim_hid:
        modules = [
            nn.Sequential(
                nn.Linear(dim_in, dim_hid),
                nn.ReLU(),
            )
        ]
    else:
        modules = [nn.Linear(dim_in, dim_hid), nn.ReLU()]
    for _ in range(int(depth) - 2):
        modules.append(
            nn.Sequential(
                nn.Linear(dim_hid, dim_hid),
                nn.ReLU(),
            )
        )
    modules.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*modules)


class CausalTransformerEncoder(nn.Module):
    """
    Causal Transformer that alternates attention between samples and nodes.
    """

    def __init__(
        self,
        encoder_layers: nn.ModuleList,
        norm=None,
        enable_nested_tensor=True,
        mask_check=True,
    ) -> None:
        super(CausalTransformerEncoder, self).__init__()
        assert len(encoder_layers) > 0, "Encoder must have at least one layer."
        assert len(encoder_layers) % 2 == 0, "Encoder must have an even number of layers."
        self.layers = encoder_layers

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        # src: [batch_size, num_samples, num_nodes, d_model]
        # We need to reshape the tensor to [batch_size * num_nodes, num_samples, d_model]
        # to carry out attention over samples
        batch_size, num_samples, num_nodes, d_model = src.size()
        for idx_layer, mod in enumerate(self.layers):
            if idx_layer % 2 == 0:
                # shape [batch_size, num_nodes, num_samples, d_model]
                src = src.permute(0, 2, 1, 3)
                # shape [batch_size * num_nodes, num_samples, d_model]
                src = src.contiguous().view(batch_size * num_nodes, num_samples, d_model)
                src = mod(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask, is_causal=is_causal)
                # Reshape the tensor back to [batch_size, num_nodes, num_samples, d_model]
                src = src.view(batch_size, num_nodes, num_samples, d_model)
            else:
                # shape [batch_size, num_samples, num_nodes, d_model]
                src = src.permute(0, 2, 1, 3)
                # shape [batch_size * num_samples, num_nodes, d_model]
                src = src.contiguous().view(batch_size * num_samples, num_nodes, d_model)
                src = mod(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask, is_causal=is_causal)
                # Reshape the tensor back to [batch_size, num_samples, num_nodes, d_model]
                src = src.contiguous().view(batch_size, num_samples, num_nodes, d_model)
        return src


class CausalTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """
    Causal Transformer for Decoders. There is no memory in the decoder.
    This will simply perform self-attention and feedforward operations.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable = F.relu,
        layer_norm_eps: float = 0.00001,
        batch_first: bool = True,
        norm_first: bool = True,
        bias: bool = False,
        device=None,
        dtype=None
    ) -> None:
        super(CausalTransformerDecoderLayer, self).__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            device=device,
            bias=bias,
            dtype=dtype,
        )
        self.dim_feedforward = dim_feedforward

    def forward(
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        r"""
        Pass the inputs (and mask) through the decoder layer.

        It takes in memory but does nothing with it. This is to ensure
        compatibility with the nn.TransformerDecoder class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        assert memory is None, "Memory is not used in the decoder."

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm3(x + self._ff_block(x))
        return x


class CausalAdjacencyMatrix(nn.Module):

        def __init__(
            self,
            nhead,
            d_model,
            device,
            dtype,
        ):
            super(CausalAdjacencyMatrix, self).__init__()
            self.num_heads = nhead
            self.d_model = d_model
            self.in_proj_weight = Parameter(
                torch.empty((3 * d_model, d_model), device=device, dtype=dtype)
            )
            self.in_proj_bias = Parameter(
                torch.empty(3 * d_model, device=device, dtype=dtype)
            )
            self.out_proj_weight = Parameter(
                torch.empty(nhead, 1, device=device, dtype=dtype)
            )
            self.out_proj_bias = Parameter(
                torch.empty(1, device=device, dtype=dtype)
            )
            self.reset_parameters()

        def reset_parameters(self):
            xavier_uniform_(self.in_proj_weight)
            xavier_uniform_(self.out_proj_weight)
            self.in_proj_bias.data.zero_()
            self.out_proj_bias.data.zero_()

        def forward(self, representation):
            """
            Performs attention over the representation to compute the adjacency matrix.

            Args:
            -----
                representation: torch.Tensor, shape [batch_size, num_nodes, d_model]

            Returns:
            --------
                pred: torch.Tensor, shape [batch_size, num_nodes, num_nodes]
            """
            query = representation
            key = representation
            # We don't need to compute the value tensor but helps with
            # compatibility with the nn.MultiheadAttention class
            #TODO: Remove the value tensor computation
            value = representation
            # set up shape vars
            bsz, tgt_len, embed_dim = query.shape

            # Tranpose the query, key, and value tensors
            # shape [num_nodes, batch_size, d_model]
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

            #
            # compute in-projection
            #
            q, k, v = F._in_projection_packed(
                query, key, value, self.in_proj_weight, self.in_proj_bias
            )
            del v # we don't need this

            head_dim = self.d_model // self.num_heads

            # reshape q, k, v for multihead attention and make em batch first
            #
            q = q.view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
            k = k.view(k.shape[0], bsz * self.num_heads, head_dim).transpose(0, 1)

            # update source sequence length after adjustments
            src_len = k.size(1)

            #
            # (deep breath) calculate attention and out projection
            #
            q = q.view(bsz, self.num_heads, tgt_len, head_dim)
            k = k.view(bsz, self.num_heads, src_len, head_dim)

            # Efficient implementation equivalent to the following:
            L, S = q.size(-2), k.size(-2)
            scale_factor = 1 / math.sqrt(query.size(-1))
            attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

            attn_weight = q @ k.transpose(-2, -1) * scale_factor
            # shape [batch_size, num_heads, num_nodes, num_nodes]
            attn_weight += attn_bias[None, None, :, :]
            attn_weight = attn_weight.permute(0, 2, 3, 1)
            pred = attn_weight @ self.out_proj_weight + self.out_proj_bias
            pred = pred.squeeze(-1)
            return pred


class CausalTNPEncoder(nn.Module):
    """
    CausalTNPEncoder is a module that encodes target data into a d_model
    dimensional space and computes summary representations.

    Args:
    -----
    - d_model (int): The dimensionality of the model.
    - dim_feedforward (int): The dimensionality of the feedforward network.
    - nhead (int): The number of attention heads.
    - num_layers (int): The number of transformer encoder layers.
    - use_positional_encoding (bool): Whether to use positional encoding.
    - num_nodes (int): The number of nodes.
    - device (str): The device to run the module on.
    - dtype (torch.dtype): The data type of the module's parameters.
    - emb_depth (int, optional): The depth of the embedding MLP. Defaults to 2.
    - dropout (float, optional): The dropout rate. Defaults to 0.0.
    - avici_summary (bool, optional): Whether to use the Avici summary. Defaults to False.
        This will use max pool over the samples to compute the summary representation.

    Methods:
    --------
    - embed(target_data): Embeds the target data into a d_model dimensional space.
    - compute_summary(query, key, value): Computes the summary representation for the query.
    - encode(target_data): Encodes the target data and computes the summary representation.

    Attributes:
    ----------
    - embedder (nn.Module): The MLP used for embedding.
    - encoder (CausalTransformerEncoder): The CausalTransformerEncoder module.
    - representation (nn.MultiheadAttention): The multi-head attention module.
    - use_positional_encoding (bool): Whether to use positional encoding.
    - positional_encoding (PositionalEncoding): The positional encoding module.
        ...
        - target_data (torch.Tensor): The target data with shape [batch_size, num_samples, num_nodes, 1].
        - embedding (torch.Tensor): The embedded target data with shape [batch_size, num_samples + 1, num_nodes, d_model].
        ...
        - query (torch.Tensor): The query tensor with shape [batch_size, 1, num_nodes, d_model].
        - key (torch.Tensor): The key tensor with shape [batch_size, num_samples, num_nodes, d_model].
        - value (torch.Tensor): The value tensor with shape [batch_size, num_samples, num_nodes, d_model].
        - summary_rep (torch.Tensor): The summary representation with shape [batch_size, num_nodes, 1, d_model].
        ...
        Encode the target data and compute the summary representation.
        - target_data (torch.Tensor): The target data with shape [batch_size, num_samples, num_nodes, 1].
        - summary_rep (torch.Tensor): The summary representation with shape [batch_size, num_nodes, 1, d_model].
        ...
    """

    def __init__(
        self,
        d_model,
        dim_feedforward,
        nhead,
        num_layers,
        use_positional_encoding,
        num_nodes,
        device,
        dtype,
        emb_depth: int = 1,
        avici_summary: bool = False,
        dropout: Optional[float] = 0.0,
    ):
        super(CausalTNPEncoder, self).__init__()
        self.embedder = build_mlp(
            dim_in=1,
            dim_hid=d_model if not use_positional_encoding else d_model // 2,
            dim_out=d_model if not use_positional_encoding else d_model // 2,
            depth=emb_depth,
        )
        module = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            bias=False,
            batch_first=True,
            norm_first=True,
            device=device,
            dtype=dtype,
        )
        encoderlayers = nn.ModuleList(
            [copy.deepcopy(module) for i in range(num_layers)]
        )
        self.encoder = CausalTransformerEncoder(
            encoder_layers=encoderlayers,
        )
        self.representation = nn.MultiheadAttention(
            d_model,
            nhead,
            batch_first=True,
            device=device,
            dtype=dtype,
        )
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model=d_model // 2, dropout=0.0, max_len=num_nodes)

        self.avici_summary = avici_summary

    def embed(self, target_data):
        """
        Embed the target data into a d_model dimensional space.

        Args:
        --------
            target_data: torch.Tensor, shape [batch_size, num_samples, num_nodes, 1]

        Returns:
        --------
            embedding: torch.Tensor, shape [batch_size, num_samples + 1, num_nodes, d_model]
        """
        # shape [batch_size, num_samples, num_nodes, d_model]
        embedding = self.embedder(target_data)
        if self.use_positional_encoding:
            pos_embedding = self.positional_encoding(target_data)
            embedding = torch.cat([embedding, pos_embedding], dim=-1)
        # Concatenate 0s to samples to use as query
        query_emb = torch.zeros_like(embedding[:, 0:1, :, :])
        embedding = torch.cat([embedding, query_emb], dim=1)
        return embedding

    def compute_summary(self, query, key, value, avici_summary=False):
        """
        Compute the summary representation for the query.

        Args:
        -----
            query: torch.Tensor, shape [batch_size, 1, num_nodes, d_model]
            key: torch.Tensor, shape [batch_size, num_samples, num_nodes, d_model]
            value: torch.Tensor, shape [batch_size, num_samples, num_nodes, d_model]
            avici_summary: bool, whether to use the Avici summary. This is a max
                pool over the samples.

        Returns:
        --------
            summary_rep: torch.Tensor, shape [batch_size, num_nodes, 1, d_model]
        """
        if avici_summary:
            # Max pool over the value
            # shape [batch_size, 1, num_nodes, d_model]
            summary_rep = value.max(dim=1, keepdim=True)[0]
            # shape [batch_size, num_nodes, 1, d_model]
            summary_rep = summary_rep.permute(0, 2, 1, 3)
            return summary_rep
        else:
            # Perform attention over the samples
            batch, num_samples, num_nodes, d_model = key.size()
            # shape [batch, num_nodes, 1, d_model]
            query = query.permute(0, 2, 1, 3)
            query = query.contiguous().view(batch * num_nodes, 1, d_model)
            # shape [batch, num_nodes, num_samples, d_model]
            key = key.permute(0, 2, 1, 3)
            key = key.contiguous().view(batch * num_nodes, num_samples, d_model)
            # shape [batch, num_nodes, num_samples, d_model]
            value = value.permute(0, 2, 1, 3)
            value = value.contiguous().view(batch * num_nodes, num_samples, d_model)
            # shape [batch * num_nodes, 1, d_model]
            summary_rep = self.representation(
                query=query,
                key=key,
                value=value,
            )[0]
            summary_rep = summary_rep.contiguous().view(batch, num_nodes, 1, d_model)
            return summary_rep

    def encode(self, target_data):
        # First step is to embed the nodes and samples
        # shape [batch_size, num_samples + 1, num_nodes, d_model]
        embedding = self.embed(target_data)
        # Encode the data
        # TODO: Take advantage of fastpath for causal transformer!
        # shape [batch_size, num_samples + 1, num_nodes, d_model]
        representation = self.encoder(embedding)
        query_rep = representation[:, -1:, :, :]
        # shape [batch_size, num_nodes, 1, d_model]

        summary_rep = self.compute_summary(
            query=query_rep,
            key=representation[:, :-1, :, :],
            value=representation[:, :-1, :, :],
            avici_summary=self.avici_summary,
        )
        return summary_rep