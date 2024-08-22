import torch
import torch.nn as nn
from typing import Callable, Optional
from torch import Tensor
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import math
from torch.nn.init import xavier_uniform_
from ml2_meta_causal_discovery.utils.permutations import sample_permutation, sinkhorn
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
        emb_depth: int = 2,
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
            batch_first=True,
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

    def compute_summary(self, query, key, value):
        """
        Compute the summary representation for the query.

        Args:
        -----
            query: torch.Tensor, shape [batch_size, 1, num_nodes, d_model]
            key: torch.Tensor, shape [batch_size, num_samples, num_nodes, d_model]
            value: torch.Tensor, shape [batch_size, num_samples, num_nodes, d_model]

        Returns:
        --------
            summary_rep: torch.Tensor, shape [batch_size, num_nodes, 1, d_model]
        """
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
        )
        return summary_rep


class CausalTNPDecoder(CausalTNPEncoder):

    def __init__(
        self,
        d_model,
        emb_depth,
        dim_feedforward,
        nhead,
        dropout,
        num_layers_encoder,
        num_layers_decoder,
        use_positional_encoding,
        num_nodes,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super(CausalTNPDecoder, self).__init__(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            num_layers=num_layers_encoder,
            emb_depth=emb_depth,
            use_positional_encoding=use_positional_encoding,
            num_nodes=num_nodes,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=CausalTransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                norm_first=True,
                batch_first=True,
            ),
            num_layers=num_layers_decoder,
        )
        self.predictor = CausalAdjacencyMatrix(
            nhead=nhead,
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

    def decode(self, representation):
        # shape [batch_size, num_nodes, d_model]
        decoder_rep = self.decoder(tgt=representation, memory=None)
        return decoder_rep

    def calculate_loss(self, logits, target):
        """
        Args:
        -----
            logits: torch.Tensor, shape [batch_size, num_samples, num_nodes, num_nodes]
            target: torch.Tensor, shape [batch_size, num_nodes, num_nodes]

        Returns:
        --------
            loss: torch.Tensor, shape [batch_size]
            logits: torch.Tensor, shape [batch_size, num_nodes ** 2]
        """
        logits = logits.contiguous().view(logits.size(0), logits.size(1), -1)
        target = target.contiguous().view(target.size(0), -1)
        # Classification loss
        loss_func = torch.nn.BCEWithLogitsLoss(reduction="none")
        # TODO: Diagonal is always 0! We can hardcode this!
        loss = loss_func(logits, target)
        loss = loss.mean(dim=1)
        return loss

    def forward(self, target_data, graph, is_training=True):
        # target_data: [batch_size, num_samples, num_nodes]
        if target_data.dim() == 3:
            target_data = target_data.unsqueeze(-1)
        # Extract representation
        # shape [batch_size, num_nodes, 1, d_model]
        representation = self.encode(target_data=target_data)
        # Decode the representation
        representation = representation.squeeze(2)
        out = self.decode(representation=representation)
        # Final predictor for adjacency matrix
        adj_matrix = self.predictor(out)
        # graph is shape [batch_size, num_nodes, num_nodes]
        # adj_matrix is shape [batch_size, num_nodes, num_nodes]
        target_graph = target_graph.view(target_graph.size(0), -1)
        adj_matrix = adj_matrix.view(adj_matrix.size(0), -1)
        return adj_matrix


class CausalAutoregressiveDecoder(CausalTNPEncoder):

    def __init__(
        self,
        d_model,
        emb_depth,
        dim_feedforward,
        nhead,
        dropout,
        num_layers_encoder,
        num_layers_decoder,
        num_nodes,
        use_positional_encoding,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super(CausalAutoregressiveDecoder, self).__init__(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            num_layers=num_layers_encoder,
            emb_depth=emb_depth,
            use_positional_encoding=use_positional_encoding,
            num_nodes=num_nodes,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )
        self.output_embedder = build_mlp(
            dim_in=1,
            dim_hid=d_model,
            dim_out=d_model,
            depth=emb_depth,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                norm_first=True,
                batch_first=True,
                device=device,
                dtype=dtype,
            ),
            num_layers=num_layers_decoder,
        )
        self.predictor = build_mlp(
            dim_in=d_model,
            dim_hid=d_model,
            dim_out=1,
            depth=emb_depth,
        )

    def decode(self, representation, targets, is_training=True):
        # shape [batch_size, num_nodes, d_model]
        if is_training:
            # we will auto-regressively predict the adjacency matrix
            # targets will be the target_graph with -1 as the first one
            # shape [batch_size, num_nodes ** 2, 1]
            target_graph = targets.view(targets.size(0), -1)[:, :, None]
            minus_one_trgt = torch.ones_like(target_graph[:, 0:1, :]) * -1
            full_target = torch.cat([minus_one_trgt, target_graph], dim=1)[:, :-1, :]
            full_target_emb = self.output_embedder(full_target)
            tgt_mask = torch.zeros((full_target_emb.size(1), full_target_emb.size(1)), device=full_target_emb.device).fill_(float('-inf'))
            tgt_mask = tgt_mask.triu_(1)
            decoder_rep = self.decoder(
                tgt=full_target_emb,
                memory=representation,
                tgt_mask=tgt_mask,
            )
            tgt_input = full_target
        else:
            num_nodes = representation.size(1)
            adj_size = num_nodes ** 2
            # shape [batch_size, 1, 1]
            tgt_input = torch.ones_like(representation[:, 0:1, 0:1]) * -1
            while True:
                # shape [batch_size, loop_iteration, d_model]
                tgt_emb = self.output_embedder(tgt_input)
                # Same tgt_mask in validation as for training
                tgt_mask = torch.zeros((tgt_emb.size(1), tgt_emb.size(1)), device=tgt_emb.device).fill_(float('-inf'))
                tgt_mask = tgt_mask.triu_(1)
                # size of decoder rep will be [batch, tgt_input.size(1), d_model]
                decoder_rep = self.decoder(
                    tgt=tgt_emb,
                    memory=representation,
                    tgt_mask=tgt_mask,
                )
                # sample bernoulli distribution
                logit = self.predictor(decoder_rep[:, -1:, :])
                # shape [batch_size, 1, 1]
                bernoulli = torch.bernoulli(torch.sigmoid(logit))
                tgt_input = torch.cat([tgt_input, bernoulli], dim=1)
                if tgt_input.size(1) - 1 == adj_size:
                    break
        return decoder_rep, tgt_input

    def calculate_loss(self, logits, target):
        """
        Args:
        -----
            logits: torch.Tensor, shape [batch_size, num_nodes, num_nodes]
            target: torch.Tensor, shape [batch_size, num_nodes, num_nodes]

        Returns:
        --------
            loss: torch.Tensor, shape [batch_size]
            logits: torch.Tensor, shape [batch_size, num_nodes ** 2]
        """
       #  shape [batch_size, num_nodes ** 2]
        logits = logits.contiguous().view(logits.size(0), -1)
        target_graph = target.view(target.size(0), -1)
        # Classification loss
        loss_func = torch.nn.BCEWithLogitsLoss(reduction="none")
        # TODO: Diagonal is always 0! We can hardcode this!
        loss = loss_func(logits, target_graph)
        loss = loss.mean(dim=1)
        return loss

    def forward(self, target_data, graph, is_training=True):
        """
        Args:
        -----
            target_data: torch.Tensor, shape [batch_size, num_samples, num_nodes]
            graph: torch.Tensor, shape [batch_size, num_nodes, num_nodes]
                This is needed for teacher forcing
            is_training: bool.
                during training, we will use the ground truth adjacency matrix
                but during inference, we will use the predicted adjacency matrix

        Returns:
        --------
            all_logits: torch.Tensor, shape [num_samples, batch_size, num_nodes, num_nodes]
                Logits of the adjacency matrix of the DAG.
        """
        # target_data: [batch_size, num_samples, num_nodes]
        if target_data.dim() == 3:
            target_data = target_data.unsqueeze(-1)
        # Extract representation
        # shape [batch_size, num_nodes, 1, d_model]
        representation = self.encode(target_data=target_data)
        # Decode the representation
        # shape [batch_size, num_nodes, d_model]
        representation = representation.squeeze(2)
        # shape [batch_size, num_nodes ** 2, d_model]
        out, predict_graph = self.decode(representation=representation, targets=graph.clone(), is_training=is_training)
        # shape [batch_size, num_nodes ** 2]
        logit = self.predictor(out).squeeze(-1)
        logit = logit.reshape(logit.size(0), graph.size(1), graph.size(2))
        return logit


class CausalProbabilisticDecoder(CausalTNPEncoder):

    def __init__(
        self,
        d_model,
        emb_depth,
        dim_feedforward,
        nhead,
        dropout,
        num_layers_encoder,
        num_layers_decoder,
        num_nodes,
        n_perm_samples,
        sinkhorn_iter,
        use_positional_encoding,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super(CausalProbabilisticDecoder, self).__init__(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            num_layers=num_layers_encoder,
            emb_depth=emb_depth,
            num_nodes=num_nodes,
            use_positional_encoding=use_positional_encoding,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )
        self.num_nodes = num_nodes
        self.n_perm_samples = n_perm_samples
        self.sinkhorn_iter = sinkhorn_iter
        self.output_embedder = build_mlp(
            dim_in=1,
            dim_hid=d_model,
            dim_out=d_model,
            depth=emb_depth,
        )
        # Decoder for the adjacency matrix
        # A = QLQ^Q where L is a lower triangular matrix
        # Q is the permutation matrix
        print(f"Using {num_layers_decoder // 2} decoder layers.")
        self.decoder_L = nn.TransformerDecoder(
            decoder_layer=CausalTransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                norm_first=True,
                batch_first=True,
                device=device,
                dtype=dtype,
            ),
            num_layers=num_layers_decoder // 2,
        )
        self.decoder_Q = nn.TransformerDecoder(
            decoder_layer=CausalTransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                norm_first=True,
                batch_first=True,
                device=device,
                dtype=dtype,
            ),
            num_layers=num_layers_decoder // 2,
        )
        self.Q_param = CausalAdjacencyMatrix(
            nhead=nhead,
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        self.L_param = CausalAdjacencyMatrix(
            nhead=nhead,
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        self.p_param = build_mlp(
            dim_in=d_model,
            dim_hid=d_model,
            dim_out=1,
            depth=emb_depth,
        )

    def decode(self, representation, is_training=True):
        # shape [batch_size, num_nodes, d_model]
        L_rep = self.decoder_L(representation, memory=None)
        # We will pass L_param into permutation
        Q_rep = self.decoder_Q(L_rep, memory=None)
        # shape [batch_size, num_nodes, num_nodes]
        L_param = self.L_param(L_rep)
        # Q_param = self.Q_param(Q_rep)
        return L_param, Q_rep

    def calculate_loss(self, probs, target):
        """
        Args:
        -----
            probs: torch.Tensor, shape [batch_size, num_samples, num_nodes, num_nodes]
            target: torch.Tensor, shape [batch_size, num_nodes, num_nodes]

        Returns:
        --------
            loss: torch.Tensor, shape [batch_size]
        """
        # Reshape the last axis
        probs = probs.contiguous().view(probs.size(0), probs.size(1), -1)
        target_graph = target.view(target.size(0), -1)
        # Calculate the loss
        existence_dist = torch.distributions.Bernoulli(
            probs=probs
        )
        log_prob = existence_dist.log_prob(target_graph[None])
        # # Mean across pemutation samples
        log_prob_sum = torch.logsumexp(log_prob, dim=0) - math.log(log_prob.size(0))
        # # shape [batch, num_nodes**2]
        loss_per_edge = - log_prob_sum
        loss = loss_per_edge.mean(dim=1)
        return loss

    def forward(self, target_data, graph, is_training=True):
        """
        Args:
        -----
            target_data: torch.Tensor, shape [batch_size, num_samples, num_nodes]
            is_training: bool.
                during training, we will use the ground truth adjacency matrix
                but during inference, we will use the predicted adjacency matrix
                Only needed for autoregressive model.

        Returns:
        --------
            probs: torch.Tensor, shape [num_samples, batch_size, num_nodes, num_nodes]
                probs of the adjacency matrix of the DAG.
        """
        # target_data: [batch_size, num_samples, num_nodes]
        if target_data.dim() == 3:
            target_data = target_data.unsqueeze(-1)
        # Extract representation
        # shape [batch_size, num_nodes, 1, d_model]
        representation = self.encode(target_data=target_data)
        # Decode the representation
        # shape [batch_size, num_nodes, d_model]
        representation = representation.squeeze(2)
        # L: shape [batch_size, num_nodes, num_nodes]
        # Q: shape [batch_size, num_nodes, d_model]
        L_param, Q_rep = self.decode(representation=representation)
        # shape [batch_size, num_nodes]
        p_param = self.p_param(Q_rep).squeeze(-1)
        ovector = torch.arange(
            self.num_nodes,
            device=p_param.device,
            dtype=p_param.dtype
        )
        Q_param = torch.einsum(
            "bn,m->bnm",
            p_param,
            ovector[: representation.size(1)],
        )
        # Sample permutations
        # shape = [batch_size, n_samples, num_nodes, num_nodes]
        Q_param = torch.functional.F.logsigmoid(Q_param)
        perm, _ = sample_permutation(
            log_alpha=Q_param,
            temp=1.0,
            noise_factor=1.0,
            n_samples=self.n_perm_samples,
            hard=True,
            n_iters=self.sinkhorn_iter,
            squeeze=False,
            device=Q_param.device,
        )
        perm = perm.transpose(1, 0)
        perm_inv = torch.transpose(perm, 3, 2)
        # # All matrices
        # extract mask for variable node size
        mask = torch.tril(
            torch.ones(
                (self.num_nodes, self.num_nodes),
                device=perm.device,
                dtype=perm.dtype
            ),
            diagonal=-1
        )
        my_mask = mask[: representation.size(1), : representation.size(1)]
        all_masks = torch.einsum(
            "bnij,jk,bnkl->bnil",
            perm,
            my_mask,
            perm_inv,
        )
        # Find probs
        probs = torch.sigmoid(L_param)
        # shape [num_samples, batch_size, num_nodes, num_nodes]
        # Elementwise multiplication
        all_probs = torch.mul(probs[None], all_masks)
        return all_probs

    def sample(self, target_data: torch.Tensor, num_samples: int):
        """
        Sample DAGs, one for each permutation.

        Returns:
        --------
            samples: torch.Tensor, shape [num_samples, batch_size, num_nodes, num_nodes]
        """
        # Override number of samples
        self.n_perm_samples = num_samples
        # probs: [num_samples, batch_size, num_nodes, num_nodes]
        probs = self.forward(target_data, graph=None, is_training=False)
        # Sample from the distribution
        existence_dist = torch.distributions.Bernoulli(
            probs=probs
        )
        samples = existence_dist.sample()
        return samples

