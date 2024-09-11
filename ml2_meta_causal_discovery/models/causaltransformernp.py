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
from ml2_meta_causal_discovery.models.causaltransformercomponents import (
    CausalTNPEncoder,
    CausalAdjacencyMatrix,
    CausalTransformerDecoderLayer,
    build_mlp,
)


class AviciDecoder(CausalTNPEncoder):

    """
    Differences:
    - Max pool for summary representation in encoder
    - No decoder
    - Linear layer after which attention operation gives adjacency matrix
    """

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
        super(AviciDecoder, self).__init__(
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
            avici_summary=True, # This is the only difference in encoding
        )
        # Decoder is a linear layer.
        # The linear layer with heads is implemented in CausalAdjacencyMatrix
        self.decoder = nn.Identity()

        self.predictor = CausalAdjacencyMatrix(
            nhead=1, # There is only one head for the final prediction
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

    def decode(self, representation):
        # shape [batch_size, num_nodes, d_model]
        decoder_rep = self.decoder(representation)
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
        logits = logits.contiguous().view(logits.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)
        # Classification loss
        loss_func = torch.nn.BCEWithLogitsLoss(reduction="none")
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
        return adj_matrix

    def sample(self, target_data: torch.Tensor, num_samples: int):
        """
        Sample. num_samples here is samples of the graph.

        Returns:
        --------
            samples: torch.Tensor, shape [num_samples, batch_size, num_nodes, num_nodes]
        """
        adj_matrix = self.forward(target_data, graph=None, is_training=False)
        # Sample from the distribution using a Bernoulli distribution
        existence_dist = torch.distributions.Bernoulli(
            probs=torch.nn.Sigmoid()(adj_matrix)
        )
        samples = existence_dist.sample(
            sample_shape=(num_samples,)
        )
        return samples


class CsivaDecoder(CausalTNPEncoder):

    """"
    Differences:
    - Autoregressive decoder
    """

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
        super(CsivaDecoder, self).__init__(
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
        num_nodes = target_data.size(-1)
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
        out, predict_graph = self.decode(
            representation=representation,
            targets=graph.clone() if is_training else graph,
            is_training=is_training
        )
        # shape [batch_size, num_nodes ** 2]
        logit = self.predictor(out).squeeze(-1)
        logit = logit.reshape(logit.size(0), num_nodes, num_nodes)

        if is_training:
            return logit
        else:
            predict_graph = predict_graph.squeeze(-1)[:, 1:]
            predict_graph = predict_graph.view(predict_graph.size(0), num_nodes, num_nodes)
            return logit, predict_graph

    def sample(self, target_data: torch.Tensor, num_samples: int):
        """
        Sample. num_samples here is samples of the graph.

        Returns:
        --------
            samples: torch.Tensor, shape [num_samples, batch_size, num_nodes, num_nodes]
        """
        all_samples = torch.zeros(
            (num_samples, target_data.size(0), target_data.size(-1), target_data.size(-1))
        )
        for i in range(num_samples):
            _, sample = self.forward(target_data, graph=None, is_training=False)
            all_samples[i] = sample
        return all_samples


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
        Q_before_L=True, # Whether to infer Q (perm) before L (bernoulli)
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
        self.Q_before_L = Q_before_L
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
                bias=False,
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
                bias=False,
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
        if not self.Q_before_L:
            # shape [batch_size, nu`m_nodes, d_model]
            L_rep = self.decoder_L(representation, memory=None)
            # We will pass L_param into permutation
            Q_rep = self.decoder_Q(L_rep, memory=None)
            # shape [batch_size, num_nodes, num_nodes]
            L_param = self.L_param(L_rep)
            # Q_param = self.Q_param(Q_rep)
        else:
            # shape [batch_size, num_nodes, d_model]
            Q_rep = self.decoder_Q(representation, memory=None)
            # We will pass Q_param into L
            L_rep = self.decoder_L(Q_rep, memory=None)
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
        if self.num_nodes != target_data.size(-1):
            raise ValueError("Number of nodes in the input data should be equal to num_nodes.")
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
        # tril doesnt work on some dtypes
        mask = torch.tril(
            torch.ones(
                (self.num_nodes, self.num_nodes),
                device=perm.device,
                dtype=torch.float32,
            ),
            diagonal=-1
        ).to(perm.dtype)
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

