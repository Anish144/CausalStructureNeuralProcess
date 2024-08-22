"""
Losses for the meta causal discovery model.

Main loss used will be the IS version of the loss from:
https://arxiv.org/abs/2007.01332
"""
import abc
import math

import torch as th
import torch.nn as nn
from npf.losses import BaseLossNPF
from ml2_meta_causal_discovery.utils.autoregeressive_helper import (
    get_next_data_dict,
)
from torch.distributions.independent import Independent
from attrdict import AttrDict
from torch.distributions.kl import kl_divergence
from tqdm import trange


def sum_from_nth_dim(t, dim):
    """Sum all dims from `dim`. E.g. sum_after_nth_dim(torch.rand(2,3,4,5), 2).shape = [2,3]"""
    return t.view(*t.shape[:dim], -1).sum(-1)


def sum_log_prob(prob, sample, dim=2):
    """Compute log probability then sum all but the z_samples and batch."""
    # size = [n_z_samples, batch_size, *]
    log_p = prob.log_prob(sample)
    # size = [n_z_samples, batch_size]
    sum_log_p = sum_from_nth_dim(log_p, dim=dim)
    return sum_log_p


def get_loss_dist(all_dist, Y_trgt, inverse_transform):
    all_means = [dist.base_dist.loc.detach() for dist in all_dist]
    all_stds = [dist.base_dist.scale.detach() for dist in all_dist]
    # shape (batch_size=1, n_samples, n_targets, y_dim)
    all_means = th.cat(all_means, dim=2)
    all_stds = th.cat(all_stds, dim=2)
    # To make sure that we mean over the correct samples, we have to revert the
    # permutation
    means_reverted = th.gather(all_means, dim=2, index=inverse_transform[None])
    stds_reverted = th.gather(all_stds, dim=2, index=inverse_transform[None])
    Y_reverted = th.gather(Y_trgt, dim=1, index=inverse_transform)
    sample_dist = Independent(
        th.distributions.Normal(means_reverted, stds_reverted), 1
    )
    return sample_dist, Y_reverted


def compute_autoregressive_loss(
    model, X_cntxt, Y_cntxt, X_trgt, Y_trgt, n_samples, loss,
    X_trgt_int=None, Y_trgt_int=None,
    intervention=False
):
    """Compute the autoregressive loss."""
    num_target = X_trgt.shape[1]
    if intervention:
        # The intervention and target points are the same
        # Otherwise this will not work!
        assert th.equal(X_trgt_int, X_trgt)
        Xi = dict(
            X_cntxt=X_cntxt, Y_cntxt=Y_cntxt, X_trgt=X_trgt, X_trgt_int=X_trgt_int
        )
    else:
        Xi = dict(X_cntxt=X_cntxt, Y_cntxt=Y_cntxt, X_trgt=X_trgt)
    # Need to tile Y_cnxt
    Y_cntxt_new = th.tile(Xi["Y_cntxt"], (n_samples, 1, 1))
    X_cntxt_new = th.tile(Xi["X_cntxt"], (n_samples, 1, 1))
    X_trgt_new = th.tile(Xi["X_trgt"], (n_samples, 1, 1))
    Y_trgt_new = th.tile(Y_trgt, (n_samples, 1, 1))
    # Shuffle the target points
    # Generate random indices of points
    indices = th.argsort(th.rand_like(X_trgt_new), dim=1)
    inverse_transform = th.argsort(indices, dim=1)
    X_trgt_new = th.gather(X_trgt_new, dim=1, index=indices)
    Y_trgt_new = th.gather(Y_trgt_new, dim=1, index=indices)
    Xi["Y_cntxt"] = Y_cntxt_new
    Xi["X_cntxt"] = X_cntxt_new
    Xi["X_trgt"] = X_trgt_new
    if intervention:
        X_trgt_int_new = th.tile(Xi["X_trgt_int"], (n_samples, 1, 1))
        Y_trgt_int_new = th.tile(Y_trgt_int, (n_samples, 1, 1))
        X_trgt_int_new = th.gather(X_trgt_int_new, dim=1, index=indices)
        Y_trgt_int_new = th.gather(Y_trgt_int_new, dim=1, index=indices)
        Xi["X_trgt_int"] = X_trgt_int_new
    all_dist = []
    all_dist_int = []
    Xi = {k: v.to("cpu") for k,v in Xi.items()}
    for i in trange(num_target):
        if i == 0:
            new_Xi = get_next_data_dict(Xi, i, torch=True)
            new_Xi = {k: v.to("cuda") for k,v in new_Xi.items()}
            y_pred_full = model.forward(**new_Xi)
            y_pred_dist = y_pred_full[0]
            y_pred_sample = y_pred_dist.sample()[0, :, :, :]
            if intervention:
                y_pred_int_dist = y_pred_full[1]
            new_Xi = {k: v.to("cpu") for k,v in new_Xi.items()}
        else:
            y_pred_sample = y_pred_sample.to("cpu")
            new_Xi = {k: v.to("cpu") for k,v in new_Xi.items()}
            new_Xi = get_next_data_dict(
                Xi, i, new_Xi, y_pred_sample, torch=True, intervention=intervention
            )
            new_Xi = {k: v.to("cuda") for k,v in new_Xi.items()}
            y_pred_full = model.forward(**new_Xi)
            y_pred_dist = y_pred_full[0]
            y_pred_sample = y_pred_full[0].sample()[0, :, :, :]
            if intervention:
                y_pred_int_dist = y_pred_full[1]
        # y_pred_dist.to("cpu")
        # y_pred_int_dist.to("cpu")
        all_dist.append(y_pred_dist)
        if intervention:
            all_dist_int.append(y_pred_int_dist)

    new_Xi = get_next_data_dict(Xi, i, new_Xi, y_pred_sample, torch=True)

    Y_trgt_new = Y_trgt_new.to("cpu")
    sample_dist, Y_reverted = get_loss_dist(all_dist, Y_trgt_new, inverse_transform)
    # Compute the log prob
    loss_instance = loss(
        pred_outputs=(sample_dist, None, None, None), Y_trgt=Y_reverted
    )
    if intervention:
        Y_trgt_int_new = Y_trgt_int_new.to("cpu")
        sample_dist_int, Y_int_reverted = get_loss_dist(all_dist_int, Y_trgt_int_new, inverse_transform)
        # Compute the log prob
        loss_int_instance = loss(
            pred_outputs=(sample_dist_int, None, None, None), Y_trgt=Y_int_reverted
        )
        return loss_instance, loss_int_instance
    else:
        return loss_instance


class BaseLossNPFInterventional(nn.Module, abc.ABC):
    """
    Compute the negative log likelihood loss for members of the conditional neural process (sub-)family.
    This base class makes sure that losses for interventional samples can be computed.

    Parameters
    ----------
    reduction : {None,"mean","sum"}, optional
        Batch wise reduction.

    is_force_mle_eval : bool, optional
        Whether to force mac likelihood eval even if has access to q_zCct
    """

    def __init__(self, reduction="mean", is_force_mle_eval=True):
        super().__init__()
        self.reduction = reduction
        self.is_force_mle_eval = is_force_mle_eval

    def forward(self, pred_outputs, Y_trgt_all):
        """Compute the Neural Process Loss.

        Parameters
        ----------
        pred_outputs : tuple
            Output of `NeuralProcessFamily`.

        Y_trgt : torch.Tensor, size=[batch_size, *n_trgt, y_dim]
            Set of all target values {y_t}.

        Return
        ------
        loss : torch.Tensor
            size=[batch_size] if `reduction=None` else [1].
        """
        p_yCc, p_yCc_int, z_samples, q_zCc, q_zCct = pred_outputs
        Y_trgt = Y_trgt_all["Y_trgt"]
        Y_trgt_int = Y_trgt_all["Y_trgt_int"]

        if self.training:
            loss = self.get_loss(
                p_yCc, p_yCc_int, z_samples, q_zCc, q_zCct, Y_trgt, Y_trgt_int
            )
        else:
            # always uses NPML for evaluation
            if self.is_force_mle_eval:
                q_zCct = None
            loss = NLLLossLNPFInterventional.get_loss(
                self, p_yCc, p_yCc_int, z_samples, q_zCc, q_zCct, Y_trgt, Y_trgt_int
            )

        if self.reduction is None:
            # size = [batch_size]
            return loss
        elif self.reduction == "mean":
            # size = [1]
            return loss.mean(0)
        elif self.reduction == "sum":
            # size = [1]
            return loss.sum(0)
        else:
            raise ValueError(f"Unknown {self.reduction}")

    @abc.abstractmethod
    def get_loss(
        self, p_yCc, p_yCc_int, z_samples, q_zCc, q_zCct, Y_trgt, Y_trgt_int
    ):
        """Compute the Neural Process Loss

        Parameters
        ------
        p_yCc: torch.distributions.Distribution, batch shape=[n_z_samples, batch_size, *n_trgt] ; event shape=[y_dim]
            Posterior distribution for target values {p(Y^t|y_c; x_c, x_t)}_t

        p_yCc_int: torch.distributions.Distribution, batch shape=[n_z_samples, batch_size, *n_trgt] ; event shape=[y_dim]
            Posterior distribution for interventional target values {p(Y^t|y_c; x_c, do(x_t))}_t

        z_samples: torch.Tensor, size=[n_z_samples, batch_size, *n_lat, z_dim]
            Sampled latents. `None` if `encoded_path==deterministic`.

        q_zCc: torch.distributions.Distribution, batch shape=[batch_size, *n_lat] ; event shape=[z_dim]
            Latent distribution for the context points. `None` if `encoded_path==deterministic`.

        q_zCct: torch.distributions.Distribution, batch shape=[batch_size, *n_lat] ; event shape=[z_dim]
            Latent distribution for the targets. `None` if `encoded_path==deterministic`
            or not training or not `is_q_zCct`.

        Y_trgt: torch.Tensor, size=[batch_size, *n_trgt, y_dim]
            Set of all target values {y_t}.

        Y_trgt_int: torch.Tensor, size=[batch_size, *n_trgt, y_dim]
            Set of all target values {y_t | do(x_t)}.

        Return
        ------
        loss : torch.Tensor, size=[1].
        """
        pass


class NLLLossLNPFInterventional(BaseLossNPFInterventional):
    """
    Compute the approximate negative log likelihood for Neural Process family[?].

     Notes
    -----
    - might be high variance
    - biased
    - approximate because expectation over q(z|cntxt) instead of p(z|cntxt)
    - if q_zCct is not None then uses importance sampling (i.e. assumes that sampled from it).

    References
    ----------
    [?]
    """

    def get_loss(
        self, p_yCc, p_yCc_int, z_samples, q_zCc, q_zCct, Y_trgt, Y_trgt_int
    ):
        n_z_samples, batch_size, *n_trgt = p_yCc.batch_shape

        # computes approximate LL in a numerically stable way
        # LL = E_{q(z|y_cntxt)}[ \prod_t p(y^t|z)]
        # LL MC = log ( mean_z ( \prod_t p(y^t|z)) )
        # = log [ sum_z ( \prod_t p(y^t|z)) ] - log(n_z_samples)
        # = log [ sum_z ( exp \sum_t log p(y^t|z)) ] - log(n_z_samples)
        # = log_sum_exp_z ( \sum_t log p(y^t|z)) - log(n_z_samples)

        # \sum_t log p(y^t|z). size = [n_z_samples, batch_size]
        sum_log_p_yCz = sum_log_prob(p_yCc, Y_trgt)
        sum_log_p_yCz_int = sum_log_prob(p_yCc_int, Y_trgt_int)
        # uses importance sampling weights if necessary
        if q_zCct is not None:
            # All latents are treated as independent. size = [n_z_samples, batch_size]
            sum_log_q_zCc = sum_log_prob(q_zCc, z_samples)
            sum_log_q_zCct = sum_log_prob(q_zCct, z_samples)

            # importance sampling : multiply \prod_t p(y^t|z)) by q(z|y_cntxt) / q(z|y_cntxt, y_trgt)
            # i.e. add log q(z|y_cntxt) - log q(z|y_cntxt, y_trgt)
            sum_log_w_k = (
                sum_log_p_yCz
                + sum_log_p_yCz_int
                + sum_log_q_zCc
                - sum_log_q_zCct
            )
        else:
            sum_log_w_k = sum_log_p_yCz + sum_log_p_yCz_int

        # log_sum_exp_z ... . size = [batch_size]
        log_S_z_sum_p_yCz = th.logsumexp(sum_log_w_k, 0)

        # - log(n_z_samples)
        log_E_z_sum_p_yCz = log_S_z_sum_p_yCz - math.log(n_z_samples)

        # NEGATIVE log likelihood
        return -log_E_z_sum_p_yCz


class ELBOLossLNPFInterventional(BaseLossNPFInterventional):
    """Approximate conditional ELBO [1].

    References
    ----------
    [1] Garnelo, Marta, et al. "Neural processes." arXiv preprint
        arXiv:1807.01622 (2018).
    """

    def get_loss(self, p_yCc, p_yCc_int, _, q_zCc, q_zCct, Y_trgt, Y_trgt_int):

        # first term in loss is E_{q(z|y_cntxt,y_trgt)}[\sum_t log p(y^t|z)]
        # \sum_t log p(y^t|z). size = [z_samples, batch_size]
        sum_log_p_yCz = sum_log_prob(p_yCc, Y_trgt)
        sum_log_p_yCz_int = sum_log_prob(p_yCc_int, Y_trgt_int)

        # E_{q(z|y_cntxt,y_trgt)}[...] . size = [batch_size]
        E_z_sum_log_p_yCz = sum_log_p_yCz.mean(0)
        E_z_sum_log_p_yCz_int = sum_log_p_yCz_int.mean(0)

        # second term in loss is \sum_l KL[q(z^l|y_cntxt,y_trgt)||q(z^l|y_cntxt)]
        # KL[q(z^l|y_cntxt,y_trgt)||q(z^l|y_cntxt)]. size = [batch_size, *n_lat]
        kl_z = kl_divergence(q_zCct, q_zCc)
        # \sum_l ... . size = [batch_size]
        E_z_kl = sum_from_nth_dim(kl_z, 1)

        return -(E_z_sum_log_p_yCz + E_z_sum_log_p_yCz_int - 2 * E_z_kl)


def sum_log_mean_prob(prob, sample):
    """Compute log probability then sum all but the z_samples and batch."""
    # size = [n_z_samples, batch_size, *]
    log_p = th.log(th.mean(th.exp(prob.log_prob(sample)), 1))
    # size = [n_z_samples, batch_size]
    sum_log_p = sum_from_nth_dim(log_p, 1)
    return sum_log_p


class IntCNPFLoss(BaseLossNPFInterventional):
    """Losss for conditional neural process (suf-)family [1]."""

    def get_loss(self, p_yCc, p_yCc_int, z_samples, q_zCc, q_zCct, Y_trgt, Y_trgt_int):
        assert q_zCc is None
        # \sum_t log p(y^t|z)
        # \sum_t log p(y^t|z). size = [z_samples, batch_size]
        sum_log_p_yCz = sum_log_prob(p_yCc, Y_trgt)
        sum_log_p_yCz_int = sum_log_prob(p_yCc_int, Y_trgt_int)

        # size = [batch_size]
        nll = - sum_log_p_yCz.squeeze(0)
        nll_int = - sum_log_p_yCz_int.squeeze(0)
        return nll + nll_int


class CNPFAutoLoss(BaseLossNPF):
    """Losss for autoregressive conditional neural process (suf-)family [1]."""

    def get_loss(self, p_yCc, _, q_zCc, ___, Y_trgt):
        assert q_zCc is None
        # \sum_t log p(y^t|z)
        # \sum_t log p(y^t|z). size = [z_samples, batch_size]
        sum_log_p_yCz = sum_log_mean_prob(p_yCc, Y_trgt)

        # size = [batch_size]
        nll = -sum_log_p_yCz.squeeze(0)
        return nll


class TransformerLoss(nn.Module):
    """
    Computes the loss for the transformer model.

    Note that this is already done in the forward pass, so this will simply
    return the loss from the forward pass.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, model_output: AttrDict, Y_trgt: None) -> th.Tensor:
        return model_output[1].loss


class ClassifyingRegressionLoss(nn.Module):
    """
    Computes the loss that is the classification of the regression model.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, model_output, graph_trgt) -> th.Tensor:
        return self.loss(model_output, graph_trgt)


class ClassifyingRegressionAndELBOLoss(nn.Module):
    """
    Computes the loss that is the classification of the regression model.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.classification_loss = nn.BCEWithLogitsLoss()
        self.neural_process_loss = ELBOLossLNPFInterventional()

    def forward(
        self,
        pred_outputs,
        target,
    ) -> th.Tensor:
        p_yCc, p_yCc_int, _, q_zCc, q_zCct, class_logits = pred_outputs
        Y_trgt = target["Y_trgt"]
        Y_trgt_int = target["Y_trgt_int"]
        graph_trgt = target["graph_trgt"]
        np_loss = self.neural_process_loss(
            (p_yCc, p_yCc_int, None, q_zCc, q_zCct),
            dict(Y_trgt=Y_trgt, Y_trgt_int=Y_trgt_int),
        )
        class_loss = self.classification_loss(class_logits, graph_trgt)
        return np_loss + class_loss


class BaseLossNPFInterventionalLocalLatent(nn.Module, abc.ABC):
    """
    Compute the negative log likelihood loss for members of the conditional neural process (sub-)family.
    This base class makes sure that losses for interventional samples can be computed.

    Parameters
    ----------
    reduction : {None,"mean","sum"}, optional
        Batch wise reduction.

    is_force_mle_eval : bool, optional
        Whether to force mac likelihood eval even if has access to q_zCct
    """

    def __init__(self, reduction="mean", is_force_mle_eval=True):
        super().__init__()
        self.reduction = reduction
        self.is_force_mle_eval = is_force_mle_eval

    def forward(self, pred_outputs, Y_trgt_all):
        """Compute the Neural Process Loss.

        Parameters
        ----------
        pred_outputs : tuple
            Output of `NeuralProcessFamily`.

        Y_trgt : torch.Tensor, size=[batch_size, *n_trgt, y_dim]
            Set of all target values {y_t}.

        Return
        ------
        loss : torch.Tensor
            size=[batch_size] if `reduction=None` else [1].
        """
        (
            p_yCc, p_yCc_int,
            z_samples, q_zCc, q_zCct,
            z_samples_local_obs, q_z_local_prior_obs, q_z_local_t_obs,
            z_samples_local_int, q_z_local_prior_int, q_z_local_t_int,
        ) = pred_outputs
        Y_trgt = Y_trgt_all["Y_trgt"]
        if "Y_trgt_int" in Y_trgt_all.keys():
            Y_trgt_int = Y_trgt_all["Y_trgt_int"]
        else:
            Y_trgt_int = None

        if self.training:
            loss = self.get_loss(
                p_yCc, p_yCc_int,
                z_samples, q_zCc, q_zCct,
                z_samples_local_obs, q_z_local_prior_obs, q_z_local_t_obs,
                z_samples_local_int, q_z_local_prior_int, q_z_local_t_int,
                Y_trgt, Y_trgt_int
            )
        else:
            # always uses NPML for evaluation
            if self.is_force_mle_eval:
                q_zCct = None
            loss = NLLLossLocalLNPFInterventional.get_loss(
                self, p_yCc, p_yCc_int,
                z_samples, q_zCc, q_zCct,
                z_samples_local_obs, q_z_local_prior_obs, q_z_local_t_obs,
                z_samples_local_int, q_z_local_prior_int, q_z_local_t_int,
                Y_trgt, Y_trgt_int
            )

        if self.reduction is None:
            # size = [batch_size]
            return loss
        elif self.reduction == "mean":
            # size = [1]
            return loss.mean(0)
        elif self.reduction == "sum":
            # size = [1]
            return loss.sum(0)
        else:
            raise ValueError(f"Unknown {self.reduction}")

    @abc.abstractmethod
    def get_loss(
        self, p_yCc, p_yCc_int,
        _, q_zCc, q_zCct,
        z_local_samples_obs, q_z_local_obs_prior, q_z_local_obs,
        z_local_samples_int, q_z_local_int_prior, q_z_local_int,
        Y_trgt, Y_trgt_int
    ):
        """Compute the Neural Process Loss

        Parameters
        ------
        p_yCc: torch.distributions.Distribution, batch shape=[n_z_samples, batch_size, *n_trgt] ; event shape=[y_dim]
            Posterior distribution for target values {p(Y^t|y_c; x_c, x_t)}_t

        p_yCc_int: torch.distributions.Distribution, batch shape=[n_z_samples, batch_size, *n_trgt] ; event shape=[y_dim]
            Posterior distribution for interventional target values {p(Y^t|y_c; x_c, do(x_t))}_t

        z_samples: torch.Tensor, size=[n_z_samples, batch_size, *n_lat, z_dim]
            Sampled latents. `None` if `encoded_path==deterministic`.

        q_zCc: torch.distributions.Distribution, batch shape=[batch_size, *n_lat] ; event shape=[z_dim]
            Latent distribution for the context points. `None` if `encoded_path==deterministic`.

        q_zCct: torch.distributions.Distribution, batch shape=[batch_size, *n_lat] ; event shape=[z_dim]
            Latent distribution for the targets. `None` if `encoded_path==deterministic`
            or not training or not `is_q_zCct`.

        Y_trgt: torch.Tensor, size=[batch_size, *n_trgt, y_dim]
            Set of all target values {y_t}.

        Y_trgt_int: torch.Tensor, size=[batch_size, *n_trgt, y_dim]
            Set of all target values {y_t | do(x_t)}.

        Return
        ------
        loss : torch.Tensor, size=[1].
        """
        pass



class ELBOLossLNPFInterventionalLocalLatent(BaseLossNPFInterventionalLocalLatent):

    def get_loss(
        self, p_yCc, p_yCc_int,
        _, q_zCc, q_zCct,
        z_local_samples_obs, q_z_local_obs_prior, q_z_local_obs,
        z_local_samples_int, q_z_local_int_prior, q_z_local_int,
        Y_trgt, Y_trgt_int
    ):
        """
        Right now we only allow for a single local latent sample. This means that
        there is no need to mean over the local latent samples.
        """
        # first term in loss is E_{q(z|y_cntxt,y_trgt)}[\sum_t log p(y^t|z)]
        # \sum_t log p(y^t|z). size = [localz_samples, z_samples, batch_size, n_trgt]
        p_obs_logprob = p_yCc.log_prob(Y_trgt)
        p_int_logprob = p_yCc_int.log_prob(Y_trgt_int)

        # Mean over the local latent samples
        # size = [z_samples, batch_size, n_trgt]
        E_zi_p_obs_logprob = p_obs_logprob.mean(0)
        E_zi_p_int_logprob = p_int_logprob.mean(0)

        # Sum over all the target points
        # size = [z_samples, batch_size]
        sum_E_zi_p_obs_logprob = sum_from_nth_dim(E_zi_p_obs_logprob, 2)
        sum_E_zi_p_int_logprob = sum_from_nth_dim(E_zi_p_int_logprob, 2)

        # E_{q(z|y_cntxt,y_trgt)}[...] . size = [batch_size]
        E_z_sum_log_p_yCz = sum_E_zi_p_obs_logprob.mean(0)
        E_z_sum_log_p_yCz_int = sum_E_zi_p_int_logprob.mean(0)

        # second term in loss is \sum_l KL[q(z^l|y_cntxt,y_trgt)||q(z^l|y_cntxt)]
        # KL[q(z^l|y_cntxt,y_trgt)||q(z^l|y_cntxt)]. size = [batch_size, *n_lat]
        kl_z_global = kl_divergence(q_zCct, q_zCc)
        # \sum_l ... . size = [batch_size]
        E_z_kl_global = sum_from_nth_dim(kl_z_global, 1)

        # Local KL term
        kl_z_obs_local = kl_divergence(q_z_local_obs, q_z_local_obs_prior)
        kl_z_int_local = kl_divergence(q_z_local_int, q_z_local_int_prior)
        # sum over the points
        all_kl_local_obs = kl_z_obs_local.sum(2)
        all_kl_local_int = kl_z_int_local.sum(2)
        # mean over z_samples
        # size = [batch_size]
        E_z_kl_local = all_kl_local_obs.mean(0) + all_kl_local_int.mean(0)
        return -(E_z_sum_log_p_yCz + E_z_sum_log_p_yCz_int - 2 * E_z_kl_global - E_z_kl_local)


class NLLLossLocalLNPFInterventional(BaseLossNPFInterventionalLocalLatent):
    """
    Compute the approximate negative log likelihood for Neural Process family[?].

     Notes
    -----
    - might be high variance
    - biased
    - approximate because expectation over q(z|cntxt) instead of p(z|cntxt)
    - if q_zCct is not None then uses importance sampling (i.e. assumes that sampled from it).

    References
    ----------
    [?]
    """

    def get_loss(
        self, p_yCc, p_yCc_int,
        z_samples, q_zCc, q_zCct,
        z_local_samples_obs, q_z_local_obs_prior, q_z_local_obs,
        z_local_samples_int, q_z_local_int_prior, q_z_local_int,
        Y_trgt, Y_trgt_int,
    ):
        n_z_samples, batch_size, *n_trgt = p_yCc.batch_shape

        log_p_yCz_obs = p_yCc.log_prob(Y_trgt)
        log_p_yCz_int = p_yCc_int.log_prob(Y_trgt_int)

        # mean over the local latent samples
        # size = [z_samples, batch_size, n_trgt]
        E_local_zi_p_obs_logprob = log_p_yCz_obs.mean(0)
        E_local_zi_p_int_logprob = log_p_yCz_int.mean(0)

        # \sum_t log p(y^t|z). size = [z_samples, batch_size]
        sum_log_p_yCz_obs = sum_from_nth_dim(E_local_zi_p_obs_logprob, 2)
        sum_log_p_yCz_int = sum_from_nth_dim(E_local_zi_p_int_logprob, 2)

        sum_log_w_k = sum_log_p_yCz_obs + sum_log_p_yCz_int

        if q_zCct is not None:
            raise NotImplementedError("Importance sampling not implemented for local latent variables")

        # log_sum_exp_z ... . size = [batch_size]
        log_S_z_sum_p_yCz = th.logsumexp(sum_log_w_k, 0)

        # - log(n_z_samples)
        log_E_z_sum_p_yCz = log_S_z_sum_p_yCz - math.log(n_z_samples)

        # NEGATIVE log likelihood
        return -log_E_z_sum_p_yCz



def locallatent_val_loss(
    p_yCc, Y_trgt,
):
        log_p_yCz_obs = p_yCc.log_prob(Y_trgt)
        # mean over the local latent samples
        # size = [z_samples, batch_size, n_trgt]
        if log_p_yCz_obs.ndim == 4:
            n_z_samples = log_p_yCz_obs.shape[1]
            E_local_zi_p_obs_logprob = log_p_yCz_obs.mean(0)
        else:
            n_z_samples = log_p_yCz_obs.shape[0]
            E_local_zi_p_obs_logprob = log_p_yCz_obs
        # sum over the points
        # size = [z_samples, batch_size]
        sum_log_p_yCz_obs = sum_from_nth_dim(E_local_zi_p_obs_logprob, 2)

        sum_log_w_k = sum_log_p_yCz_obs

        # log_sum_exp_z ... . size = [batch_size]
        log_S_z_sum_p_yCz = th.logsumexp(sum_log_w_k, 0)

        # - log(n_z_samples)
        log_E_z_sum_p_yCz = log_S_z_sum_p_yCz - math.log(n_z_samples)
        # NEGATIVE log likelihood
        return -log_E_z_sum_p_yCz


class BaseLossClassifier(nn.Module, abc.ABC):
    """
    Computes the log likelihood of the classifier.

    Parameters
    ----------
    reduction : {None,"mean","sum"}, optional
        Batch wise reduction.

    is_force_mle_eval : bool, optional
        Whether to force mac likelihood eval even if has access to q_zCct
    """

    def __init__(self, reduction="mean", is_force_mle_eval=True):
        super().__init__()
        self.reduction = reduction
        self.is_force_mle_eval = is_force_mle_eval

    def forward(self, pred, trgt):
        """Compute the Neural Process Loss.

        Parameters
        ----------
        pred : tuple
            Output of `NeuralProcessFamily`.

        trgt : torch.Tensor, size=[batch_size,]
            Set of classification target values.

        Return
        ------
        loss : torch.Tensor
            size=[batch_size] if `reduction=None` else [1].
        """
        p_trgt = pred
        trgt = trgt["trgt"]

        # Squeeze if there is an extra dimension
        if p_trgt.ndim == 2:
            p_trgt = p_trgt.squeeze(-1)
        if trgt.ndim == 2:
            trgt = trgt.squeeze(-1)

        if self.training:
            loss = self.get_loss(
                p_trgt, trgt
            )
        else:
            loss = self.get_loss(
                p_trgt, trgt
            )

        if self.reduction is None:
            # size = [batch_size]
            return loss
        elif self.reduction == "mean":
            # size = [1]
            return loss.mean(0)
        elif self.reduction == "sum":
            # size = [1]
            return loss.sum(0)
        else:
            raise ValueError(f"Unknown {self.reduction}")


class ConvCNPClassifierLoss(BaseLossClassifier):

    def get_loss(self, pred, trgt):
        loss = nn.BCEWithLogitsLoss()
        return loss(pred, trgt)