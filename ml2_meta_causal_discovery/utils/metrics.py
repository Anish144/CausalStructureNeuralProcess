from sklearn.metrics import roc_auc_score
from tqdm import trange
import numpy as np
from sklearn.metrics import f1_score
import torch as th


def cyclicity(A):
    """
    Code adapted from DiBS:
    Differentiable acyclicity constraint from Yu et al. (2019). If h = 0 then the graph is acyclic.
    http://proceedings.mlr.press/v97/yu19a/yu19a.pdf

    Args:
        mat (ndarray): graph adjacency matrix of shape ``[n_vars, n_vars]``
        n_vars (int): number of variables, to allow for ``jax.jit``-compilation

    Returns:
        bool: True if the graph is cyclic, False otherwise
    """
    n_vars = A.shape[0]
    # alpha = 1.0 / n_vars

    # M = tf.add(tf.eye(n_vars, dtype=default_float()), alpha * A)

    M_mult = th.linalg.matrix_exp(A)

    h = th.trace(M_mult) - n_vars

    return h


def balance_for_auc(target, pred_scores):
    # Targets are {-1, 1}, need to make sure it sums to zero
    balance = int(np.sum(target))
    if balance != 0:
        # There are more negative examples
        if balance < 0:
            switch_cand_idx = np.nonzero(target < 0)[0]
        # There are more positive examples here
        else:
            switch_cand_idx = np.nonzero(target > 0)[0]
        # get "balance" number of indices
        switch_idx = np.random.choice(
            switch_cand_idx, size=int(np.abs(balance) // 2), replace=False
        )
        final_target = target.copy()
        final_target[switch_idx] *= -1
        final_pred_scores = pred_scores.copy()
        final_pred_scores[switch_idx] *= -1
    else:
        final_target = target.copy()
        final_pred_scores = pred_scores.copy()
    if (balance % 2) == 0:
        assert np.sum(final_target) == 0
    else:
        assert np.abs(np.sum(final_target)) == 1
    return final_target, final_pred_scores


def calculate_auc(target, pred_scores, num_shuffles=1000):
    # Need to make sure that the classes are evenly balanced
    auc_all = []
    for i in trange(num_shuffles):
        total_runs = len(target)
        flip_idx = np.random.choice(np.arange(total_runs), total_runs // 2, replace=False)
        for i in range(total_runs):
            if i in flip_idx:
                target[i] *= -1
                pred_scores[i] *= -1
        final_target, final_pred_scores = balance_for_auc(target, pred_scores)
        roc_auc = roc_auc_score(final_target, final_pred_scores)
        auc_all.append(roc_auc)
    return np.mean(auc_all)


def calc_SHD(target, pred, double_for_anticausal=True):
    r"""Compute the Structural Hamming Distance.

    The Structural Hamming Distance (SHD) is a standard distance to compare
    graphs by their adjacency matrix. It consists in computing the difference
    between the two (binary) adjacency matrixes: every edge that is either
    missing or not in the target graph is counted as a mistake. Note that
    for directed graph, two mistakes can be counted as the edge in the wrong
    direction is false and the edge in the good direction is missing ; the
    `double_for_anticausal` argument accounts for this remark. Setting it to
    `False` will count this as a single mistake.

    Args:
        target (numpy.ndarray or networkx.DiGraph): Target graph, must be of
            ones and zeros.
        prediction (numpy.ndarray or networkx.DiGraph): Prediction made by the
            algorithm to evaluate.
        double_for_anticausal (bool): Count the badly oriented edges as two
            mistakes. Default: True

    Returns:
        int: Structural Hamming Distance (int).

            The value tends to zero as the graphs tend to be identical.

    Examples:
        >>> from cdt.metrics import SHD
        >>> from numpy.random import randint
        >>> tar, pred = randint(2, size=(10, 10)), randint(2, size=(10, 10))
        >>> SHD(tar, pred, double_for_anticausal=False)
    """
    diff = np.abs(target - pred)
    return np.sum(diff)


def expected_shd(target, pred, check_acyclic=False):
    """
    Expected SHD for a batch of predictions

    Args:
    -----
    target: np.ndarray, shape (batch_size, num_nodes, num_nodes)
        The target graph
    pred: np.ndarray, shape (num_samples, batch_size, num_nodes, num_nodes)
    """
    shd_all = np.zeros(pred.shape[1])
    for i in range(pred.shape[1]):
        # Select batch
        curr_pred = pred[:, i]
        curr_target = target[i]
        # Loop over samples
        shd_sample = []
        for j in range(curr_pred.shape[0]):
            curr_sample_pred = curr_pred[j]
            if check_acyclic:
                if cyclicity(curr_sample_pred) != 0:
                    continue
                else:
                    pass
            shd = calc_SHD(curr_target, curr_sample_pred, double_for_anticausal=True)
            shd_sample.append(shd)
        shd_all[i] = np.mean(shd_sample)
    return shd_all

def expected_f1_score(target, pred, check_acyclic=False):
    """
    Expected F1 score for a batch of predictions

    Args:
    -----
    target: np.ndarray, shape (batch_size, num_nodes, num_nodes)
        The target graph
    pred: np.ndarray, shape (num_samples, batch_size, num_nodes, num_nodes)
    """
    f1_all = np.zeros(pred.shape[1])
    for i in range(pred.shape[1]):
        # Select batch
        curr_pred = pred[:, i]
        curr_target = target[i]
        # Loop over samples
        f1_sample = []
        for j in range(curr_pred.shape[0]):
            curr_sample_pred = curr_pred[j]
            if check_acyclic:
                if cyclicity(curr_sample_pred) != 0:
                    continue
                else:
                    pass
            f1 = f1_score(
                curr_target.flatten(),
                curr_sample_pred.flatten(),
                average="binary",
                zero_division=0,
            )
            f1_sample.append(f1)
        f1_all[i] = np.mean(f1_sample)
    return f1_all


def log_prob_graph_scores(targets, preds):
    """
    Get log prob of Bernoulli score for a batch of predictions

    Args:
    -----
    targets: torch.Tensor, shape (batch_size, num_nodes, num_nodes)
        The target graph.
    preds: torch.Tensor, shape (num_samples, batch_size, num_nodes, num_nodes)
    """
    all_log_probs = []
    for batch_idx in range(targets.shape[0]):
        # Take mean across the samples
        # Shape (num_nodes, num_nodes)
        sample_mean = th.mean(preds[:, batch_idx], axis=0)
        # Shape (num_nodes ** 2)
        sample_mean_flatten = sample_mean.flatten()
        # Compute log prob of Bernoulli
        current_batch = targets[batch_idx]
        current_batch_flatten = current_batch.flatten()

        bern_dist = th.distributions.bernoulli.Bernoulli(
            probs=sample_mean_flatten,
        )
        # Sum log probs over edges
        log_prob = bern_dist.log_prob(current_batch_flatten).sum()
        all_log_probs.append(log_prob.cpu().item())
    return all_log_probs


def auc_graph_scores(targets, preds):
    """
    Get mean auc score for a batch of predictions

    Args:
    -----
    targets: torch.Tensor, shape (batch_size, num_nodes, num_nodes)
        The target graph.
    preds: torch.Tensor, shape (num_samples, batch_size, num_nodes, num_nodes)
    """
    if isinstance(targets, th.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(preds, th.Tensor):
        preds = preds.cpu().numpy()

    all_aucs = []
    for batch_idx in range(targets.shape[0]):
        # Take mean across the samples to get the probs
        # Shape (num_nodes, num_nodes)
        sample_mean = np.mean(preds[:, batch_idx], axis=0)
        # Shape (num_nodes ** 2)
        sample_mean_flatten = sample_mean.flatten()
        # Compute AUC
        current_batch = targets[batch_idx]
        current_batch_flatten = current_batch.flatten()
        auc = roc_auc_score(current_batch_flatten, sample_mean_flatten, average="macro")
        all_aucs.append(auc)
    return all_aucs

