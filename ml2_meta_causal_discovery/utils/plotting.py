"""
Helpers for plotting predictions from Neural Processes.

Taken heavily from NPF.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

# from ml2_meta_causal_discovery.models.convcnp import (
#     InterventionalLatentNeuralProcessFamily,
# )
from ml2_meta_causal_discovery.utils.helpers import set_seed
from ml2_meta_causal_discovery.utils.autoregeressive_helper import (
    get_next_data_dict,
)
from skorch.utils import to_tensor
from tqdm import trange
from attrdict import AttrDict

DFLT_FIGSIZE = (11, 11)


def gen_p_y_pred_transformer(
    model, X_cntxt, Y_cntxt, X_trgt, n_samples, **kwargs
):
    inputs = AttrDict(dict(xc=X_cntxt, yc=Y_cntxt, xt=X_trgt))
    out_dist = model.predict(**inputs)
    samples = out_dist.sample(torch.Size([n_samples]))
    for i in range(samples.shape[0]):
        yield samples[i, 0, :, 0].flatten(), None, X_trgt[0, :, 0].flatten()


def gen_p_y_pred(
    model,
    X_cntxt,
    Y_cntxt,
    X_trgt,
    n_samples,
    autoregressive=True,
    global_latent=False,
):
    """Get the estimated (conditional) posterior predictive from a model."""
    if X_cntxt is None:
        X_cntxt = torch.zeros(1, 0, model.x_dim)
        Y_cntxt = torch.zeros(1, 0, model.y_dim)

    if autoregressive and global_latent:
        raise ValueError("Can't have both autoregressive and global latent")

    if global_latent:
        # Set new number of samples for test time
        old_n_z_samples_test = model.n_z_samples_test
        model.n_z_samples_test = n_samples
        p_yCc, *_ = model.forward(X_cntxt, Y_cntxt, X_trgt)
        model.n_z_samples_test = old_n_z_samples_test
        y_pred_sample = p_yCc.sample()[:, 0, :, :]
    elif autoregressive:
        num_target = X_trgt.shape[1]
        Xi = dict(X_cntxt=X_cntxt, Y_cntxt=Y_cntxt, X_trgt=X_trgt)
        # need to tile Y_cnxt
        Y_cntxt_new = torch.tile(Xi["Y_cntxt"], (n_samples, 1, 1))
        X_cntxt_new = torch.tile(Xi["X_cntxt"], (n_samples, 1, 1))
        X_trgt_new = torch.tile(Xi["X_trgt"], (n_samples, 1, 1))
        # Shuffle the target points
        # Generate random indices
        indices = torch.argsort(torch.rand_like(X_trgt_new), dim=1)
        X_trgt_new = torch.gather(X_trgt_new, dim=1, index=indices)
        Xi["Y_cntxt"] = Y_cntxt_new
        Xi["X_cntxt"] = X_cntxt_new
        Xi["X_trgt"] = X_trgt_new
        all_dist = []
        for i in trange(
            num_target,
            total=num_target,
            desc="Generating predictions",
            position=0,
            leave=True,
        ):
            if i == 0:
                new_Xi = get_next_data_dict(Xi, i, torch=True)
                y_pred_full = model.forward(**new_Xi)
                y_pred_dist = y_pred_full[0]
                y_pred_sample = y_pred_dist.sample()[0, :, :, :]
            else:
                new_Xi = get_next_data_dict(
                    Xi, i, new_Xi, y_pred_sample, torch=True
                )
                y_pred_full = model.forward(**new_Xi)
                y_pred_dist = y_pred_full[0]
                y_pred_sample = y_pred_full[0].sample()[0, :, :, :]
            all_dist.append(y_pred_dist)
        new_Xi = get_next_data_dict(Xi, i, new_Xi, y_pred_sample, torch=True)
    else:
        p_yCc, *_ = model.forward(X_cntxt, Y_cntxt, X_trgt)
        y_pred_sample = p_yCc.sample(torch.Size([n_samples]))[:, 0, 0, :, :]
        # p_yCc, *_ = model.forward(X_cntxt, Y_cntxt, X_trgt)

    if autoregressive:
        # shape (n_samples, batch_size, n_targets, y_dim)
        all_means = [dist.base_dist.loc.detach() for dist in all_dist]
        all_stds = [dist.base_dist.scale.detach() for dist in all_dist]
        mean_ys = new_Xi["Y_cntxt"][:, X_cntxt.shape[1] :, :]
        std_ys = torch.stack(all_stds, dim=1)[None]
        for i in range(mean_ys.shape[0]):
            yield mean_ys[i, :, 0].flatten(), None, X_trgt_new[
                i, :, 0
            ].flatten()
    else:
        # mean_ys = p_yCc.base_dist.loc.detach().numpy()
        # std_ys = p_yCc.base_dist.scale.detach().numpy()

        for i in range(y_pred_sample.shape[0]):
            yield y_pred_sample[i, :, 0].flatten(), None, X_trgt[
                0, :, 0
            ].flatten()


def plot_posterior_predefined_cntxt(
    model,
    X_cntxt,
    Y_cntxt,
    X_trgt,
    Y_trgt,
    n_samples=20,
    is_plot_std=False,
    train_min_max=(-2, 2),
    model_label="Model",
    scatter_label=None,
    alpha_init=1,
    mean_std_colors=("b", "tab:blue"),
    title=None,
    figsize=DFLT_FIGSIZE,
    ax=None,
    is_legend=True,
    scatter_kwargs={},
    kwargs_std={},
    autoregressive=True,
    global_latent=False,
    transformer=True,
    **kwargs,
):
    """
    Plot (samples from) the conditional posterior predictive estimated by a model.

    Parameters
    ----------
    model : nn.Module

    X_cntxt: torch.Tensor, size=[1, n_cntxt, x_dim]
        Set of all context features {x_i}.

    Y_cntxt: torch.Tensor, size=[1, n_cntxt, y_dim]
        Set of all context values {y_i}.

    X_trgt: torch.Tensor, size=[1, n_trgt, x_dim]
        Set of all target features {x_t}.

    Y_trgt: torch.Tensor, size=[1, n_trgt, y_dim], optional
        Set of all target values {y_t}. If not `None` plots the underlying function.

    n_samples : int, optional
        Number of samples from the posterior.

    is_plot_std : bool, optional
        Wheter to plot the predicted standard deviation.

    train_min_max : tuple of float, optional
        Min and maximum boundary used during training. Important to unscale X to
        its actual values (i.e. plot will not be in -1,1).

    alpha_init : float, optional
        Transparency level to use.

    mean_std_colors : tuple of str, optional
        Color of the predicted mean and std for plotting.

    model_label : str, optional
        Name of the model for the legend.

    title : str, optional

    figsize : tuple, optional

    ax : plt.axes.Axes, optional

    is_smooth : bool, optional
        Whether to plot a smooth function instead of a scatter plot.

    is_legend : bool, optional
        Whether to add a legend.

    scatter_kwargs : dict, optional
        Kwargs for the scatter function.

    kwargs_std : dict, optional
        Kwargs for plot std function (`fil_between` if smooth else `errorbar`).

    kwargs :
        Additional
    """
    mean_color, std_color = mean_std_colors
    mean_color = "tab:red"

    # plot posterior instead prior ?
    is_conditioned = X_cntxt is not None and X_cntxt.shape[1] >= 1

    model.eval()
    num_context = X_cntxt.shape[1]

    X_trgt_plot = X_trgt.cpu().numpy()[0].flatten()

    x_min = min(X_trgt_plot)
    x_max = max(X_trgt_plot)

    if is_conditioned:
        X_cntxt_plot = X_cntxt.cpu().numpy()[0].flatten()

    # make alpha dependent on number of samples
    alpha = alpha_init / (n_samples) ** 0.5

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        y_min, y_max = 0, 0
    else:
        y_min, y_max = ax.get_ylim()

    predict_func = gen_p_y_pred if not transformer else gen_p_y_pred_transformer

    for i, (mean_y, std_y, X_trgt_plot) in enumerate(
        predict_func(
            model,
            X_cntxt,
            Y_cntxt,
            X_trgt,
            n_samples,
            autoregressive=autoregressive,
            global_latent=global_latent,
        )
    ):
        X_trgt_plot = X_trgt_plot.cpu().numpy()
        mean_y = mean_y.cpu().numpy()
        if i == 0:
            # only add single label
            # if autoregressive:
            ax.scatter(
                X_trgt_plot,
                mean_y,
                alpha=alpha,
                c=mean_color,
                label=f"{model_label}",
                **kwargs,
            )
            # else:
            #     ax.plot(
            #         X_trgt_plot,
            #         mean_y,
            #         alpha=alpha,
            #         c=mean_color,
            #         label=f"{model_label}",
            #         **kwargs,
            #     )
        else:
            ax.scatter(X_trgt_plot, mean_y, alpha=alpha, c=mean_color, **kwargs)

        if is_plot_std:
            if std_y is None:
                raise ValueError(
                    f"Cannot plot std when sampling (n_samples={n_samples}) from a CNPF."
                )

            kwargs_new = dict(alpha=alpha / 7)
            kwargs_new.update(kwargs_std)
            # only when smooth, if not already plotted error bars
            ax.fill_between(
                X_trgt_plot,
                mean_y - 2 * std_y,
                mean_y + 2 * std_y,
                color=std_color,
                **kwargs_new,
            )
            y_min = min(y_min, (mean_y - std_y).min())
            y_max = max(y_max, (mean_y + std_y).max())

    if Y_trgt is not None:
        X_trgt = X_trgt.cpu().numpy()[0].flatten()
        Y_trgt = Y_trgt.cpu().numpy()[0, :, 0].flatten()
        ax.scatter(X_trgt, Y_trgt, alpha=0.7, label="Target Function")
        y_min = min(y_min, Y_trgt.min())
        y_max = max(y_max, Y_trgt.max())

    if is_conditioned:
        if scatter_label is not None:
            scatter_kwargs["label"] = scatter_label
        ax.scatter(
            X_cntxt_plot,
            Y_cntxt[0, :, 0].cpu().numpy(),
            c="k",
            **scatter_kwargs,
        )
        x_min = min(min(X_cntxt_plot), x_min)
        x_max = max(max(X_cntxt_plot), x_max)

    ax.set_xlim(x_min, x_max)

    # extrapolation might give huge values => rescale to have y lim as interpolation
    ax.set_ylim(y_min, y_max)

    if title is not None:
        ax.set_title(title, fontsize=14)

    if is_legend:
        ax.legend()

    return ax


def plot_posterior_samples_1d(
    model,
    X_cntxt,
    Y_cntxt,
    X_trgt,
    Y_trgt=None,
    model_labels=dict(main="Model", compare="Compare", generator="Oracle GP"),
    is_plot_real=True,
    train_min_max=(-2, 2),
    ax=None,
    seed=None,
    is_fill_generator_std=True,
    y_lim=(None, None),
    is_legend=True,
    autoregressive=True,
    plot_config_kwargs={},
    transformer=False,
    **kwargs,
):
    """
    Plot and compare (samples from) the conditional posterior predictive estimated
    by some models at random cntxt and target points.

    Parameters
    ----------
    X : torch.tensor, size=[1, n_trgt, x_dim]
        All features X, should be rescaled shuch that interpolation is in (-1,1).

    Y : torch.tensor, size=[1, n_trgt, x_dim]
        Actual values Y for all X.

    get_cntxt_trgt : callable
        Function that takes as input the features and tagrets `X`, `y` and return
        the corresponding `X_cntxt, Y_cntxt, X_trgt, Y_trgt`.

    model : nn.Module
        Main prediction model.

    compare_model : nn.Module, optional
        Secondary prediction model used for comparaisons.

    model_labels : dict, optional
        Name of the `main`, `compare`, `generator` model.

    generator : sklearn.estimator, optional
        Underlying generator. If not `None` will plot its own predictions.

    is_plot_real : bool, optional
        Whether to plot the underlying `Y_trgt`.

    train_min_max : tuple of float, optional
        Min and maximum boundary used during training. Important to unscale X to
        its actual values (i.e. plot will not be in -1,1).

    is_fill_generator_std : bool, optional
        Whether to show the generator's std filled in or simply the outline.

    y_lim : tuple of int, optional
        Min max y limit. If one is None, then auto.

    is_legend : bool, optional
        Whether to add a legend.

    plot_config_kwargs : dict, optional
        Other arguments to `plot_config_kwargs`.

    kwargs :
        Additional arguments to `_plot_posterior_predefined_cntxt`.
    """
    set_seed(seed)

    alpha_init = 1
    ax = plot_posterior_predefined_cntxt(
        model,
        X_cntxt,
        Y_cntxt,
        X_trgt,
        train_min_max=train_min_max,
        Y_trgt=Y_trgt if is_plot_real else None,
        model_label=model_labels["main"],
        alpha_init=alpha_init,
        mean_std_colors=("b", "tab:blue"),
        ax=ax,
        is_legend=is_legend,
        autoregressive=autoregressive,
        transformer=transformer,
        **kwargs,
    )

    # TODO: Think about plotting the generator, this might require fitting a
    # GPLVM so might be too much for now
    return ax


def plot_multi_posterior_samples_1d(
    trainer,
    dataset,
    n_cntxt,
    plot_config_kwargs={},
    title="Model : {model_name} | Data : {data_name} | Num. Context : {n_cntxt}",
    imgsize=(8, 3),
    seed=None,
    autoregressive=True,
    transformer=False,
    **kwargs,
):
    """Plot posterior samples conditioned on `n_cntxt` context points for a set of trained trainers."""
    set_seed(seed)

    n_trainers = 1
    n_col = 1
    fig, axes = plt.subplots(
        n_trainers,
        n_col,
        figsize=(imgsize[0] * n_col, imgsize[1] * n_trainers),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    curr_title = None

    XY_cntxt, XY_trgt, XY_trgt_int, causal_graph, unique_trgt_indices = next(
        dataset.generate_next_dataset(n_context=n_cntxt)
    )
    XY_trgt = XY_trgt[:, unique_trgt_indices, :]

    ind_sort = np.argsort(XY_trgt[0, :, 0])
    XY_trgt = XY_trgt[:, ind_sort, :]

    plot_posterior_samples_1d(
        model=trainer.module_,
        X_cntxt=torch.from_numpy(XY_cntxt[:, :, 0][:, :, None]),
        Y_cntxt=torch.from_numpy(XY_cntxt[:, :, 1][:, :, None]),
        X_trgt=torch.from_numpy(XY_trgt[:, :, 0][:, :, None]),
        Y_trgt=torch.from_numpy(XY_trgt[:, :, 1][:, :, None]),
        title=curr_title,
        ax=axes[0, 0],
        scatter_label="Context Set",
        autoregressive=autoregressive,
        transformer=transformer,
        **kwargs,
    )

    plt.tight_layout()

    return fig
