"""
The following file will contain class of functions that will generate the data
given a causal graph.

All classses will contain the generate_data method, which will generate the
data.

Contains:
- GPLVMFunctionGenerator: Data is generated using a GPLVM.
"""
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from collections import defaultdict
import dill
import numpy as np
import random
import traceback
from copy import deepcopy

from ml2_meta_causal_discovery.utils.gplvm_utils import (
    sample_kernel,
    sample_lengthscale,
    sample_likelihood_variance,
    sample_normal_latent,
    sample_sum_kernels,
    sample_variance,
)
from ml2_meta_causal_discovery.utils.processing import (
    normalise_variable,
    rescale_variable,
)
import gpflow
from gpflow.config import default_float
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Optional


class GPLVMFunctions:
    """Helper class to sample from GPLVM."""

    def __init__(
        self,
        mean: str,
        kernel: gpflow.kernels.Kernel,
        likelihood_variance: float,
    ):
        self.mean = mean
        self.kernel = kernel
        self.likelihood_variance = likelihood_variance

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Samples from GP given kernel and mean.
        """
        kernel_instantiated = self.kernel.K(inputs, inputs)
        identity = tf.eye(inputs.shape[0], dtype=default_float())
        cov = kernel_instantiated + identity * self.likelihood_variance
        if self.mean == "zero":
            mean_func = tf.zeros(inputs.shape[0], dtype=default_float())
        elif self.mean == "latent":
            mean_func = inputs[:, -1]
        else:
            raise NotImplementedError(
                f"Mean function {self.mean} not implemented."
            )
        try:
            scale_tril = tf.linalg.cholesky(cov)
        except Exception as e:
            cov = cov + identity * 1e-4
            scale_tril = tf.linalg.cholesky(cov)
        normal_dist = tfp.distributions.MultivariateNormalTriL(
            loc=mean_func, scale_tril=scale_tril
        )
        output = normal_dist.sample()
        assert output.shape == (inputs.shape[0],), f"Shape is {output.shape}!"
        return output


class GPFunctions(GPLVMFunctions):
    """Helper class to sample from GPs."""

    def __init__(
        self,
        mean: str,
        kernel,
        likelihood_variance: float,
    ):
        super().__init__(
            mean=mean,
            kernel=kernel,
            likelihood_variance=likelihood_variance,
        )

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Samples from GP given kernel and mean.
        """
        input_num = inputs.shape[-1]
        if input_num == 1:
            inputs = np.random.normal(loc=0, scale=1.0, size=inputs.shape)
        else:
            inputs = inputs[:, :-1]
        outputs = super().__call__(inputs)
        return outputs


class DataGenerator(ABC):
    """
    Base class for all causal data generators.
    """

    def __init__(
        self,
        num_variables: int,
        num_samples: int,
        interventions: bool,
    ):
        self.number_of_variables = num_variables
        self.num_samples = num_samples
        self.interventions = interventions

    def _get_inputs(
        self, parents_of_i: np.ndarray, data: np.ndarray
    ) -> np.ndarray:
        """
        Get the inputs for the variable i.
        """
        parents_of_i = np.where(parents_of_i)[0]
        inputs = data[:, parents_of_i]
        assert inputs.ndim == 2
        return inputs

    def generate_data(
        self,
        causal_graph: np.ndarray,
        num_int_samples: int,
    ) -> np.ndarray:
        """
        Generate functions for the SCM.

        For now, we always intervene on the 0th index variable.

        Args:
        ----------
        causal_graph : np.ndarray shape (num_variables, num_variables)
            Causal graph of the SCM.

        num_int_samples : int
            Number of interventional samples to generate.

        Returns:
        ----------
        permuted_data : np.ndarray shape (num_samples, num_variables)
            Data generated from the causal graph.

        permuted_int_data : np.ndarray shape (num_interventions, num_variables)
            Data with interventions carried out.
        """
        # Functions will be a dict with keys being the variable number
        function_dict = self.generate_functions(causal_graph)
        data = np.zeros((self.num_samples, self.number_of_variables))
        interventional_data = np.zeros(
            (num_int_samples, self.number_of_variables)
        )
        # Causal graph row i is a parent of column j.
        # We always need to generate the cause first.
        # Thus, we need to loop in order of the causal graph.
        if self.number_of_variables == 2:
            if np.sum(causal_graph[1, :]) == 1:
                loop_order = np.arange(self.number_of_variables)[::-1]
            else:
                loop_order = np.arange(self.number_of_variables)
        else:
            loop_order = np.arange(self.number_of_variables)

        if causal_graph[1, 0] == 1:
            import pdb; pdb.set_trace()

        # We need to make sure that the inerventions are samplef from the
        # SAME FUNCTION!
        for i in loop_order:
            function_for_i = function_dict[i]
            parents_of_i = causal_graph[:, i]

            # Observational data
            # Sample latent
            latent = sample_normal_latent(self.num_samples)
            # Inputs will be an empty array if there are no parents.
            inputs = self._get_inputs(parents_of_i, data)
            full_inputs_obs = np.concatenate((inputs, latent), axis=1)

            # Interventional data from the same function as the observational.
            # Sample interventional data
            if i == 0:
                full_inputs = full_inputs_obs
            else:
                if self.interventions:
                    latent_int = sample_normal_latent(num_int_samples)
                    inputs_int = self._get_inputs(
                        parents_of_i, interventional_data
                    )
                    full_inputs_int = np.concatenate(
                        (inputs_int, latent_int), axis=1
                    )
                    # the last [num_int_samples, :] is the interventional data
                    full_inputs = np.concatenate(
                        (full_inputs_obs, full_inputs_int), axis=0
                    )
                    full_inputs = tf.convert_to_tensor(full_inputs)
                else:
                    full_inputs = full_inputs_obs

            # Sometimes hyperparams give badly conditioned cov matrices,
            # This resamples until it works.
            finish = 0
            while finish == 0:
                try:
                    variable = function_for_i(full_inputs)
                    finish = 1
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    function_dict = self.generate_functions(causal_graph)
                    function_for_i = function_dict[i]

            # Sometimes the function can return nan values
            # This resamples until it works.
            while np.isnan(variable).any():
                function_dict = self.generate_functions(causal_graph)
                function_for_i = function_dict[i]
                variable = function_for_i(full_inputs)

            if i == 0:
                # variable_obs = normalise_variable(variable, axis=0)
                variable_obs = variable
                # Intervened on variable does not need to be normalised
                if self.interventions:
                    # variable_int = uniform_interventions(
                    #     num_samples=num_int_samples, range=(tf.math.reduce_min(variable_obs), tf.math.reduce_max(variable_obs))
                    # )
                    variable_int = deepcopy(variable_obs)
            else:
                # variable_normed = normalise_variable(variable, axis=0)
                if self.interventions:
                    variable_obs = variable[: self.num_samples]
                    variable_int = variable[self.num_samples :]
                    var_obs_mean = np.mean(variable_obs)
                    var_obs_std = np.std(variable_obs)
                    variable_obs = normalise_variable(variable_obs, axis=0)
                    variable_int = (variable_int - var_obs_mean) / var_obs_std
                else:
                    # variable_obs = variable_normed
                    variable_obs = variable

            data[:, i] = variable_obs
            if self.interventions:
                interventional_data[:, i] = variable_int
            else:
                pass

        return data, interventional_data

    @abstractmethod
    def generate_functions(
        self,
        causal_graph: np.ndarray,
    ) -> dict:
        """
        Generate functions given a causal graph.

        This will instantiate a class that can then be used to generate data.
        """
        raise NotImplementedError()

    @abstractmethod
    def return_data(self):
        """
        Return the data.
        """
        raise NotImplementedError()


class GPLVMFunctionGenerator(DataGenerator):
    """
    Will generate data using Gaussian Process latent variable model priors
    respecting a given causal graph.

    Args:
    ----------
    num_variables : int
        Number of variables to generate.

    num_samples : int
        Number of samples to generate.

    lengthscale_fixed : bool
        Whether to fix the lengthscale distrbution or draw its parameters from
        another distribution.

    lengthscale_gamma_vals : list
    """

    def __init__(
        self,
        num_variables: int,
        num_samples: int,
        interventions: bool = False,
        kernel_sum: bool = False,
        mean_function: str = "latent",
        device: str = "cpu",
    ):
        super().__init__(num_variables, num_samples, interventions)
        self.kernel_sum = kernel_sum
        self.mean_function = mean_function
        self.device = device

    def return_data(self, causal_graph) -> np.ndarray:
        """Generate the data.

        Args:
        ----------
        causal_graph : np.ndarray shape (num_variables, num_variables)
            Causal graph to use for the data generation.

        Returns:
        ----------
        data : np.ndarray (num_samples, num_variables)
        interventional_data : np.ndarray (num_interventions, num_variables)
        """
        data, interventional_data = self.generate_data(
            causal_graph=causal_graph
        )
        return data, interventional_data

    def generate_functions(
        self,
        causal_graph: np.ndarray,
    ) -> dict:
        """
        Generate functions given a causal graph.

        This will instantiate a class that can then be used to generate data.
        This is necessary as we have to save the functions to generate
        interventional data.
        """
        function_dict = {}
        for i in range(self.number_of_variables):
            parents_of_i = causal_graph[:, i]
            # Plus one for latent variable.
            num_parents = int(np.sum(parents_of_i) + 1)

            # Set kernel
            if not self.kernel_sum:
                variance = sample_variance(1)
                lengthscale = sample_lengthscale(
                    num_parents,
                    fixed_lengthscale=self.lengthscale_fixed,
                    gamma_vals=self.lengthscale_gamma_vals,
                )
                kernel_init = sample_kernel()
                kernel = kernel_init(
                    variance=variance[0],
                    lengthscales=lengthscale,
                )

                linear_variance = sample_variance(1)
                linear_kernel = gpflow.kernels.Linear(variance=linear_variance)
                kernel = gpflow.kernels.Sum([kernel, linear_kernel])
            else:
                kernel = sample_sum_kernels(
                    num_parents=num_parents,
                )

            # Set likelihood noise
            likelihood_variance = sample_likelihood_variance()

            function = GPLVMFunctions(
                mean=self.mean_function,
                kernel=kernel,
                likelihood_variance=likelihood_variance,
            )
            function_dict[i] = function
        return function_dict


class GPFunctionGenerator(DataGenerator):
    """
    Will generate data using Gaussian Process model priors
    respecting a given causal graph.

    Args:
    ----------
    num_variables : int
        Number of variables to generate.

    num_samples : int
        Number of samples to generate.
    """

    def return_data(self, causal_graph) -> np.ndarray:
        """Generate the data.

        Args:
        ----------
        causal_graph : np.ndarray shape (num_variables, num_variables)
            Causal graph to use for the data generation.

        Returns:
        ----------
        data : np.ndarray (num_samples, num_variables)
        interventional_data : np.ndarray (num_interventions, num_variables)
        """
        data, interventional_data = self.generate_data(
            causal_graph=causal_graph
        )
        return data, interventional_data

    def generate_functions(
        self,
        causal_graph: np.ndarray,
    ) -> dict:
        """
        Generate functions given a causal graph.

        This will instantiate a class that can then be used to generate data.
        This is necessary as we have to save the functions to generate
        interventional data.
        """
        function_dict = {}
        for i in range(self.number_of_variables):
            parents_of_i = causal_graph[:, i]
            # We don't add latent variable for GP.
            num_parents = int(np.sum(parents_of_i))

            # Set kernel
            if num_parents > 0:
                variance = sample_variance(1)
                lengthscale = sample_lengthscale(num_parents)
                kernel_init = sample_kernel()
                kernel = kernel_init(
                    variance=variance[0],
                    lengthscales=lengthscale,
                )

                linear_variance = sample_variance(1)
                linear_kernel = gpflow.kernels.Linear(variance=linear_variance)
                kernel = gpflow.kernels.Sum([kernel, linear_kernel])
            else:
                # Sample a simply normal for the cause
                kernel = gpflow.kernels.White(variance=1.0)

            # Set likelihood noise
            likelihood_variance = sample_likelihood_variance()

            function = GPFunctions(
                mean="zero",
                kernel=kernel,
                likelihood_variance=likelihood_variance,
            )
            function_dict[i] = function

        return function_dict


class GPLVMFixedHyperparam(GPLVMFunctionGenerator):
    """
    Generates from a GPLVM but with fixed hyperaparameters that are the result
    of GPLVM experiments.

    Args:
    ----------
    num_variables : int
        Number of variables to generate.

    num_samples : int
        Number of samples to generate.

    lengthscale_fixed : bool
        Whether to fix the lengthscale distrbution or draw its parameters from
        another distribution.

    lengthscale_gamma_vals : list

    sample_hyperparams_collectively : bool
        Whether the hyperparameters should be sampled together or one by one.
    """

    def __init__(
        self,
        num_variables: int,
        num_samples: int,
        interventions: bool = False,
        lengthscale_fixed: bool = False,
        lengthscale_gamma_vals: list = [1.0, 1.0],
        kernel_sum: bool = False,
        mean_function: str = "latent",
        sample_hyperparams_collectively: bool = False,
        sample_hyperparam_index: Optional[int] = None,
    ):
        super().__init__(
            num_variables,
            num_samples,
            interventions,
            lengthscale_fixed,
            lengthscale_gamma_vals,
            kernel_sum,
            mean_function,
        )
        self.sample_hyperparams_collectively = sample_hyperparams_collectively
        self.sample_hyperparam_index = sample_hyperparam_index

    def load_hyperparams(self):
        work_dir = Path(__file__).parent.parent
        marginal_hyperparam_dir = (
            work_dir
            / "experiments/gplvm_experiments/results/marginal_hyperparams.p"
        )
        condtional_hyperparam_dir = (
            work_dir
            / "experiments/gplvm_experiments/results/conditional_hyperparams.p"
        )
        with open(marginal_hyperparam_dir, "rb") as f:
            marginal_hyperparams = dill.load(f)
        with open(condtional_hyperparam_dir, "rb") as f:
            conditional_hyperparams = dill.load(f)
        # kernel lengthscale, kernel variance, likelihood variance
        return marginal_hyperparams, conditional_hyperparams

    def sample_hyperparams(self, param_list):
        param_choice = random.choice(param_list)
        sample = np.random.normal(loc=param_choice, scale=0.01)
        # Need to make sure hyperparam is positive
        sample_fix = np.maximum(sample, 0.001)
        return sample_fix

    def get_hyerparams(self, marginal=False):
        marginal_hyperparams, conditional_hyperparams = self.load_hyperparams()
        if marginal:
            lengthscale = self.sample_hyperparams(
                marginal_hyperparams["lengthscale"]
            )
            kern_var = self.sample_hyperparams(
                marginal_hyperparams["kern_variance"]
            )
            like_var = self.sample_hyperparams(
                marginal_hyperparams["like_variance"]
            )
        else:
            lengthscale = self.sample_hyperparams(
                conditional_hyperparams["lengthscale"]
            )
            kern_var = self.sample_hyperparams(
                conditional_hyperparams["kern_variance"]
            )
            like_var = self.sample_hyperparams(
                conditional_hyperparams["like_variance"]
            )
        # Convert to the right dtype
        lengthscale = np.array(lengthscale, dtype=default_float())
        kern_var = np.array(kern_var, dtype=default_float())
        like_var = np.array(like_var, dtype=default_float())
        return lengthscale, kern_var, like_var

    def get_collective_hyerparams(self, marginal=False, index=None):
        marginal_hyperparams, conditional_hyperparams = self.load_hyperparams()
        # Sample the same hyperparam index from all the lists
        if index is None:
            hyperparam_index = random.choice(range(len(marginal_hyperparams["lengthscale"])))
        else:
            hyperparam_index = index
            assert hyperparam_index < len(marginal_hyperparams["lengthscale"]), f"Index {hyperparam_index} is out of range!"
        if marginal:
            lengthscale = marginal_hyperparams["lengthscale"][hyperparam_index]
            kern_var = marginal_hyperparams["kern_variance"][hyperparam_index]
            like_var = marginal_hyperparams["like_variance"][hyperparam_index]
        else:
            lengthscale = conditional_hyperparams["lengthscale"][hyperparam_index]
            kern_var = conditional_hyperparams["kern_variance"][hyperparam_index]
            like_var = conditional_hyperparams["like_variance"][hyperparam_index]
        # Sample around the valued
        lengthscale = np.random.normal(loc=lengthscale, scale=0.001)
        lengthscale = np.maximum(lengthscale, 0.0001)
        kern_var = np.random.normal(loc=kern_var, scale=0.001)
        kern_var = np.maximum(kern_var, 0.0001)
        like_var = np.random.normal(loc=like_var, scale=0.001)
        like_var = np.maximum(like_var, 0.0001)

        # Convert to the right dtype
        lengthscale = np.array(lengthscale, dtype=default_float())
        kern_var = np.array(kern_var, dtype=default_float())
        like_var = np.array(like_var, dtype=default_float())
        return lengthscale, kern_var, like_var

    def generate_functions(
        self,
        causal_graph: np.ndarray,
    ) -> dict:
        """
        Generate functions given a causal graph.

        This will instantiate a class that can then be used to generate data.
        This is necessary as we have to save the functions to generate
        interventional data.
        """
        function_dict = {}
        for i in range(self.number_of_variables):
            parents_of_i = causal_graph[:, i]
            # Plus one for latent variable.
            num_parents = int(np.sum(parents_of_i) + 1)

            # Set kernel
            marginal = True if np.sum(parents_of_i) == 0 else False
            if self.sample_hyperparam_index is not None:
                lengthscale, kern_var, like_var = self.get_collective_hyerparams(
                    marginal=marginal,
                    index=self.sample_hyperparam_index,
                )
            elif self.sample_hyperparams_collectively:
                # sample hyperparams collectively
                lengthscale, kern_var, like_var = self.get_collective_hyerparams(
                    marginal=marginal
                )
            else:
                # sample hyperparams one by one
                lengthscale, kern_var, like_var = self.get_hyerparams(
                    marginal=marginal
                )

            kernel = gpflow.kernels.SquaredExponential(
                variance=kern_var,
                lengthscales=lengthscale,
            )

            function = GPLVMFunctions(
                mean=self.mean_function,
                kernel=kernel,
                likelihood_variance=like_var,
            )
            function_dict[i] = function
        return function_dict
