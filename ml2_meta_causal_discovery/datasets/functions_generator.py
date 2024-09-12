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
import torch as th


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


class LinearFunctions:

    def __init__(self, no_latent=True) -> None:
        self.no_latent = no_latent

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Samples from a linear function.
        """
        if self.no_latent:
            # Ignore the latent variable
            inputs = inputs[:, :-1]

        input_num = inputs.shape[-1]

        if input_num == 1:
            return np.random.normal(loc=0, scale=1.0, size=(inputs.shape[0]))
        else:
            weights = np.random.normal(loc=0, scale=10.0, size=(input_num))
            noise_hyper = np.random.gamma(shape=1.5, scale=2.5, size=(inputs.shape[0]))
            noise = np.random.normal(loc=0, scale=noise_hyper)
            return inputs @ weights + noise


class NeuralNetFunction(th.nn.Module):

    def __init__(self, num_parents: int, no_latent=False) -> None:
        super(NeuralNetFunction, self).__init__()
        self.num_parents = num_parents
        self.no_latent = no_latent
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

        # Define the network
        self.linear_1 = th.nn.Linear(num_parents, 32)
        self.act_1 = th.nn.LeakyReLU()
        self.linear_2 = th.nn.Linear(32, 32)
        self.act_2 = th.nn.LeakyReLU()
        self.linear_3 = th.nn.Linear(32, 1)
        self.to(self.device)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Sample from an NN
        """
        assert inputs.shape[-1] == self.num_parents, "Incorrect input shape!"
        with th.no_grad():
            x = th.tensor(inputs, dtype=th.float32, device=self.device)
            x = self.linear_1(x)
            x = self.act_1(x)
            x = self.linear_2(x)
            x = self.act_2(x)
            x = self.linear_3(x)
            return x.detach().cpu().numpy().squeeze(-1)


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
        # Make sure that the causal graph is a is topologically sorted!
        # Functions will be a dict with keys being the variable number
        function_dict = self.generate_functions(causal_graph)
        data = np.zeros((self.num_samples, self.number_of_variables))
        # Causal graph row i is a parent of column j.
        # We always need to generate the cause first.
        # Thus, we need to loop in order of the causal graph.
        loop_order = np.arange(self.number_of_variables)

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

            variable_obs = variable

            data[:, i] = variable_obs
        return data

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
        """
        data = self.generate_data(
            causal_graph=causal_graph
        )
        return data

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
        data = self.generate_data(
            causal_graph=causal_graph
        )
        return data

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


class LinearFunctionGenerator(DataGenerator):

    def return_data(self, causal_graph: np.ndarray) -> np.ndarray:
        data = self.generate_data(
            causal_graph=causal_graph
        )
        return data

    def generate_functions(
        self,
        causal_graph: np.ndarray,
    ) -> dict:
        function_dict = {}
        for i in range(self.number_of_variables):
            function = LinearFunctions(no_latent=True)
            function_dict[i] = function
        return function_dict


class NeuralNetFunctionGenerator(DataGenerator):
    """
    Generate data by passing a uniform [-1, 1] latent along with inputs
    into a random 2 layer neural network.
    """

    def return_data(self, causal_graph: np.ndarray) -> np.ndarray:
        data = self.generate_data(
            causal_graph=causal_graph
        )
        return data

    def generate_functions(
        self,
        causal_graph: np.ndarray,
    ) -> dict:
        function_dict = {}
        for i in range(self.number_of_variables):
            parents_of_i = causal_graph[:, i]
            # Plus one for latent variable.
            num_parents = int(np.sum(parents_of_i) + 1)
            function = NeuralNetFunction(
                no_latent=False,
                num_parents=num_parents
            )
            function_dict[i] = function
        return function_dict