"""
This file contains dataset generators.
"""
from typing import Optional, List

import numpy as np
from torch.utils.data import IterableDataset

from ml2_meta_causal_discovery.datasets.causal_graph_generator import (
    generate_synthetic_dag,
)
from ml2_meta_causal_discovery.datasets.functions_generator import (
    GPFunctionGenerator,
    GPLVMFunctionGenerator,
    LinearFunctionGenerator,
    NeuralNetFunctionGenerator,
)


class ClassifyDatasetGenerator(IterableDataset):
    """Generate datasets using a function generator. This will generate data
    for classification tasks.

    Args:
    ----------
    function_generator : str
        String that specifies the function generator to use.

    num_variables : int
        The number of variables to generate.

    expected_node_degree : int
        Expected node degree of the causal graph.

    batch_size : int
        Batch size to use for the dataset. The batch size shoule be None in the
        DataLoader.

    num_samples : int
        Number of samples to generate. These are the samples of a signle function

    graph_type : str
        The type of graph to generate. Either "erdos-renyi" or "barabasi-albert".

    graph_degrees : List[int]
        The expected degrees of the graph. This is a list of integers. Each graph will
        be sampled uniformly from the list of degrees.

    max_context_size : Optional[int]

    min_context_size : Optional[int]
    """

    def __init__(
        self,
        num_variables: int,
        function_generator: str,
        batch_size: int,
        num_samples: int,
        graph_type: str,
        graph_degrees: List[int],
        kernel_sum: Optional[bool] = False,
        mean_function: str = "latent",
        device: str = "cpu",
    ):
        valid_function_generators = ["gplvm", "gp", "linear", "neuralnet"]
        valid_graph_types = ["ER", "SF"]
        assert (
            function_generator in valid_function_generators
        ), "Function generator is not valid."
        assert graph_type in valid_graph_types, "Graph type is not valid."

        self.num_variables = num_variables
        self.function_generator = function_generator
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.graph_type = graph_type
        self.graph_degrees = graph_degrees
        if self.function_generator == "gplvm":
            self.data_generator = GPLVMFunctionGenerator(
                num_variables=self.num_variables,
                num_samples=self.num_samples,
                interventions=False,
                kernel_sum=kernel_sum,
                mean_function=mean_function,
                device=device,
            )
        elif self.function_generator == "gp":
            self.data_generator = GPFunctionGenerator(
                num_variables=self.num_variables,
                num_samples=self.num_samples,
                interventions=False,
            )
        elif self.function_generator == "linear":
            self.data_generator = LinearFunctionGenerator(
                num_variables=self.num_variables,
                num_samples=self.num_samples,
                interventions=False,
            )
        elif self.function_generator == "neuralnet":
            self.data_generator = NeuralNetFunctionGenerator(
                num_variables=self.num_variables,
                num_samples=self.num_samples,
                interventions=False,
            )
        # elif self.function_generator == "gplvm_fixed_hyperparam":
        #     self.data_generator = GPLVMFixedHyperparam(
        #         num_variables=self.num_variables,
        #         num_samples=self.num_samples,
        #         lengthscale_fixed=lengthscale_fixed,
        #         lengthscale_gamma_vals=lengthscale_gamma_vals,
        #         interventions=False,
        #         kernel_sum=kernel_sum,
        #         mean_function=mean_function,
        #         sample_hyperparams_collectively=sample_hyperparams_collectively,
        #         sample_hyperparam_index=sample_hyperparam_index,
        #     )

    def permute_data(self, *args, permutation_indices: np.ndarray) -> list:
        """
        Permute the data to randomise the causal graph.

        Args:
        ----------
        permutation_indices : np.ndarray shape (num_variables,)

        args : np.ndarray shape (num_samples, num_variables)
            Data to permute.

        Returns:
        ----------
        permuted_data : list of np.ndarray shape (num_samples, num_variables)
        """
        permuted_data = [data[:, permutation_indices] for data in args]
        return permuted_data

    def permute_causal_graph(
        self, *args, permutation_indices: np.ndarray
    ) -> list:
        """
        Permutes the causal graph.

        Args:
        ----------
        permutation_indices : np.ndarray shape (num_variables,)

        args : np.ndarray shape (num_variables, num_variables)
            Causal graph to permute.

        Returns:
        ----------
        permuted_causal_graphs : list of np.ndarray shape (num_variables, num_variables)
        """
        permuted_causal_graphs = [
            dag[permutation_indices, :][:, permutation_indices] for dag in args
        ]
        return permuted_causal_graphs

    def sample_uniform_expected_degree(self) -> int:
        """
        Sample a degree from the list of expected degrees.

        Returns:
        ----------
        degree : int
        """
        degree = np.random.choice(self.graph_degrees)
        return degree

    def generate_next_dataset(
        self,
    ) -> np.ndarray:
        """
        Generate the next dataset point.

        This will be done by sampling a causal graph, and then using the data
        generation functions to generate the data.

        This will generate the observational and interventional data. The
        intervention can only be carried out on the input.

        Args:
        ----------

        Returns:
        ----------
        cntxt_data : np.ndarray shape (batch_size, cntxt_samples, num_variables)
        target_data: np.ndarray shape (batch_size, num_samples, num_variables)
        intervention_data: np.ndarray shape (batch_size, target_samples, num_variables)
        causal_graphs: np.ndarray shape (batch_size, num_variables, num_variables)
        intervened_causal_graphs: np.ndarray shape (batch_size, num_variables, num_variables)
        """
        # Context and target data.
        target_data = np.zeros(
            (self.batch_size, self.num_samples, self.num_variables)
        )
        causal_graphs = np.zeros(
            (self.batch_size, self.num_variables, self.num_variables)
        )
        # Loop over to batch; a different function (causal graph) for each batch
        for b in range(self.batch_size):
            expected_node_degree = self.sample_uniform_expected_degree()
            dag = generate_synthetic_dag(
                d=self.num_variables,
                s0=expected_node_degree,
                graph_type=self.graph_type,
            )

            # Need to permute the graph so that the graph is randomised.
            permutation_indices = np.random.permutation(self.num_variables)

            # [num_samples, num_variables]
            (
                single_data
            ) = self.data_generator.generate_data(
                causal_graph=dag,
                num_int_samples=self.num_samples,
            )

            dag = self.permute_causal_graph(
                dag,
                permutation_indices=permutation_indices,
            )[0]
            # permute the data
            single_data = self.permute_data(
                single_data,
                permutation_indices=permutation_indices,
            )[0]
            # Generate interventional target set
            target_data[b] = single_data
            causal_graphs[b] = dag
        yield (
            target_data,
            causal_graphs,
        )

    def __iter__(self):
        return iter(self.generate_next_dataset())