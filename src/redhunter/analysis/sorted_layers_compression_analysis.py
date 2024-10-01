from __future__ import annotations

from abc import abstractmethod
import copy
import os
from typing import Any, override
from tqdm import tqdm

import numpy as np

import torch

import transformers

from exporch import Config, get_available_device, evaluate_model_on_benchmark
from exporch.utils.causal_language_modeling import load_model_for_causal_lm
from exporch.utils.plot_utils import plot_heatmap_with_additional_row_column, plot_heatmap

from redhunter.analysis.analysis_utils import AnalysisTensorDict, AnalysisTensorWrapper, extract_based_on_path
from redhunter.analysis.delta_layers_rank_analysis import compute_delta_matrices
from redhunter.analysis.layer_replacement_analysis_utils import LayerReplacingModelWrapper
from redhunter.analysis_experiment import AnalysisExperiment

from redhunter.analysis.sorted_layers_compression_analysis_utils import sort_columns, sort_rows


class SortedLayersCompressionAnalysis(AnalysisExperiment):
    """
    Class to perform the analysis of the sorted layers rank.
    """

    mandatory_keys = ["num_layers", "benchmark_id"]

    def __init__(
            self,
            config_file_path: str
    ) -> None:

        super().__init__(config_file_path)

        self.title = "Sorted Layers Compression Analysis"
        self.axis_titles = None
        self.x_title = "Index of the block"
        self.y_title = "Index of the block"

    def _perform_analysis(
            self
    ) -> None:
        """
        Performs the aligned layers rank analysis.
        """

        self._perform_sorted_layers_compression_analysis()

    def _perform_sorted_layers_compression_analysis(
            self
    ) -> None:
        """
        Performs the sorted layers compression analysis.
        """

        self._compute_sorted_layers_deltas()
        self.store_data()

    # TODO problems with the storage of the wrappers, they are too huge
    def _compute_sorted_layers_deltas(
            self
    ) -> None:
        """
        Computes the difference between the layers in a block and the sorted layers in the other block.
        """

        config = self.config
        verbose = config.get_verbose()
        device_str = config.get("device") if config.contains("device") else "cpu"
        device = get_available_device(device_str)
        store_interval = config.get("store_interval") if config.contains("store_interval") else 10

        if self.get_data() is not None:
            # Loading the data
            original_tensor_wrappers, sorted_layers_deltas, objective_function_stats_dict = self.get_data()
            model = None
        else:
            # Initializing the data structures
            original_tensor_wrappers = AnalysisTensorDict()
            sorted_layers_deltas = AnalysisTensorDict()
            objective_function_stats_dict = {}

            # Loading the model
            config.set("device", "cpu")
            model = load_model_for_causal_lm(config)
            config.set("device", device_str)
            tokenizer = transformers.AutoTokenizer.from_pretrained(config.get("model_id"))

            # Extracting the layers to analyze
            extract_based_on_path(
                module_tree=model,
                target_paths=self.config.get("targets"),
                layers_storage=original_tensor_wrappers,
                blacklist=config.get("blacklist") if config.contains("blacklist") else [],
                verbose=verbose
            )
            self.log(f"Layers extracted: {original_tensor_wrappers}")

            # Setting the configurations to analyze
            original_tensor_wrappers.set_layer_paths_configurations_to_analyze(self.get_experiments_configurations())

            # Storing the data
            self.set_data((original_tensor_wrappers, sorted_layers_deltas, objective_function_stats_dict))
            self.store_data()

        # Iterating over the remaining configurations that have to be analyzed
        remaining_configurations_to_analyze = original_tensor_wrappers.get_layer_paths_configurations_to_analyze()
        self.process_tensor_wrappers(original_tensor_wrappers)
        configurations_to_remove = []

        for configurations_index, configuration_to_analyze in enumerate(remaining_configurations_to_analyze):
            self.log(f"Analyzing the configuration: {configuration_to_analyze}")
            print(f"Analyzing the configuration: {configuration_to_analyze}")
            # Getting the layers to analyze
            layer_wrappers_to_analyze = original_tensor_wrappers.get_wrappers_for_analysis(configuration_to_analyze)
            layers_in_block_1 = layer_wrappers_to_analyze[0]
            layers_in_block_2 = layer_wrappers_to_analyze[1]

            # Sorting the rows/columns of a layer based minimizing a certain metrics given a couple of transformer blocks
            sorting_axes = [1 if index == len(layers_in_block_2) - 1 else 0 for index in range(len(layers_in_block_2))]
            sorting_indices, objective_function_stats = self.compute_indices_sorting(layers_in_block_1, layers_in_block_2, sorting_axes)
            #objective_function_stats = {"abc": 1} # TESTING
            self.log(f"Sorting indices computed.")
            self.log(f"Sorting indices: {sorting_indices}")
            self.log(f"Objective function stats: {objective_function_stats}")

            # Using the ordering to sort the vectors in the matrices of block 2
            sorted_layers_in_block_2 = self.sort_elements_in_layers(layers_in_block_2, sorting_indices, sorting_axes)
            self.log(f"Layers sorted.")
            #sorted_layers_in_block_2 = layers_in_block_2

            # Subtracting the layers of block and the sorted layers of another block
            delta_matrices = compute_delta_matrices(layers_in_block_1, sorted_layers_in_block_2)

            # Processing the delta matrices of the layers based on the specific operations that the analysis performs
            for delta_matrix in delta_matrices:
                self.process_tensor_wrappers(delta_matrix)
            # objective_function_stats = {"final_sorting_resettable_elements": 42, "initial_sorting_resettable_elements": 2018} # TESTING
            self.log("Delta matrices computed and processed.")

            key = (tuple(tuple(layer.get_path()) for layer in layers_in_block_1), tuple(tuple(layer.get_path()) for layer in layers_in_block_2))
            sorted_layers_deltas.set_tensor(key, delta_matrices)

            # Preparing the model for evaluation
            results, model = self.evaluate_model(
                model,
                tokenizer,
                self.get_compressed_weights_mapping(
                    configuration_to_analyze,
                    layers_in_block_1,
                    layers_in_block_2,
                    sorted_layers_in_block_2,
                    delta_matrices
                ),
                benchmark_id=config.get("benchmark_id"),
                device=device_str
            )

            objective_function_stats.update(results)
            objective_function_stats_dict[key] = objective_function_stats
            self.log(f"Performance dictionary updated with the results.")

            # Storing the data
            self.set_data((original_tensor_wrappers, sorted_layers_deltas, objective_function_stats_dict))
            configurations_to_remove.append(configuration_to_analyze)
            if (configurations_index + 1) % store_interval == 0:
                self.store_data()
                # Removing the configurations that have been analyzed
                original_tensor_wrappers.remove_layer_paths_configuration_to_analyze(configurations_to_remove)
                configurations_to_remove = []

        # Storing the data
        self.set_data((original_tensor_wrappers, sorted_layers_deltas, objective_function_stats_dict))
        self.store_data()
        original_tensor_wrappers.remove_layer_paths_configuration_to_analyze(configurations_to_remove)
        self.log("All configurations analyzed and all data stored.\nAnalysis completed.")
        print("Analysis completed.")

    def get_experiments_configurations(
            self
    ) -> list[list[list[list[str]]]]:
        """
        Returns the configurations of the experiments to perform the analysis.
        The structure of the configurations is the following:
        [
            [
                [
                    [str, ...], # Path of a single layer as a list of strings
                     ...
                ], # List of paths for the first block matrices
                [
                    [str, ...], # Path of a single layer as a list of strings
                     ...
                ] # List of paths for the second block matrices
            ],
            ...
        ]

        Returns:
            list[list[list[list[str]]]]:
                The configurations of the experiments to perform the analysis.
        """

        targets_lists = self.config.get("targets")
        num_layers = self.config.get("num_layers")

        return [
            [
                [[str(i) if string == "block_index" else string for string in target] for target in targets_lists],
                [[str(j) if string == "block_index" else string for string in target] for target in targets_lists]
            ] for i in range(num_layers) for j in range(num_layers)
        ]

    def process_tensor_wrappers(
            self,
            tensor_wrapper: AnalysisTensorWrapper
    ) -> None:
        """
        Processes the tensor wrapper to compute the quantities needed to perform the analysis.

        Args:
            tensor_wrapper (AnalysisTensorWrapper):
                The tensor wrapper to process.
        """

        pass

    @abstractmethod
    def compute_indices_sorting(
            self,
            layers_in_block_1: list[AnalysisTensorWrapper],
            layers_in_block_2: list[AnalysisTensorWrapper],
            axes: list[int]
    ) -> [list, dict[torch.Tensor]]:
        """
        Computes the indices for the sorting of the elements of the layers in block 2 that minimizes a certain loss with
        respect to the layers in block 1.

        Args:
            layers_in_block_1 (list[AnalysisTensorWrapper]):
                The list of layers in block 1.
            layers_in_block_2 (list[AnalysisTensorWrapper]):
                The list of layers in block 2.
            axes (list[int]):
                The list of axes where the components have to be sorted.
        """

        pass

    def sort_elements_in_layers(
            self,
            layers_in_block: list[AnalysisTensorWrapper],
            sorting_indices_list: list[list[int]] | list[int],
            axes: list[int]
    ) -> list[AnalysisTensorWrapper]:
        """
        Sorts the elements in the layers based on a list of indices.

        Args:
            layers_in_block (list[AnalysisTensorWrapper]):
                The list of layers in the block.
            sorting_indices_list (list[list[int]] | list[int]):
                The list of the lists of indices to sort the components.
            axes (list[int]):
                The list of axes where the components have to be sorted.

        Returns:
            list[AnalysisTensorWrapper]:
                The matrices sorted in the given dimension based on the sorting indices.
        """

        if len(layers_in_block) <= 0:
            raise ValueError("The list of layers in the block must contain at least one element.")
        if len(sorting_indices_list) <= 0:
            raise ValueError("The list of indices must contain at least one element.")
        if len(axes) <= 0:
            raise ValueError("The list of axes must contain at least one element.")

        if isinstance(sorting_indices_list[0], int):
            sorting_indices_list = [sorting_indices_list for _ in range(len(layers_in_block))]

        if len(sorting_indices_list) != len(layers_in_block) or len(axes) != len(layers_in_block):
            raise ValueError("The length of the ordering and axes lists must be the same as the length of the layers list.")

        sorted_layers = []
        for layer, ordering, axis in zip(layers_in_block, sorting_indices_list, axes):
            if axis == 0:
                sorted_tensor = sort_rows(layer.get_tensor(), ordering)
            elif axis == 1:
                sorted_tensor = sort_columns(layer.get_tensor(), ordering)
            else:
                raise Exception(f"Axis '{axis}' not supported.")

            sorted_layers.append(
                AnalysisTensorWrapper(
                    sorted_tensor,
                    name=layer.get_name(),
                    label=layer.get_label(),
                    path=layer.get_path(),
                    block_index=layer.get_block_index(),
                    layer=layer.get_layer()
                )
            )
        self.log(f"Layers sorted for the tensor with labels:\n\t{'\n\t'.join([layer.get_label() for layer in layers_in_block])}.")

        return sorted_layers

    @abstractmethod
    def get_compressed_weights_mapping(
            self,
            configuration_to_analyze:  list[list[list[str]]],
            layers_in_block_1: list[AnalysisTensorWrapper],
            layers_in_block_2: list[AnalysisTensorWrapper],
            sorted_layers_in_block_2: list[AnalysisTensorWrapper],
            delta_matrices: list[AnalysisTensorWrapper]
    ) -> dict[tuple[str], torch.Tensor]:
        """
        Returns the compressed weights.

        Args:
            configuration_to_analyze (list[list[list[str]]]):
                The configuration to analyze.
            layers_in_block_1 (list[AnalysisTensorWrapper]):
                The list of layers in block 1.
            layers_in_block_2 (list[AnalysisTensorWrapper]):
                The list of layers in block 2.
            sorted_layers_in_block_2 (list[AnalysisTensorWrapper]):
                The list of sorted layers in block 2.
            delta_matrices (list[AnalysisTensorWrapper]):
                The list of delta matrices.

        Returns:
            dict[tuple(str), torch.Tensor]:
                The compressed weights.
        """

        return {}

    def evaluate_model(
            self,
            model: [torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel],
            tokenizer: [transformers.AutoTokenizer | transformers.PreTrainedTokenizer],
            mapping_compressed_layers: dict,
            benchmark_id: str,
            device: str
    ) -> tuple[dict, torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel]:
        """
        Evaluates the model on a benchmark.

        Args:
            model (torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel):
                The model to evaluate.
            tokenizer (transformers.AutoTokenizer | transformers.PreTrainedTokenizer):
                The tokenizer to use.
            mapping_compressed_layers (dict):
                The mapping of the compressed layers.
            benchmark_id (str):
                The id of the benchmark to evaluate the model on.
            device (str):
                The device where the model has to be evaluated.

        Returns:
            dict:
                The results of the evaluation.
            torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel:
                The loaded original model.
        """

        if model is None:
            model = load_model_for_causal_lm(self.config)

        model.to(device)
        model_wrapper = LayerReplacingModelWrapper(
            model,
            mapping_compressed_layers
        )

        self.log("Model prepared for evaluation.")

        # Evaluating the processed model
        self.log(f"Starting the evaluation of the model on the device {model_wrapper.get_model().device}.")
        results = evaluate_model_on_benchmark(model_wrapper.get_model(), tokenizer, benchmark_id, {}, device)
        #results = {self.config.get("benchmark_id"): {"acc_norm,none": 0.5}}  # Testing
        self.log(f"Results of the modified model: {results}.")
        print(f"Results of the modified model: {results}.")

        model_wrapper.reset_replacement()

        return results, model_wrapper.get_model()

    def _plot_results(
            self,
            config: Config,
            data: Any
    ) -> None:
        """
        Plots the results obtained from the analysis.
        The performed analysis will depend on the specific subclass of AnalysisExperiment.

        Args:
            config (Config):
                The configuration of the experiment.
            data (Any):
                The data obtained from the analysis.
        """

        # Extracting the information to plot
        statistics_delta_layers, statistics_original_layers, row_labels, column_labels = self._get_formatted_results_to_plot(config, data)

        # Plotting the results
        if statistics_original_layers is None:
            plot_heatmap(
                statistics_delta_layers,
                save_path=os.path.join(config.get("directory_path"), "heatmap.png"),
                title=self.title + f"(Model: {config.get("model_id")})", axis_titles=self.axis_titles,
                x_title=self.x_title, y_title=self.y_title,
                x_labels=column_labels, y_labels=row_labels,
                fig_size=config.get("figure_size") if config.contains("figure_size") else (30, 30)
            )
        else:
            plot_heatmap_with_additional_row_column(
                statistics_delta_layers,
                values_rows_lists=statistics_original_layers, values_columns_lists=statistics_original_layers,
                save_path=os.path.join(config.get("directory_path"), "heatmap.png"),
                title=self.title + f"(Model: {config.get("model_id")})", axis_titles=self.axis_titles,
                x_title=self.x_title, y_title=self.y_title,
                x_labels=column_labels, y_labels=row_labels,
                fig_size=config.get("figure_size") if config.contains("figure_size") else (30, 30)
            )

    def _get_formatted_results_to_plot(
            self,
            config: Config,
            data: Any
    ) -> tuple[list[list[np.ndarray | torch.Tensor]], list[list[np.ndarray | torch.Tensor]], list[list[str]], list[list[str]]]:
        """
        Returns the formatted results to plot.

        Args:
            config (Config):
                The configuration of the experiment.
            data (Any):
                The data obtained from the analysis.

        Returns:
            list[list[np.ndarray | torch.Tensor]]:
                The matrices to plot containing the results of the analysis on the delta matrices.
            list[list[np.ndarray | torch.Tensor]]:
                The matrices to plot related to the original layers, if any.
            list[list[str]]:
                The labels of the rows of the matrices to plot.
            list[list[str]]:
                The labels of the columns of the matrices to plot
        """

        return self.get_formatted_results_to_plot(config, data)

    @abstractmethod
    def get_formatted_results_to_plot(
            self,
            config: Config,
            data: Any
    ) -> tuple[list[list[np.ndarray | torch.Tensor]], list[list[np.ndarray | torch.Tensor]], list[list[str]], list[list[str]]]:
        """
        Returns the formatted results to plot.

        Args:
            config (Config):
                The configuration of the experiment.
            data (Any):
                The data obtained from the analysis.

        Returns:
            list[list[np.ndarray | torch.Tensor]]:
                The matrices to plot containing the results of the analysis on the delta matrices.
            list[list[np.ndarray | torch.Tensor]]:
                The matrices to plot related to the original layers, if any.
            list[list[str]]:
                The labels of the rows of the matrices to plot.
            list[list[str]]:
                The labels of the columns of the matrices to plot
        """

        pass


class ResettableElementsSortedLayersCompressionAnalysisWithConcatenatedMatrices(SortedLayersCompressionAnalysis):
    """
    Class to perform the analysis of the sorted layers.
    The sorting is based on the number of resettable elements in the difference between the layers.
    """

    mandatory_keys = ["zero_threshold"]

    def __init__(
            self,
            config_file_path: str
    ) -> None:

        super().__init__(config_file_path)
        self.title = "Resettable Elements Sorted Layers Compression Analysis with Concatenated Matrices"
        self.axis_titles = ["Number of elements that can be reset before sorting",
                            "Number of elements that can be reset after sorting"]

    @override
    def process_tensor_wrappers(
            self,
            tensor_wrapper: AnalysisTensorWrapper
    ) -> None:
        """
        Processes the tensor wrapper to compute the quantities needed to perform the analysis.

        Args:
            tensor_wrapper (AnalysisTensorWrapper):
                The tensor wrapper to process.
        """

        pass

    @override
    def compute_indices_sorting(
            self,
            layers_in_block_1: list[AnalysisTensorWrapper],
            layers_in_block_2: list[AnalysisTensorWrapper],
            axes: list[int]
    ) -> [list, dict[torch.Tensor]]:
        """
        Computes the indices for the sorting of the elements of the layers in block 2 that maximizes the number of zeroed
        elements in the difference with respect to the layers in block 1 when thresholding all the smallest components.

        Args:
            layers_in_block_1 (list[AnalysisTensorWrapper]):
                The list of layers in block 1.
            layers_in_block_2 (list[AnalysisTensorWrapper]):
                The list of layers in block 2.
            axes (list[int]):
                The list of axes where the components have to be sorted.
        """

        if len(layers_in_block_1) != len(layers_in_block_2):
            raise ValueError("The number of layers in block 1 and block 2 must be the same.")
        if len(axes) != len(layers_in_block_1) or len(axes) != len(layers_in_block_2):
            raise ValueError(f"The length of the axes list must be the same as the length of the layers list.\n"
                             f"The length of the axes list is {len(axes)}; the length of the layers in block 1 is "
                             f"{len(layers_in_block_1)}; the length of the layers in block 2 is {len(layers_in_block_2)}.")
        if not all(axis == 0 or axis == 1 for axis in axes):
            raise ValueError("The axes must be 0 or 1.")

        zero_threshold = self.config.get("zero_threshold")
        device = get_available_device(self.config.get("device") if self.config.contains("device") else "cpu")
        batch_size = self.config.get("batch_size") if self.config.contains("batch_size") else 1

        concatenated_matrix_1 = torch.cat(
            [layer.get_tensor().t() if axis == 0
             else layer.get_tensor()
             for layer, axis in zip(layers_in_block_1, axes)],
            dim=0
        ).to(device)
        concatenated_matrix_2 = torch.cat(
            [layer.get_tensor().t() if axis == 0
             else layer.get_tensor()
             for layer, axis in zip(layers_in_block_2, axes)],
            dim=0
        ).to(device)
        if concatenated_matrix_1.shape[0] != concatenated_matrix_2.shape[0]:
            raise ValueError("The dimension of the concatenated matrices where the subtraction is computed must be the same.")
        if concatenated_matrix_1.shape != concatenated_matrix_2.shape:
            self.log("The shapes of the concatenated matrices are different.")
            print("The shapes of the concatenated matrices are different.")

        concatenated_dim_1, dim_to_sort_1 = concatenated_matrix_1.shape
        concatenated_dim_2, dim_to_sort_2 = concatenated_matrix_2.shape

        sorting_stats = {
            "initial_sorting_resettable_elements": torch.sum(torch.abs(concatenated_matrix_1 - concatenated_matrix_2) < zero_threshold).item(),
            "manhattan_norm_initial_sorting_resettable_elements": torch.sum(torch.abs(concatenated_matrix_1 - concatenated_matrix_2) * (torch.abs(concatenated_matrix_1 - concatenated_matrix_2) < zero_threshold)).item(),
            "frobenius_norm_initial_sorting_resettable_elements": torch.norm(concatenated_matrix_1 - concatenated_matrix_2 * (torch.abs(concatenated_matrix_1 - concatenated_matrix_2) < zero_threshold)).item()
        }

        self.log("Counting the number of smaller-than-threshold entries for each couple of stacked internal vectors of "
                 "the layers.")
        print("Counting the number of smaller-than-threshold entries for each couple of stacked internal vectors of "
              "the layers.")

        count_matrix = torch.zeros((dim_to_sort_1, dim_to_sort_2), device=concatenated_matrix_1.device)
        # Iterating over batches to optimize the computation of the value of the objective function
        for start in tqdm(range(0, dim_to_sort_1, batch_size)):
            end = min(start + batch_size, dim_to_sort_1)

            # Expanding the dimensions for broadcasting and computing the difference between the two matrices
            diff =  concatenated_matrix_1[:, start:end].unsqueeze(2) - concatenated_matrix_2.unsqueeze(1)

            # Counting how many elements are smaller than the threshold
            count_matrix[start:end, :] += (diff.abs() < zero_threshold).sum(dim=0)

        # Greedily associating columns
        sorting_indices = []
        used_indices = torch.zeros(dim_to_sort_2, dtype=torch.bool)

        for count_row in tqdm(count_matrix):
            # Getting the indices that have not been used yet
            available_indices = torch.arange(dim_to_sort_2)[~used_indices]
            # Getting the index of the column with the maximum number of resettable elements
            max_index = available_indices[torch.argmax(count_row[available_indices])]
            sorting_indices.append(max_index.item())
            used_indices[max_index] = True

        final_delta_concatenated_matrices = concatenated_matrix_1 - sort_columns(concatenated_matrix_2, sorting_indices)

        sorting_stats.update({
            "final_sorting_resettable_elements": torch.sum(torch.abs(final_delta_concatenated_matrices) < zero_threshold).item(),
            "manhattan_norm_final_sorting_resettable_elements": torch.sum(torch.abs(final_delta_concatenated_matrices) * (torch.abs(final_delta_concatenated_matrices) < zero_threshold)).item(),
            "frobenius_norm_final_sorting_resettable_elements": torch.norm(final_delta_concatenated_matrices * (torch.abs(final_delta_concatenated_matrices) < zero_threshold)).item()
        })

        return sorting_indices, sorting_stats

    @override
    def get_compressed_weights_mapping(
            self,
            configuration_to_analyze: list[list[list[str]]],
            layers_in_block_1: list[AnalysisTensorWrapper],
            layers_in_block_2: list[AnalysisTensorWrapper],
            sorted_layers_in_block_2: list[AnalysisTensorWrapper],
            delta_matrices: list[AnalysisTensorWrapper]
    ) -> dict[tuple[str, ...], torch.Module]:
        """
        Returns the compressed weights.

        Args:
            configuration_to_analyze (list[list[list[str]]]):
                The configuration to analyze.
            layers_in_block_1 (list[AnalysisTensorWrapper]):
                The list of layers in block 1.
            layers_in_block_2 (list[AnalysisTensorWrapper]):
                The list of layers in block 2.
            sorted_layers_in_block_2 (list[AnalysisTensorWrapper]):
                The list of sorted layers in block 2.
            delta_matrices (list[AnalysisTensorWrapper]):
                The list of delta matrices.

        Returns:
            dict[tuple(str), torch.Tensor]:
                The compressed weights.
        """

        mapping = {}
        for layer_path in configuration_to_analyze[1]:
            layer_index = configuration_to_analyze[1].index(layer_path)
            layer = copy.deepcopy(layers_in_block_2[layer_index].get_layer())
            try:
                layer.weight.data = layers_in_block_1[layer_index].get_tensor().data + self.threshold_zeros(delta_matrices[layer_index].get_tensor().data, self.config.get("zero_threshold"))
            except AttributeError:
                self.log(f"Layer {layer.get_label()} does not have a weight attribute.")
                print(f"Layer {layer.get_label()} does not have a weight attribute.")
            # TODO for now we are not considering the bias
            """
            try:
                layer.bias = layers_in_block_1[layer_index].get_tensor() + self.threshold_zeros(delta_matrices[layer_index].get_tensor())
            except AttributeError:
                self.log(f"Layer {layer.get_label()} does not have a bias attribute.")
                print(f"Layer {layer.get_label()} does not have a bias attribute.")
            """

            mapping[tuple(layer_path)] = layer

        return mapping

    @staticmethod
    def threshold_zeros(
            tensor: torch.Tensor,
            threshold: float = 1e-6
    ) -> torch.Tensor:
        """
        Returns the tensor with the elements bigger than the threshold set to zero.

        Args:
            tensor (torch.Tensor):
                The tensor to threshold.
            threshold (float):
                The threshold to apply.

        Returns:
            torch.Tensor:
                The tensor with the elements bigger than the threshold set to zero.
        """

        return tensor * (torch.abs(tensor) > threshold)

    @override
    def get_formatted_results_to_plot(
            self,
            config: Config,
            data: Any
    ) -> tuple[list[list[np.ndarray | torch.Tensor]], list[list[np.ndarray | torch.Tensor]], list[list[str]], list[list[str]]]:
        """
        Returns the formatted results to plot.

        Args:
            config (Config):
                The configuration of the experiment.
            data (Any):
                The data obtained from the analysis.

        Returns:
            list[list[np.ndarray | torch.Tensor]]:
                The matrices to plot containing the results of the analysis on the delta matrices.
            list[list[np.ndarray | torch.Tensor]]:
                The matrices to plot related to the original layers, if any.
            list[list[str]]:
                The labels of the rows of the matrices to plot.
            list[list[str]]:
                The labels of the columns of the matrices to plot
        """

        # Helper function to extract all numerical sequences from tuple keys
        def extract_numerical_sequence(path_key):
            # Helper function to extract numbers from a single sublist
            def extract_numbers(sublist):
                return [int(s) for s in sublist if s.isdigit()]

            # Extracting numbers from all parts of the first tuple element
            seq = tuple(extract_numbers(sublist) for sublist in path_key)

            # Flattening nested tuples and return as one single tuple
            flat_seq = tuple(num for sublist in seq for num in sublist)

            return flat_seq

        _, _, results_dict = data

        row_elements = set()
        column_elements = set()

        # Collecting row and column elements
        for key in results_dict.keys():
            row_elements.add(key[0])
            column_elements.add(key[1])

        # Converting sets to sorted lists using the custom numeric sequence extraction and sorting
        first_elements = sorted(row_elements, key=lambda elem: extract_numerical_sequence(elem))
        second_elements = sorted(column_elements, key=lambda elem: extract_numerical_sequence(elem))

        # Creating matrices to be plotted
        final_sorting_matrix = np.zeros((len(first_elements), len(second_elements)))
        initial_sorting_matrix = np.zeros((len(first_elements), len(second_elements)))

        # Filling matrices with values from the dictionary
        for key, value in results_dict.items():
            first_index = first_elements.index(key[0])
            second_index = second_elements.index(key[1])

            final_sorting_matrix[first_index, second_index] = value["final_sorting_resettable_elements"]
            initial_sorting_matrix[first_index, second_index] = value["initial_sorting_resettable_elements"]

        return [[initial_sorting_matrix], [final_sorting_matrix]], None, [first_elements, first_elements], [
            second_elements, second_elements]


class SimilarityBasedSortedLayersCompressionAnalysis(SortedLayersCompressionAnalysis):
    """
    Class to perform the analysis of the sorted layers.
    The sorting is based on the similarity between the layers.
    """

    mandatory_keys = ["similarity_guide_index", "singular_values_threshold"]

    def __init__(
            self,
            config_file_path: str
    ) -> None:

        super().__init__(config_file_path)

        self.title = "Similarity-Based Sorted Layers Compression Analysis"
        self.axis_titles = None
        self.x_title = "Index of the block"
        self.y_title = "Index of the block"

    @override
    def process_tensor_wrappers(
            self,
            tensor_wrapper: AnalysisTensorWrapper
    ) -> None:
        """
        Processes the tensor wrapper to compute the quantities needed to perform the analysis.

        Args:
            tensor_wrapper (AnalysisTensorWrapper):
                The tensor wrapper to process.
        """

        pass

    @override
    def compute_indices_sorting(
            self,
            layers_in_block_1: list[AnalysisTensorWrapper],
            layers_in_block_2: list[AnalysisTensorWrapper],
            axes: list[int]
    ) -> [list, dict[torch.Tensor]]:
        """
        Computes the indices for the sorting of the elements of the layers in block 2 that maximizes the number of zeroed
        elements in the difference with respect to the layers in block 1 when thresholding all the smallest components.

        Args:
            layers_in_block_1 (list[AnalysisTensorWrapper]):
                The list of layers in block 1.
            layers_in_block_2 (list[AnalysisTensorWrapper]):
                The list of layers in block 2.
            axes (list[int]):
                The list of axes where the components have to be sorted.
        """

        return [], {}

    @override
    def get_formatted_results_to_plot(
            self,
            config: Config,
            data: Any
    ) -> tuple[list[list[np.ndarray | torch.Tensor]], list[list[np.ndarray | torch.Tensor]], list[list[str]], list[list[str]]]:
        """
        Returns the formatted results to plot.

        Args:
            config (Config):
                The configuration of the experiment.
            data (Any):
                The data obtained from the analysis.

        Returns:
            list[list[np.ndarray | torch.Tensor]]:
                The matrices to plot containing the results of the analysis on the delta matrices.
            list[list[np.ndarray | torch.Tensor]]:
                The matrices to plot related to the original layers, if any.
            list[list[str]]:
                The labels of the rows of the matrices to plot.
            list[list[str]]:
                The labels of the columns of the matrices to plot
        """
        
        return [], [], [], []