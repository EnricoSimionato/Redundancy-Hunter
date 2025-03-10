from __future__ import annotations

from abc import abstractmethod
import copy
import logging
import os
from typing import Any, override
from tqdm import tqdm

import matplotlib.pyplot as plt

import numpy as np

import torch

import transformers

from exporch import Config, get_available_device, Verbose
from exporch.experiment import evaluate_model_on_benchmark
from exporch.utils.causal_language_modeling import load_model_for_causal_lm
from exporch.utils.plot_utils import plot_heatmap_with_additional_row_column, plot_heatmap, set_axis_labels, get_text_color

from redhunter.analysis.analysis_utils import AnalysisTensorDict, AnalysisTensorWrapper, extract
from redhunter.utils.layer_replacement_wrapper.layer_replacement_wrapper import LayerReplacingModelWrapper
from redhunter.analysis_experiment import AnalysisExperiment


# TODO fix it, I don't know what point it is


def compute_cosine(
        x: torch.Tensor,
        y: torch.Tensor,
        dim: int = 0,
        verbose: Verbose = Verbose.INFO
) -> torch.Tensor:
    """
    Computes the cosine similarity between two tensors.

    Args:
        x (torch.Tensor):
            The first tensor.
        y (torch.Tensor):
            The second tensor.
        dim (int):
            The dimension to compute the cosine similarity. Default: 0.
        verbose (Verbose):
            The verbosity level. Default: Verbose.INFO.

    Returns:
        torch.Tensor:
            The cosine similarity between the two tensors.
    """

    if verbose > Verbose.INFO:
        print(f"Computing cosine similarity.")
        print(f"The shape of x is {x.shape}.")
        print(f"The shape of y is {y.shape}.")
        print(f"The cosine similarity between x and y is "
              f"{torch.nn.functional.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0)).item()}")

    return torch.nn.functional.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0), dim=dim)


def plot_similarity_stats(
        similarity_stats: dict,
        save_path: str,
        title: str,
        axis_titles: tuple = ("Initial statistics", "Final statistics"),
        x_title: str = "Column Index",
        y_title: str = "Row Index",
        x_labels: list = None,
        y_labels: list = None,
        fig_size: tuple = (28, 14),
        precision: int = 4
) -> None:
    """
    Plot the statistics about the similarities between the layers of the different blocks.

    Args:
        similarity_stats (dict):
            The dictionary containing the statistics about the similarities between the layers of the different blocks.
        save_path (str):
            The path where to save the plot.
        title (str):
            The title of the plot.
        axis_titles (tuple):
            The titles of the x-axis and y-axis. Default: ("Initial statistics", "Final statistics").
        x_title (str):
            The title of the x-axis. Default: "Column Index".
        y_title (str):
            The title of the y-axis. Default: "Row Index".
        x_labels (list):
            The labels of the x-axis. Default: None.
        y_labels (list):
            The labels of the y-axis. Default: None.
        fig_size (tuple):
            The size of the figure. Default: (14, 7).
        precision (int):
            The precision of the values. Default: 4.
    """

    # Initialize matrices to store mean and max similarities
    num_blocks = len(set([block_index[0] for block_index in similarity_stats.keys()]))

    # Preparing a list to store the result
    similarity_stats_plot = []
    similarity_stats_guide = []

    # Assuming both keys have the same length of lists
    keys = list(similarity_stats.keys())
    num_elements = len(similarity_stats[keys[0]]["initial mean similarities"])

    # Creating dictionaries for each element
    for i in range(num_elements):
        combined_dict = {}
        guide_matrix = np.zeros((num_blocks, num_blocks))

        # Looping over each key in the dictionary
        for key, values in similarity_stats.items():
            # Get the lists for each category
            initial_mean = values["initial mean similarities"]
            initial_max = values["initial maximum similarities"]
            final_mean = values["final mean similarities"]
            final_max = values["final maximum similarities"]

            # Combining the corresponding elements for the current index
            combined_dict[key] = f"{round(initial_mean[i], precision)}\n{round(initial_max[i], precision)}\n{round(final_mean[i], precision)}\n{round(final_max[i], precision)}"
            guide_matrix[key[0], key[1]] = final_mean[i]

        # Appending the combined dictionary to the result list
        similarity_stats_plot.append(combined_dict)
        similarity_stats_guide.append(guide_matrix)

    num_axis = len(similarity_stats_plot)

    # Create a figure with 2 subplots for initial and final statistics
    fig, axs = plt.subplots(1, num_axis, figsize=fig_size)
    fig.suptitle(title)
    for ax, axis_title in zip(axs, axis_titles):
        ax.set_title(axis_title)
    cmap_str = "Blues"

    for axis_index in range(num_axis):
        cax = axs[axis_index].matshow(similarity_stats_guide[axis_index], cmap=cmap_str, vmin=0, vmax=1)
        fig.colorbar(cax, ax=axs[axis_index])

        # Add text annotations for mean and max in each cell
        for i in range(num_blocks):
            for j in range(num_blocks):
                axs[axis_index].text(j, i, similarity_stats_plot[axis_index][(i, j)] if i != j else "1.0\n1.0\n1.0\n1.0", ha="center", va="center",
                                     color=get_text_color(float(similarity_stats_guide[axis_index][i, j]), plt.get_cmap(cmap_str)))

        set_axis_labels(axs[axis_index], x_title, y_title, x_labels, y_labels, x_rotation=90)

    # Adjusting layout and show plot
    plt.tight_layout()

    # Saving the plot
    plt.savefig(save_path)
    logging.info(f"Heatmap saved at '{save_path}'")


def cosine_similarity_respective_couples_of_components(
        x: torch.Tensor,
        y: torch.Tensor,
        between: str = "rows"
) -> torch.Tensor:
    """
    Compute the cosine similarity between the respective rows or columns of two matrices.

    Args:
        x (torch.Tensor):
            The first matrix.
        y (torch.Tensor):
            The second matrix.
        between (str):
            The dimension to compute the cosine similarity. Default: "rows".

    Returns:
        torch.Tensor:
            The cosine similarity between the two matrices.
            The index of the row of the similarity matrix is the index of the row of the first matrix; the index of the
            column of the similarity matrix is the index of the row of the second matrix.
    """

    if between not in ["rows", "columns"]:
        raise ValueError("between must be 'rows' or 'columns'")

    dim = 1 if between == "rows" else 0

    x_normalized = x / torch.norm(x, p=2, dim=dim, keepdim=True)
    y_normalized = y / torch.norm(y, p=2, dim=dim, keepdim=True)

    xy_normalized = x_normalized * y_normalized

    similarity = torch.sum(xy_normalized, dim=dim)

    return similarity


def cosine_similarity_all_couples_of_components(
        x: torch.Tensor,
        y: torch.Tensor,
        between="rows",
) -> torch.Tensor:
    """
    Compute cosine similarity between all the rows or all the columns of two matrices.

    Args:
        x (torch.Tensor):
            The first matrix.
        y (torch.Tensor):
            The second matrix.
        between (str):
            The dimension to compute the cosine similarity. Default: "rows".

    Returns:
        torch.Tensor:
            The cosine similarity between the two matrices.
            The index of the row of the similarity matrix is the index of the row of the first matrix; the index of the
            column of the similarity matrix is the index of the row of the second matrix.

    Raises:
        ValueError:
            If the value of the parameter between is not "rows" or "columns".
    """

    logging.info(f"Computing cosine similarity between the {between} of the two matrices.")

    if between not in ["rows", "columns"]:
        raise ValueError("between must be 'rows' or 'columns'")

    dim = 1 if between == "rows" else 0

    x_normalized = x / torch.norm(x, p=2, dim=dim, keepdim=True)
    y_normalized = y / torch.norm(y, p=2, dim=dim, keepdim=True)

    return torch.matmul(x_normalized, y_normalized.T) if between == "rows" else torch.matmul(x_normalized.T,
                                                                                             y_normalized)


def compute_similarity(
        x: torch.Tensor,
        y: torch.Tensor,
        similarity_type: str = "cosine",
        verbose: Verbose = Verbose.INFO,
        **kwargs
) -> torch.Tensor:
    """
    Compute the similarity between two tensors.

    Args:
        x (torch.Tensor):
            The first tensor.
        y (torch.Tensor):
            The second tensor.
        similarity_type (str):
            The type of similarity to compute. Default: "cosine".
        verbose (Verbose):
            The verbosity level. Default: Verbose.INFO.
        **kwargs:
            Additional arguments for the similarity computation.

    Returns:
        torch.Tensor:
            The similarity between the two tensors.
    """

    if similarity_type == "cosine":
        return compute_cosine(x, y, dim=kwargs["dim"], verbose=verbose)
    else:
        raise Exception(f"Similarity type '{similarity_type}' not supported.")


def sort_columns(
        tensor: torch.Tensor,
        list_of_indices: list,
        verbose: Verbose = Verbose.INFO
) -> torch.Tensor:
    """
    Sorts the columns of a matrix based on a list of indices.

    Args:
        tensor (torch.Tensor):
            The matrix.
        list_of_indices (list):
            The list of indices to align the columns.
        verbose (Verbose):
            The verbosity level. Default: Verbose.INFO.

    Returns:
        torch.Tensor:
            The matrix with the columns aligned.
    """

    if verbose >= Verbose.DEBUG:
        print(f"Sorting the columns")

    sorted_tensor = tensor[:, list_of_indices]

    return sorted_tensor


def sort_rows(
        tensor: torch.Tensor,
        list_of_indices: list,
        verbose: Verbose = Verbose.INFO
) -> torch.Tensor:
    """
    Sorts the rows of a matrix based on a list of indices.

    Args:
        tensor (torch.Tensor):
            The matrix.
        list_of_indices (list):
            The list of indices to align the rows.
        verbose (Verbose):
            The verbosity level. Default: Verbose.INFO.

    Returns:
        torch.Tensor:
            The matrix with the rows aligned.
    """

    if verbose >= Verbose.DEBUG:
        print(f"Sorting the rows")

    sorted_tensor = tensor[list_of_indices, :]

    return sorted_tensor


def compute_indices_sorting(
        layers_in_block_1: list[AnalysisTensorWrapper],
        layers_in_block_2: list[AnalysisTensorWrapper],
        layer_index_for_similarity_1: int,
        layer_index_for_similarity_2: int,
        axis: int,
        similarity_type: str = "cosine",
        verbose: Verbose = Verbose.INFO
) -> [list, dict[torch.Tensor]]:
    """
    Computes the indices for the sorting of the elements of the layers in block 2 to be as close as possible to the
    layers in block 1.

    Args:
        layers_in_block_1 (list[AnalysisTensorWrapper]):
            The list of layers in block 1.
        layers_in_block_2 (list[AnalysisTensorWrapper]):
            The list of layers in block 2.
        layer_index_for_similarity_1 (int):
            The index of the layer in block 1 to use for the similarity computation.
        layer_index_for_similarity_2 (int):
            The index of the layer in block 2 to use for the similarity computation.
        axis (int):
            The axis to align the elements.
        similarity_type (str):
            The type of similarity to use to align. Default: "cosine".
        verbose (Verbose):
            The verbosity level. Default: Verbose.INFO.

    Returns:
        list:
            The list of indices to reorder the layers in block 2.
        dict[torch.Tensor]:
            The matrix of similarities between the layers in block 1 and block 2.


    Raises:
        Exception:
            If the axis is not supported.
    """

    layer_for_similarity_1 = layers_in_block_1[layer_index_for_similarity_1].get_tensor()
    layer_for_similarity_2 = layers_in_block_2[layer_index_for_similarity_2].get_tensor()

    similarities = cosine_similarity_all_couples_of_components(
        layer_for_similarity_1,
        layer_for_similarity_2,
        between="rows" if axis == 0 else "columns",
    )

    """
    if axis == 0:
        pass
    elif axis == 1:
        layer_for_similarity_1 = layer_for_similarity_1.transpose(0, 1)
        layer_for_similarity_2 = layer_for_similarity_2.transpose(0, 1)
    else:
        raise Exception(f"Axis '{axis}' not supported.")

    similarities = torch.zeros(layer_for_similarity_1.shape[0], layer_for_similarity_2.shape[0])
    for i, vector_1 in enumerate(layer_for_similarity_1):
        for j, vector_2 in enumerate(layer_for_similarity_2):
            if verbose > Verbose.INFO:
                print(f"Computing similarity between "
                      f"{'row' if axis == 0 else 'column' if axis == 1 else '???'} {i} and "
                      f"{'row' if axis == 0 else 'column' if axis == 1 else '???'} {j}.")
                print()
            similarities[i, j] = compute_similarity(
                vector_1,
                vector_2,
                similarity_type=similarity_type,
                verbose=verbose
            ).item()
    """
    similarities_copy = similarities.clone()

    # Computing the ordering of the vectors of the matrices in the second block to have the maximum possible similarity
    ordering = []
    for similarity_row in similarities.transpose(0, 1):
        minimum = torch.min(similarity_row)
        while torch.argmax(similarity_row) in ordering:
            if verbose > Verbose.INFO:
                indexed_values = list(enumerate(similarity_row))
                sorted_indexed_values = sorted(indexed_values, key=lambda x: x[1])
                print("Sorted Values:", [index for index, value in sorted_indexed_values[:10]])
                print("Sorted Indices:", [value for index, value in sorted_indexed_values[:10]])

            similarity_row[torch.argmax(similarity_row)] = minimum - 1

        ordering.append(torch.argmax(similarity_row))

        if verbose > Verbose.INFO:
            print()

    if verbose > Verbose.INFO:
        print(f"New ordering: {ordering}")

    return ordering, similarities_copy


def sort_elements_in_layers(
        list_of_subsequent_matrices: list[AnalysisTensorWrapper],
        list_of_indices: list,
        axes: list,
        verbose: Verbose = Verbose.INFO
) -> list:
    """
    Sorts the elements in the layers based on a list of indices.

    Args:
        list_of_subsequent_matrices (list):
            The list of subsequent matrices.
        list_of_indices (list):
            The list of the lists of indices to align the elements.
        axes (list):
            The list of axes to align the elements.
        verbose (Verbose):
            The verbosity level. Default: Verbose.INFO.

    Returns:
        list:
            The aligned matrices.

    Raises:
        Exception:
            If the axis is not supported.
    """

    reordered_matrices = []

    for index in range(len(list_of_subsequent_matrices)):
        layer = list_of_subsequent_matrices[index]
        if axes[index] == 0:
            sorted_tensor = sort_rows(layer.get_tensor(), list_of_indices, verbose=verbose)
        elif axes[index] == 1:
            sorted_tensor = sort_columns(layer.get_tensor(), list_of_indices, verbose=verbose)
        else:
            raise Exception(f"Axis '{axes[index]}' not supported.")

        if verbose > Verbose.INFO:
            print(f"Original tensor: {layer.get_tensor()[0, :10]}")
            print(f"Indices: {list_of_indices[:10]}")
            print(f"Sorted tensor: {sorted_tensor[0, :10]}")

        reordered_matrices.append(
            AnalysisTensorWrapper(
                sorted_tensor,
                name=layer.get_name(),
                label=layer.get_label(),
                path=layer.get_path(),
                block_index=layer.get_block_index(),
                layer=layer.get_layer()
            )
        )

    return reordered_matrices


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

    def compute_delta_matrices(
            self,
            minuend_matrices: list[AnalysisTensorWrapper],
            subtrahend_matrices: list[AnalysisTensorWrapper],
    ) -> list[AnalysisTensorWrapper]:
        """
        Computes the delta between two lists of matrices.

        Args:
            minuend_matrices (list):
                List of minuend matrices.
            subtrahend_matrices (list):
                List of subtrahend matrices.
            verbose (Verbose):
                The verbosity level. Defaults to Verbose.SILENT.

        Returns:
            list[AnalysisTensorWrapper]:
                List of delta matrices.
        """

        self.log("Computing delta matrices...")

        delta_matrices = []

        for i in range(len(minuend_matrices)):
            minuend_matrix = minuend_matrices[i].get_tensor()
            subtrahend_matrix = subtrahend_matrices[i].get_tensor()

            delta_matrix = minuend_matrix - subtrahend_matrix
            delta_matrices.append(
                AnalysisTensorWrapper(
                    delta_matrix,
                    name=minuend_matrices[i].get_name(),
                    label=str(minuend_matrices[i].get_label()) + " - " + str(subtrahend_matrices[i].get_label()),
                    block_index=minuend_matrices[i].get_block_index()
                )
            )

        return delta_matrices

    @override
    def _run_experiment(
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
        store_interval = config.get("store_interval") if config.contains("store_interval") else 10

        if self.get_data() is not None:
            self.log("Trying to load previous results...")
            # Loading the data
            original_tensor_wrappers, sorted_layers_deltas, objective_function_stats_dict = self.get_data()
            model = None
            self.log("Previous results loaded.")
        else:
            # Initializing the data structures
            original_tensor_wrappers = AnalysisTensorDict()
            sorted_layers_deltas = AnalysisTensorDict()
            objective_function_stats_dict = {}

            # Loading the model
            model = load_model_for_causal_lm(config)
            model.cpu()

            # Extracting the layers to analyze
            extract(
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

        tokenizer = transformers.AutoTokenizer.from_pretrained(config.get("model_id"))

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
            #sorted_layers_in_block_2 = copy.deepcopy(layers_in_block_2)
            self.log(f"Layers sorted.")
            #sorted_layers_in_block_2 = layers_in_block_2

            # Subtracting the layers of block and the sorted layers of another block
            delta_matrices = self.compute_delta_matrices(layers_in_block_1, sorted_layers_in_block_2)

            # Processing the delta matrices of the layers based on the specific operations that the analysis performs
            for delta_matrix in delta_matrices:
                self.process_tensor_wrappers(delta_matrix)
            #objective_function_stats = {"final_sorting_resettable_elements": 42, "initial_sorting_resettable_elements": 2018} # TESTING
            self.log("Delta matrices computed and processed.")

            key = (tuple(tuple(layer.get_path()) for layer in layers_in_block_1), tuple(tuple(layer.get_path()) for layer in layers_in_block_2))
            sorted_layers_deltas.set_tensor(key, delta_matrices)

            # Preparing the model for evaluation
            benchmark_id = config.get("benchmark_id")
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
                benchmark_id=benchmark_id,
                benchmark_evaluation_args=config.get("evaluation_args")[benchmark_id] if config.contains("evaluation_args") else {},
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
            benchmark_evaluation_args: dict,
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
            benchmark_evaluation_args (dict):
                The arguments to pass to the evaluation function of the benchmark.
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
        self.log(f"Evaluation arguments: {benchmark_evaluation_args}.")
        results = evaluate_model_on_benchmark(model_wrapper.get_model(), tokenizer, benchmark_id, benchmark_evaluation_args, device)
        #results = {self.config.get("benchmark_id"): {"acc_norm,none": 0.5}} # Testing
        self.log(f"Results of the modified model: {results}.")
        print(f"Results of the modified model: {results}.")

        model_wrapper.reset_replacement()
        model = model_wrapper.get_model().cpu()
        print(model.device)

        return results, model

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
                save_path=os.path.join(config.get("directory_path"), "heatmap.pdf"),
                title=self.title + f"(Model: {config.get("model_id")})", axis_titles=self.axis_titles,
                x_title=self.x_title, y_title=self.y_title,
                x_labels=column_labels, y_labels=row_labels,
                fig_size=config.get("figure_size") if config.contains("figure_size") else (30, 30)
            )
        else:
            plot_heatmap_with_additional_row_column(
                statistics_delta_layers,
                values_rows_lists=statistics_original_layers, values_columns_lists=statistics_original_layers,
                save_path=os.path.join(config.get("directory_path"), "heatmap.pdf"),
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

        del concatenated_matrix_1, concatenated_matrix_2

        sorting_stats.update({
            "final_sorting_resettable_elements": torch.sum(torch.abs(final_delta_concatenated_matrices) < zero_threshold).item(),
            "manhattan_norm_final_sorting_resettable_elements": torch.sum(torch.abs(final_delta_concatenated_matrices) * (torch.abs(final_delta_concatenated_matrices) < zero_threshold)).item(),
            "frobenius_norm_final_sorting_resettable_elements": torch.norm(final_delta_concatenated_matrices * (torch.abs(final_delta_concatenated_matrices) < zero_threshold)).item()
        })

        del final_delta_concatenated_matrices
        torch.cuda.empty_cache()

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