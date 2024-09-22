from __future__ import annotations

import os
import pickle as pkl
import logging
from tqdm import tqdm

from matplotlib import pyplot as plt

import numpy as np

import torch

from exporch import Config, Verbose

from exporch.utils.plot_utils.heatmap import get_text_color, set_axis_labels, plot_heatmap_with_additional_row_column

from redhunter.analysis.analysis_utils import AnalysisTensorWrapper, AnalysisTensorDict, extract_based_on_path
from redhunter.analysis.delta_layers_rank_analysis import compute_delta_matrices



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

    return torch.matmul(x_normalized, y_normalized.T) if between == "rows" else torch.matmul(x_normalized.T, y_normalized)


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
                print("Sorted Indices:",  [value for index, value in sorted_indexed_values[:10]])

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


def perform_sorted_layers_rank_analysis(
        config: Config,
) -> None:
    """
    Performs the aligned layers rank analysis.

    Args:
        config (Config):
            The configuration of the experiment.
    """

    logging.basicConfig(filename=os.path.join(config.get("directory_path"), "logs.log"), level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Running perform_sorted_layers_rank_analysis in sorted_layers_rank_analysis.py.")

    # Getting the parameters related to the paths from the configuration
    logger.info(f"Getting the parameters related to the paths from the configuration")
    file_available, file_path, directory_path, file_name, file_name_no_format = [
        config.get(name)
        for name in ["file_available", "file_path", "directory_path", "file_name", "file_name_no_format"]
    ]
    logger.info(f"Information retrieved")

    # Getting the parameters related to the analysis from the configuration
    logger.info(f"Getting the parameters related to the analysis from the configuration")
    verbose = config.get_verbose()
    fig_size = config.get("figure_size") if config.contains("figure_size") else (20, 20)
    explained_variance_threshold = (
        config.get("explained_variance_threshold") if config.contains("explained_variance_threshold") else 0
    )
    singular_values_threshold = (
        config.get("singular_values_threshold") if config.contains("singular_values_threshold") else 0
    )

    if file_available:
        print(f"File already exists. Loading data from '{file_path}'...")

        # Loading the data
        with open(file_path, "rb") as f:
            extracted_layers_grouped_by_block, pre_analyzed_tensors = pkl.load(f)
    else:
        # Loading the model
        model = load_original_model_for_causal_lm(config)

        # Extracting the layers to analyze
        extracted_layers = []
        targets = config.get("targets")
        black_list = config.get("black_list")
        extract_based_on_path(model, targets, extracted_layers, black_list, verbose=verbose)
        verbose.print("Layers extracted", Verbose.SILENT)

        # Grouping the extracted layers by block
        extracted_layers_grouped_by_block = {}
        for extracted_layer in extracted_layers:
            block_index = extracted_layer.get_block_index()
            if block_index not in extracted_layers_grouped_by_block.keys():
                extracted_layers_grouped_by_block[block_index] = []

            extracted_layers_grouped_by_block[block_index].append(extracted_layer)

        verbose.print("Layers grouped by block", Verbose.SILENT)
        verbose.print(str(extracted_layers_grouped_by_block.keys()), Verbose.DEBUG)
        verbose.print(str(extracted_layers_grouped_by_block), Verbose.DEBUG)

        # Computing the ranks of the difference of the layers in the blocks
        similarity_guide_name = config.get("similarity_guide_name")
        pre_analyzed_tensors = AnalysisTensorDict()
        similarity_stats = {}
        for block_index_1 in tqdm(extracted_layers_grouped_by_block.keys()):
            for layer in extracted_layers_grouped_by_block[block_index_1]:
                layer.compute_singular_values()
            for block_index_2 in extracted_layers_grouped_by_block.keys():
                logger.info(f"Computing the similarity between block {block_index_1} and block {block_index_2}.")
                if block_index_1 != block_index_2:
                    layers_in_block_1 = extracted_layers_grouped_by_block[block_index_1]
                    layers_in_block_2 = extracted_layers_grouped_by_block[block_index_2]

                    similarity_guide_index = None
                    for index, element in enumerate(layers_in_block_1):
                        if similarity_guide_name in element.get_label():
                            similarity_guide_index = index
                            break
                    if similarity_guide_index is None:
                        raise Exception(f"Layer '{similarity_guide_name}' not found in block {block_index_1}.")

                    # Sorting the layers in block 2 to match the layers in block 1 in order to be as similar as possible

                    # Computing the ordering
                    ordering, similarities = compute_indices_sorting(
                        layers_in_block_1,
                        layers_in_block_2,
                        layer_index_for_similarity_1=similarity_guide_index,
                        layer_index_for_similarity_2=similarity_guide_index,
                        axis=1 if similarity_guide_index == len(layers_in_block_1) - 1 else 0,
                        verbose=verbose
                    )
                    verbose.print(f"\nIndices computed {ordering[:10]}", Verbose.DEBUG)

                    # Using the ordering to sort the vectors in the matrices of block 2
                    sorted_layers_in_block_2 = sort_elements_in_layers(
                        layers_in_block_2,
                        ordering,
                        [1 if index == len(layers_in_block_2) - 1 else 0 for index in range(len(layers_in_block_2))],
                        verbose=verbose
                    )
                    verbose.print(f"\nLayers sorted based on similarity", Verbose.DEBUG)

                    initial_similarities = [
                        cosine_similarity_respective_couples_of_components(
                            layers_in_block_1[index].get_tensor(),
                            layers_in_block_2[index].get_tensor(),
                            between="rows" if index == len(layers_in_block_1) - 1 else "columns"
                        )
                        for index in range(len(layers_in_block_1))
                    ]
                    final_similarities = [
                        cosine_similarity_respective_couples_of_components(
                            layers_in_block_1[index].get_tensor(),
                            sorted_layers_in_block_2[index].get_tensor(),
                            between="rows" if index == len(layers_in_block_1) - 1 else "columns"
                        )
                        for index in range(len(layers_in_block_1))
                    ]
                    similarity_stats[(block_index_1, block_index_2)] = {
                        "mean maximum similarity": torch.mean(torch.max(similarities, dim=1).values).item(),
                        "initial mean similarities": [torch.mean(similarity).item() for similarity in initial_similarities],
                        "initial maximum similarities": [torch.max(similarity).item() for similarity in initial_similarities],
                        "final mean similarities": [torch.mean(similarity).item() for similarity in final_similarities],
                        "final maximum similarities": [torch.max(similarity).item() for similarity in final_similarities],
                    }

                    # Computing the delta matrices between the layers in block 1 and the sorted layers in block 2
                    delta_matrices = compute_delta_matrices(layers_in_block_1, sorted_layers_in_block_2, verbose=verbose)

                    # Computing the singular values of the delta matrices of the layers in the two different blocks
                    for delta_matrix in delta_matrices:
                        delta_matrix.compute_singular_values()

                    """
                    _0norm_hand_delta_init = (
                                layers_in_block_1[0].get_tensor() - layers_in_block_2[0].get_tensor()).norm()
                    _0norm_hand_add_init = (
                                layers_in_block_1[0].get_tensor() + layers_in_block_2[0].get_tensor()).norm()
                    _0norm_hand_add = (layers_in_block_1[0].get_tensor() + sorted_layers_in_block_2[0].get_tensor()).norm()

                    _1norms1 = [layer.get_norm() for layer in layers_in_block_1]
                    _1norms2 = [layer.get_norm() for layer in sorted_layers_in_block_2]
                    _1norm_hand_delta_init = (
                                layers_in_block_1[1].get_tensor() - layers_in_block_2[1].get_tensor()).norm()
                    _1norm_hand_add_init = (
                                layers_in_block_1[1].get_tensor() + layers_in_block_2[1].get_tensor()).norm()

                    _1norms_delta_matrices = [delta_matrix.get_norm() for delta_matrix in delta_matrices]
                    """
                    pre_analyzed_tensors.append_tensor((block_index_1, block_index_2), delta_matrices)

        # Saving the matrix wrappers of the layers used to perform the analysis
        with open(file_path, "wb") as f:
            pkl.dump((extracted_layers_grouped_by_block, pre_analyzed_tensors), f)

        # Saving the similarity statistics
        with open(os.path.join(directory_path, "similarity_stats.pkl"), "wb") as f:
            pkl.dump(similarity_stats, f)

    """
    try:
        with open(similarities_path, "rb") as file:
            computed_similarities = pkl.load(file)
            for key, similarities in computed_similarities.items():
                block_index_1, block_index_2 = key
                print(f"Block {block_index_1} and block {block_index_2} have mean (on the rows) highest similarity "
                      f"{torch.mean(torch.max(similarities, dim=1).values).item()}.")
                print(f"Block {block_index_1} and block {block_index_2} have mean (on the columns) highest "
                      f"similarity {torch.mean(torch.max(similarities, dim=0).values).item()}.")
    except FileNotFoundError:
        print("No file containing the similarities found.")
    """

    # Retrieving the indices of the blocks of the model
    blocks_indexes_1 = pre_analyzed_tensors.get_unique_positional_keys(position=0, sort=True)
    row_column_labels = ["\n".join([tensor_wrapper.get_path() for tensor_wrapper in extracted_layers_grouped_by_block[block_index]]) for block_index in blocks_indexes_1]
    try:
        with open(os.path.join(directory_path, "similarity_stats.pkl"), "rb") as f:
            similarity_stats = pkl.load(f)
            for key in similarity_stats.keys():
                print(f"The mean of the maximum similarities between the layers of block {key[0]} and block {key[1]} is"
                      f" {similarity_stats[key]['mean maximum similarity']}.")
                logger.info(f"The mean of the maximum similarities between the layers of block {key[0]} and block "
                            f"{key[1]} is {similarity_stats[key]['mean maximum similarity']}.")
            plot_similarity_stats(
                similarity_stats, os.path.join(directory_path, "similarity_stats.png"),
                "Similarity statistics between the layers of the different blocks before and after sorting",
                ("Initial statistics", "Final statistics"),
                "Name of the block", "Name of the block", row_column_labels, row_column_labels,
                fig_size
            )
    except FileNotFoundError:
        print("No file containing the statistics about similarities found.")

    # Retrieving the indices of the blocks of the model
    """
    [
        [tensor1, tensor2, ...], # List for the first matrix
        [tensor1, tensor2, ...] # List for the second matrix
    ]
    """
    sample_wrapper_list = pre_analyzed_tensors.get_tensor_list(pre_analyzed_tensors.get_keys()[0])
    norms = [[np.zeros((len(blocks_indexes_1), len(blocks_indexes_1))), ] for _ in range(len(sample_wrapper_list))]
    norms_original_layers = [[np.zeros(len(blocks_indexes_1)), ] for _ in range(len(sample_wrapper_list))]
    ranks = [[np.zeros((len(blocks_indexes_1), len(blocks_indexes_1))), np.zeros((len(blocks_indexes_1), len(blocks_indexes_1)))] for _ in range(len(sample_wrapper_list))]
    ranks_original_layers = [[np.zeros(len(blocks_indexes_1)), np.zeros(len(blocks_indexes_1))] for _ in range(len(sample_wrapper_list))]

    # Performing the rank analysis of the difference between the matrices of the model aligned based on the similarity
    analyzed_tensors = AnalysisTensorDict()
    for block_index_1 in tqdm(blocks_indexes_1):
        # Computing the norms and the ranks of the original layers
        for index, layer in enumerate(extracted_layers_grouped_by_block[block_index_1]):
            norms_original_layers[index][0][block_index_1] = layer.get_norm()
            ranks_original_layers[index][0][block_index_1] = layer.get_rank(explained_variance_threshold, singular_values_threshold, False)
            ranks_original_layers[index][1][block_index_1] = layer.get_rank(explained_variance_threshold, singular_values_threshold, True)

        filtered_dictionary_all_delta_matrices = pre_analyzed_tensors.filter_by_positional_key(block_index_1, 0)
        blocks_indexes_2 = filtered_dictionary_all_delta_matrices.get_unique_positional_keys(1, True)
        for block_index_2 in blocks_indexes_2:
            # Getting the delta matrices
            delta_matrices = filtered_dictionary_all_delta_matrices.get_tensor_list((block_index_1, block_index_2))

            # Computing the norms of the delta matrices
            for index, delta_matrix in enumerate(delta_matrices):
                norms[index][0][int(block_index_1), int(block_index_2)] = delta_matrix.get_norm()

                # Computing the rank of the difference matrix given the explained variance threshold and the
                # singular values threshold
                ranks[index][0][int(block_index_1), int(block_index_2)] = delta_matrix.get_rank(explained_variance_threshold, singular_values_threshold, False)
                ranks[index][1][int(block_index_1), int(block_index_2)] = delta_matrix.get_rank(explained_variance_threshold, singular_values_threshold, True)

            analyzed_tensors.append_tensor((block_index_1, block_index_2), delta_matrices)

    # Saving the original extracted layers and the layers on which the analysis has been performed
    with open(file_path, "wb") as f:
        pkl.dump((extracted_layers_grouped_by_block, analyzed_tensors), f)

    row_column_labels = list(
        zip(*[[tensor_wrapper.get_path() for tensor_wrapper in extracted_layers_grouped_by_block[block_index]] for
              block_index in blocks_indexes_1]))
    row_column_labels = [["Norm of the original layers"] + list(label_list) for label_list in row_column_labels]

    # Plotting the norms of the original layers and the norms of the delta matrices
    plot_heatmap_with_additional_row_column(
        norms, norms_original_layers, norms_original_layers, os.path.join(directory_path, "norms.png"),
        "Norms of the difference of the sorted layers", None, "Name of the block",
        "Name of the block", row_column_labels, row_column_labels, fig_size
    )

    # Plotting the ranks of the difference matrices
    plot_heatmap_with_additional_row_column(
        ranks, ranks_original_layers, ranks_original_layers, os.path.join(directory_path, f"ranks_explained_variance_threshold_{explained_variance_threshold:.2f}.png"),
        "Ranks (absolute and relative) of the difference of the sorted layers", None, "Name of the block",
        "Name of the block", row_column_labels, row_column_labels, fig_size
    )


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
