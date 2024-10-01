import logging

import transformers

from exporch.utils.plot_utils import get_text_color, set_axis_labels

import torch

from exporch import Verbose
from matplotlib import pyplot as plt

import numpy as np

from redhunter.analysis.analysis_utils import AnalysisTensorWrapper


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