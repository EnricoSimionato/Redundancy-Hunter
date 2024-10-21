import os
import pickle as pkl
from tqdm import tqdm
import logging

import numpy as np

from exporch import Config, Verbose

from exporch.utils.causal_language_modeling import load_model_for_causal_lm
from exporch.utils.plot_utils import plot_heatmap

from redhunter.analysis.analysis_utils import AnalysisTensorDict, extract, compute_max_possible_rank


def perform_original_layers_rank_analysis(
        configuration: Config
) -> None:
    """
    Perform the rank analysis of the original layers of a model.

    Args:
        configuration:
            The configuration object containing the necessary information.
    """

    logging.basicConfig(filename=os.path.join(configuration.get("directory_path"), "logs.log"), level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Running perform_simple_initialization_analysis in matrix_initialization_analysis.py.")

    # Getting the parameters related to the paths from the configuration
    logger.info(f"Getting the parameters related to the paths from the configuration")
    file_available = configuration.get("file_available")
    file_path = configuration.get("file_path")
    directory_path = configuration.get("directory_path")
    file_name = configuration.get("file_name")
    file_name_no_format = file_name.split(".")[0]
    logger.info(f"Information retrieved")
    explained_variance_threshold = (
        configuration.get("explained_variance_threshold") if configuration.contains("explained_variance_threshold") else 0
    )
    singular_values_threshold = (
        configuration.get("singular_values_threshold") if configuration.contains("singular_values_threshold") else 0
    )
    verbose = configuration.get_verbose()

    if file_available:
        print(f"File already exists. Loading data from '{file_path}'...")

        # Loading the data
        with open(file_path, "rb") as f:
            data = pkl.load(f)
    else:
        # Loading the model
        model = load_model_for_causal_lm(configuration)

        # Extracting the layers to analyze
        extracted_layers = []
        extract(
            model,
            configuration.get("targets"),
            extracted_layers,
            black_list=configuration.get("black_list"),
            verbose=verbose
        )
        verbose.print("Layers extracted", Verbose.SILENT)

        # Grouping the extracted layers by block
        extracted_layers_grouped_by_label = {}
        for extracted_layer in extracted_layers:
            label = extracted_layer.get_label()
            if label not in extracted_layers_grouped_by_label.keys():
                extracted_layers_grouped_by_label[label] = []

            extracted_layers_grouped_by_label[label].append(extracted_layer)
        verbose.print("Layers grouped by label", Verbose.SILENT)

        pre_analyzed_tensors = AnalysisTensorDict()

        for label in tqdm(extracted_layers_grouped_by_label.keys()):
            for matrix in extracted_layers_grouped_by_label[label]:
                matrix.compute_singular_values()
                pre_analyzed_tensors.append_tensor(
                    label,
                    matrix
                )
                verbose.print(f"Singular values for {matrix.get_name()} - {matrix.get_label()} extracted", Verbose.DEBUG)

        data = pre_analyzed_tensors

    pre_analyzed_tensors = data
    matrix_types = pre_analyzed_tensors.get_unique_positional_keys(position=0)
    number_of_blocks = len(pre_analyzed_tensors.get_tensor_list(matrix_types[0]))
    ranks = np.zeros(
        (
            len(matrix_types),
            number_of_blocks
        )
    )
    relative_ranks = np.zeros(
        (
            len(matrix_types),
            number_of_blocks
        )
    )

    analyzed_tensors = AnalysisTensorDict()
    for index_label, label in tqdm(enumerate(matrix_types)):
        for index_block, matrix in enumerate(pre_analyzed_tensors.get_tensor_list(label)):
            rank = matrix.get_rank(explained_variance_threshold, singular_values_threshold, False)
            ranks[index_label, index_block] = rank
            rank = matrix.get_rank(explained_variance_threshold, singular_values_threshold, True)
            relative_ranks[index_label, index_block] = rank

            analyzed_tensors.append_tensor(
                label,
                matrix
            )

    # Saving the matrix wrappers of the layers used to perform the analysis
    with open(file_path, "wb") as f:
        pkl.dump(
            analyzed_tensors,
            f
        )
    if verbose > Verbose.SILENT:
        print(f"Data saved")

    heatmap_name = configuration.get("heatmap_name") if configuration.contains("heatmap_name") else "heatmap"
    heatmap_name += "_expvar_" + str(explained_variance_threshold).replace('.', '_')
    plot_heatmap(
        ranks,
        interval={"min": 0, "max": compute_max_possible_rank(analyzed_tensors)},
        title="Rank analysis of the matrices of the model" + f" (explained variance threshold: {explained_variance_threshold})",
        x_title="Block indexes",
        y_title="Layer type",
        columns_labels=list(range(number_of_blocks)),
        rows_labels=matrix_types,
        figure_size=configuration.get("figure_size") if configuration.contains("figure_size") else (10, 24),
        save_path=configuration.get("directory_path"),
        heatmap_name=heatmap_name,
        show=configuration.get("show") if configuration.contains("show") else True,
    )

    heatmap_name += "_relative"
    plot_heatmap(
        relative_ranks,
        interval={"min": 0, "max": 1},
        title="Relative rank analysis of the matrices of the model" + f" (explained variance threshold: {explained_variance_threshold})",
        x_title="Block indexes",
        y_title="Layer type",
        columns_labels=list(range(number_of_blocks)),
        rows_labels=matrix_types,
        figure_size=configuration.get("figure_size") if configuration.contains("figure_size") else (10, 24),
        save_path=configuration.get("directory_path"),
        heatmap_name=heatmap_name,
        show=configuration.get("show") if configuration.contains("show") else True,
    )
