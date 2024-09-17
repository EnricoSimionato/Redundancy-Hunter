from __future__ import annotations

import os
import logging
import pickle as pkl
from typing import Any
import gc

import numpy as np

import torch

import re

from exporch import Config, get_available_device

from exporch.utils.causal_language_modeling import load_model_for_causal_lm, load_tokenizer_for_causal_lm
from exporch import evaluate_model_on_benchmark
from exporch.utils.plot_utils.heatmap import plot_heatmap

from redhunter.analysis.swapped_layers_redundancy_analysis_utils import LayerSwitchingWrapperModel

from exporch.experiment import benchmark_id_metric_name_mapping

def post_process_result_dictionary(
    input_dict: dict[str, dict[tuple[str, str], dict[str, dict[str, float]]]]
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, np.ndarray]]:
    """
    Post-processes the result dictionary to extract the unique grouped elements and the performance arrays for each
    task.

    Args:
        input_dict (dict[str, dict[tuple[str, str], dict[str, float]]]):
            The input dictionary containing the results for each task.

    Returns:
        tuple[dict[str, list[tuple[str, str]]], dict[str, list[tuple[str, str]]], dict[str, np.ndarray]]:
            A tuple containing the task-specific unique grouped elements for the first and second elements, and the
            performance arrays for each task
    """

    # Helper function to extract tuples from string
    def extract_tuples(s):
        return re.findall(r"\('(.*?)', '(.*?)', '(.*?)'\)", s)

    task_specific_first_elements = {}
    task_specific_second_elements = {}
    performance_arrays = {}

    # Collecting task-specific unique grouped elements and their order
    for task, task_dict in input_dict.items():
        first_groups = []
        second_groups = []

        # Collecting grouped elements for each task
        for (key1, key2), _ in task_dict.items():
            first_tuples = str(extract_tuples(key1))
            second_tuples = str(extract_tuples(key2))

            # Adding unique groups for the first and second elements
            if first_tuples not in first_groups:
                first_groups.append(first_tuples)
            if second_tuples not in second_groups:
                second_groups.append(second_tuples)

        task_specific_first_elements[task] = first_groups
        task_specific_second_elements[task] = second_groups

        performance_array = np.full((len(first_groups), len(second_groups)), np.nan)
        # Filling the performance array with the metrics for this task
        for (key1, key2), value in task_dict.items():
            first_tuples = str(extract_tuples(key1))
            second_tuples = str(extract_tuples(key2))

            # Finding the correct row and column by the group index
            row = first_groups.index(first_tuples)
            col = second_groups.index(second_tuples)
            performance_array[row, col] = value[task][benchmark_id_metric_name_mapping[task]]

        performance_arrays[task] = performance_array

    return task_specific_first_elements, task_specific_second_elements, performance_arrays


def perform_swapped_layers_redundancy_analysis(
        config: Config,
) -> None:
    logger = logging.getLogger(__name__)
    logger.info(f"Running perform_layer_redundancy_analysis_launcher in redundancy_hunter.py.")

    # Getting the parameters related to the paths from the configuration
    file_available, file_path, directory_path = [
        config.get(name)
        for name in ["file_available", "file_path", "directory_path"]
    ]

    # Load the data if the file is available, otherwise process the model
    if file_available:
        print(f"The file '{file_path}' is available.")
        with open(file_path, "rb") as f:
            data = pkl.load(f)
    else:
        data = None

    perform_layer_redundancy_analysis(config, data)


def perform_layer_redundancy_analysis(
        config: Config,
        data: Any = None,
) -> None:
    logger = logging.getLogger(__name__)
    logger.info(f"Running perform_layer_redundancy_analysis in redundancy_hunter.py.")

    fig_size = config.get("figure_size") if config.contains("figure_size") else (20, 20)
    benchmark_ids = config.get("benchmark_ids")

    """
    if data is None:
        start_from_benchmark = 0
        start_from_swap = 0
    """
    # Getting the parameters from the configuration
    device = get_available_device(config.get("device") if config.contains("device") else None, just_string=True)
    targets_lists = config.get("targets")
    num_layers = config.get("num_layers")
    evaluation_args = (config.get("evaluation_args")
                       if config.contains("evaluation_args")
                       else {benchmark_id: {} for benchmark_id in benchmark_ids})

    performance_dict = {benchmark_id: {} for benchmark_id in benchmark_ids}

    # Loading the model and the tokenizer
    base_model = load_model_for_causal_lm(config)
    logger.info(f"Model loaded.")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = load_tokenizer_for_causal_lm(config)
    logger.info(f"Tokenizer loaded.")

    # Setting the parameters for the layer switching
    destination_layer_path_source_layer_path_mapping_list = [
        {
            tuple(el if el != "layer_index" else f"{i}" for el in targets):
                tuple(el if el != "layer_index" else f"{j}" for el in targets) for targets in targets_lists
        } for i in range(num_layers) for j in range(num_layers) if (i != j or (i == 0 and j == 0))
    ]
    redundant_layer_path_source_layer_path_mapping_list = [
        {
            tuple(el if el != "layer_index" else f"{i}" for el in targets):
                tuple(el if el != "layer_index" else f"{i}" for el in targets) for targets in targets_lists
        } for i in range(num_layers) if i != 0
    ]
    original_model_layer_path_source_layer_path_mapping = {
        tuple(el if el != "layer_index" else f"{0}" for el in targets):
            tuple(el if el != "layer_index" else f"{0}" for el in targets) for targets in targets_lists
    }

    # Wrapping the model to move the layers
    model_wrapper = LayerSwitchingWrapperModel(base_model, None)
    logger.info(f"Model wrapped.")

    if data is not None:
        destination_layer_path_source_layer_path_mapping_list, already_created_performance_dict = data
        performance_dict.update(already_created_performance_dict)
        start_from_swap = {
            benchmark_id: len(performance_dict[benchmark_id].keys())
            if len(performance_dict[benchmark_id].keys()) < len(destination_layer_path_source_layer_path_mapping_list)
            else len(destination_layer_path_source_layer_path_mapping_list) for benchmark_id in benchmark_ids
        }
        logger.info(f"Previous data loaded.\nLoaded data: {performance_dict}")
        print("Previous data loaded.")
    else:
        start_from_swap = {benchmark_id: 0 for benchmark_id in benchmark_ids}

    for benchmark_id in benchmark_ids:
        logger.info(f"Starting the evaluation for the benchmark: {benchmark_id}, starting from: {start_from_swap[benchmark_id]}")
        print(f"Starting the evaluation for the benchmark: {benchmark_id}, starting from: {start_from_swap[benchmark_id]}")
        for destination_layer_path_source_layer_path_mapping in destination_layer_path_source_layer_path_mapping_list[start_from_swap[benchmark_id]:]:
            logger.info(f"Evaluating the variant destination_layer_path_source_layer_path_mapping: {destination_layer_path_source_layer_path_mapping}")
            print(f"Evaluating the variant destination_layer_path_source_layer_path_mapping: {destination_layer_path_source_layer_path_mapping}")

            model_wrapper.set_destination_layer_path_source_layer_path_mapping(destination_layer_path_source_layer_path_mapping)
            logger.info(f"Layers switched.")

            # Defining the evaluation parameters
            benchmark_evaluation_args = evaluation_args[benchmark_id] if benchmark_id in evaluation_args.keys() else {}
            logger.info(f"Chosen evaluation args: {evaluation_args}")

            logger.info(f"Starting the evaluation of the model on the device {model_wrapper.get_model().device}.")
            # Evaluating the model
            results = evaluate_model_on_benchmark(model_wrapper.get_model(), tokenizer, benchmark_id, benchmark_evaluation_args, device)
            logger.info(f"Results: {results}")

            performance_dict[benchmark_id][
                (str(destination_layer_path_source_layer_path_mapping.keys()),
                 str(destination_layer_path_source_layer_path_mapping.values()))
            ] = results
            logger.info(f"Performance dictionary updated with the results.")

            model_wrapper.reset_switch()
            logger.info(f"Layers reset.")

            with open(config.get("file_path"), "wb") as f:
                pkl.dump((destination_layer_path_source_layer_path_mapping_list, performance_dict), f)
            logger.info(f"Partial data stored.")

            torch.cuda.empty_cache()
            #gc.collect()

        redundant_performance_dict = {
            (str(redundant_layer_path_source_layer_path_mapping.keys()), str(redundant_layer_path_source_layer_path_mapping.values())): performance_dict[benchmark_id][str(original_model_layer_path_source_layer_path_mapping.keys()), str(original_model_layer_path_source_layer_path_mapping.values())]
            for redundant_layer_path_source_layer_path_mapping in redundant_layer_path_source_layer_path_mapping_list
        }
        performance_dict[benchmark_id].update(redundant_performance_dict)

        data = (destination_layer_path_source_layer_path_mapping_list, performance_dict)
        logger.info(f"Trying to store the data for benchmark {benchmark_id}...")
        # Saving the data
        with open(config.get("file_path"), "wb") as f:
            pkl.dump(data, f)

        torch.cuda.empty_cache()
        #gc.collect()

    data = (destination_layer_path_source_layer_path_mapping_list, performance_dict)

    logger.info(f"Trying to store all the data...")
    # Saving the data
    with open(config.get("file_path"), "wb") as f:
        pkl.dump(data, f)
    logger.info("All data stored.")

    destination_layer_path_source_layer_path_mapping_list, destination_layer_path_source_layer_path_mapping_list = data
    rows_labels_list, columns_labels_list, post_processed_results_list = post_process_result_dictionary(
        destination_layer_path_source_layer_path_mapping_list
    )

    for benchmark_id in post_processed_results_list.keys():
        logger.info(f"Printing the results for task: {benchmark_id}")
        plot_heatmap(
            [[post_processed_results_list[benchmark_id]]],
            os.path.join(config.get("directory_path"), f"heatmap_{benchmark_id}.png"),
            f"Results for the model {config.get('model_id').split('/')[-1]} on the task {benchmark_id}",
            x_title="Substituted layers",
            y_title="Source layers",
            x_labels=[columns_labels_list[benchmark_id]],
            y_labels=[rows_labels_list[benchmark_id]],
            fig_size=fig_size,
            precision=4
        )
