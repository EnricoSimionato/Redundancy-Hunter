from __future__ import annotations

from abc import abstractmethod
import copy
import gc
import logging
import os
import re
from typing import Any, override

import numpy as np

import torch

from exporch import Config, get_available_device

from exporch.utils.causal_language_modeling import load_model_for_causal_lm, load_tokenizer_for_causal_lm
from exporch.experiment import benchmark_id_metric_name_mapping, evaluate_model_on_benchmark
from exporch.utils.plot_utils.heatmap import plot_heatmap

from redhunter.analysis.layer_replacement_analysis_utils import LayerReplacingModelWrapper
from redhunter.analysis_experiment import AnalysisExperiment


class LayerReplacementAnalysis(AnalysisExperiment):
    """
    The class for the layer replacement analysis experiments.
    """

    mandatory_keys = ["num_layers"]

    def _perform_analysis(
            self
    ) -> None:
        """
        Performs the analysis.
        It performs the layer replacement analysis experiment.
        """

        self._perform_layer_replacement_analysis()

    def _perform_layer_replacement_analysis(
            self
    ) -> None:
        """
        Performs the layer replacement analysis experiment.
        """

        gc.collect()
        config = self.config
        self.log(f"Starting the analysis.")

        # Setting the parameters for the layer switching
        destination_layer_path_source_layer_path_mapping_list = self.get_layers_replacement_mapping()

        # Initializing the dictionary to store the performance results
        benchmark_ids = config.get("benchmark_ids")
        performance_dict = {benchmark_id: {} for benchmark_id in benchmark_ids}

        remaining_destination_layer_path_source_layer_path_mapping_list = {
            benchmark_id: copy.deepcopy(destination_layer_path_source_layer_path_mapping_list) for benchmark_id in benchmark_ids
        }
        if self.data is not None:
            _, already_created_performance_dict = self.data
            performance_dict.update(already_created_performance_dict)

            remaining_destination_layer_path_source_layer_path_mapping_list = self.get_remaining_mappings_to_be_analyzed(
                remaining_destination_layer_path_source_layer_path_mapping_list, self.data[1]
            )
            if all(len(mapping) == 0 for mapping in remaining_destination_layer_path_source_layer_path_mapping_list.values()):
                self.log(f"Computation is not needed, the analysis has already been performed.")
                return

        # Getting the parameters from the configuration
        device = get_available_device(config.get("device") if config.contains("device") else "cpu", just_string=True)
        evaluation_args = (config.get("evaluation_args") if config.contains("evaluation_args")
                           else {benchmark_id: {} for benchmark_id in benchmark_ids})

        # Loading the model and the tokenizer
        base_model = load_model_for_causal_lm(config)
        self.log(f"Model loaded.")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer = load_tokenizer_for_causal_lm(config)
        self.log(f"Tokenizer loaded.")
        gc.collect()

        # Wrapping the model to move the layers
        model_wrapper = LayerReplacingModelWrapper(base_model, None)
        self.log(f"Model wrapped.")
        gc.collect()

        for benchmark_id in remaining_destination_layer_path_source_layer_path_mapping_list.keys():
            logging.info("Evaluating the original model")
            print("Evaluating the original model")
            if ("original", "original") not in performance_dict[benchmark_id].keys():
                original_model_results = evaluate_model_on_benchmark(model_wrapper.get_model(), tokenizer, benchmark_id,
                                                      evaluation_args[benchmark_id], device)
                #original_model_results = {benchmark_id: {"acc,none": 0.7}} # Testing
                performance_dict[benchmark_id][("original", "original")] = original_model_results
                self.log(f"Results of the original model: {original_model_results}")
                print(f"Results of the original model: {original_model_results}")

            for destination_layer_path_source_layer_path_mapping in remaining_destination_layer_path_source_layer_path_mapping_list[benchmark_id]:
                self.log(f"Evaluating the variant destination_layer_path_source_layer_path_mapping: "
                            f"{destination_layer_path_source_layer_path_mapping}")
                print(f"Evaluating the variant destination_layer_path_source_layer_path_mapping: "
                      f"{destination_layer_path_source_layer_path_mapping}")

                model_wrapper.set_destination_layer_path_source_layer_path_mapping(
                    destination_layer_path_source_layer_path_mapping)
                self.log(f"Layers replaced.")

                # Defining the evaluation parameters
                benchmark_evaluation_args = evaluation_args[benchmark_id]
                self.log(f"Chosen evaluation args: {benchmark_evaluation_args}")

                # Evaluating the model
                self.log(f"Starting the evaluation of the model on the device {model_wrapper.get_model().device}.")
                results = evaluate_model_on_benchmark(model_wrapper.get_model(), tokenizer, benchmark_id,
                                                      benchmark_evaluation_args, device)
                #results = {benchmark_id: {"acc,none": 0.5}} # Testing
                self.log(f"Results: {results}")
                gc.collect()

                # The key in the performance dictionary is a tuple containing the overwritten layers as first element and
                # the ones used to overwrite the destination as second elements
                performance_dict[benchmark_id][
                    self.get_performance_dict_key_from_mapping(destination_layer_path_source_layer_path_mapping)
                ] = results
                self.log(f"Performance dictionary updated with the results.")

                model_wrapper.reset_replacement()
                self.log(f"Layers reset.")

                self.data = (destination_layer_path_source_layer_path_mapping_list, performance_dict)
                # Storing the data
                self.log(f"Trying to store the data for benchmark {benchmark_id}...")
                self.store_data()
                self.log(f"Partial data stored.")

                torch.cuda.empty_cache()
                gc.collect()

            self.data = (destination_layer_path_source_layer_path_mapping_list, performance_dict)

            # Storing the data
            self.log(f"Trying to store the data for benchmark {benchmark_id}...")
            self.store_data()
            self.log(f"Stored data up to benchmark {benchmark_id}.")
            torch.cuda.empty_cache()
            gc.collect()

        self.log("All data stored.")

        self.log("The analysis has been completed.")
        print("The analysis has been completed.")

    @abstractmethod
    def get_layers_replacement_mapping(
            self
    ) -> list[dict[tuple, tuple]]:
        """
        Returns the mapping on which the analysis has to be performed.
        In the list of mappings to be analyzed, each one is a dictionary where the keys are the paths to the layers to
        be replaced and the values are the paths to the layers that will replace them.

        Returns:
            list[dict[tuple, tuple]]:
                The list of mappings to be analyzed.
        """

        pass

    @staticmethod
    def get_remaining_mappings_to_be_analyzed(
            all_destination_layer_path_source_layer_path_mapping: dict[str, list[dict[tuple, tuple]]],
            already_created_performance_dict: dict[str, dict[tuple, tuple]],
    ) -> dict[str, list[dict[tuple, tuple]]]:
        """
        Returns the samples that are still to be analyzed.

        Args:
            all_destination_layer_path_source_layer_path_mapping (dict[str, list[dict[tuple, tuple]]]):
                The dictionary containing the mappings to be analyzed.
            already_created_performance_dict (dict[str, dict[tuple, tuple]]):
                The dictionary containing the already created performance results.
        """

        remaining_destination_layer_path_source_layer_path_mapping = {}
        for benchmark_id in all_destination_layer_path_source_layer_path_mapping:
            if benchmark_id not in already_created_performance_dict.keys():
                remaining_destination_layer_path_source_layer_path_mapping[benchmark_id] = all_destination_layer_path_source_layer_path_mapping[benchmark_id]
            else:
                remaining_destination_layer_path_source_layer_path_mapping[benchmark_id] = [
                    mapping for mapping in all_destination_layer_path_source_layer_path_mapping[benchmark_id]
                    if LayerReplacementAnalysis.get_performance_dict_key_from_mapping(mapping) not in already_created_performance_dict[benchmark_id].keys()
                ]

        return remaining_destination_layer_path_source_layer_path_mapping

    @staticmethod
    def get_performance_dict_key_from_mapping(
            mapping: dict[tuple, tuple]
    ) -> tuple[str, str]:
        """
        Returns the key for the performance dictionary from the mapping.

        Args:
            mapping (dict[tuple, tuple]):
        Returns:
            tuple[str, str]:
                The key for the performance dictionary from the mapping.
        """

        # Sorting the items based on keys
        items = list(mapping.items())
        items.sort(key=lambda x: str(x[0]))
        keys = tuple(item[0] for item in items)
        values = tuple(item[1] for item in items)

        return str(keys), str(values)

    def _plot_results(
            self,
            config: Config,
            data: Any
    ) -> None:
        """
        Plots the results of the analysis.

        Args:
            config (Config):
                The configuration object containing the parameters of the experiment.
            data (Any):
                The data to be plotted.
        """

        if data is None:
            self.log("The data must be provided to plot the results of the analysis.")
            raise ValueError("The data must be provided to plot the results of the analysis.")

        fig_size = config.get("figure_size") if config.contains("figure_size") else (36, 36)
        destination_layer_path_source_layer_path_mapping_list, performance_dict = data

        # Extracting the results for the original model
        original_model_performance = {benchmark_id: all_benchmark_results.pop(("original", "original")) for benchmark_id, all_benchmark_results in performance_dict.items()}
        self.log(f"Original model performance: {original_model_performance}")

        # Plotting the results
        overwritten_layers_labels_row_list, duplicated_layers_labels_column_list, post_processed_results_list = self._format_result_dictionary_to_plot(performance_dict)
        for benchmark_id in post_processed_results_list.keys():
            self.log(f"Printing the results for task: {benchmark_id}")
            plot_heatmap(
                [[post_processed_results_list[benchmark_id]]],
                os.path.join(config.get("directory_path"), f"heatmap_{benchmark_id}.png"),
                f"Results for the model {config.get('model_id').split('/')[-1]} on the task {benchmark_id}",
                axis_titles=[f"Metric: {benchmark_id_metric_name_mapping[benchmark_id]}"],
                x_title="Duplicated layers labels",
                y_title="Overwritten layers labels",
                x_labels=[duplicated_layers_labels_column_list[benchmark_id]],
                y_labels=[overwritten_layers_labels_row_list[benchmark_id]],
                fig_size=fig_size,
                edge_color="white",
                precision=4
            )
            plot_heatmap(
                [[(post_processed_results_list[benchmark_id] >= post_processed_results_list[benchmark_id][0, 0]) *
                  post_processed_results_list[benchmark_id]]],
                os.path.join(config.get("directory_path"), f"heatmap_{benchmark_id}_greater_baseline.png"),
                f"Models with better performance than the original one ({config.get('model_id').split('/')[-1]}) on the task {benchmark_id}",
                axis_titles=["Coloured cells represent models with better performance than the original one"],
                x_title="Duplicated layers labels",
                y_title="Overwritten layers labels",
                x_labels=[duplicated_layers_labels_column_list[benchmark_id]],
                y_labels=[overwritten_layers_labels_row_list[benchmark_id]],
                fig_size=fig_size,
                edge_color="white",
                precision=4
            )

    def _format_result_dictionary_to_plot(
            self,
            result_dictionary: dict[str, dict[tuple[str, str], dict[str, dict[str, float]]]]
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, np.ndarray]]:
        """
        Formats the result dictionary to be plotted extracting the unique destination paths and the unique source paths
        and the performance arrays for each benchmark.
        After the extraction of these elements they can be formatted in a different way by the specific analysis
        overriding the method 'format_result_dictionary_to_plot'.

        Args:
            result_dictionary (dict[str, dict[tuple[str, str], dict[str, float]]]):
                The dictionary containing the results for each benchmark.

        Returns:
            tuple[dict[str, list[tuple[str, str]]], dict[str, list[tuple[str, str]]], dict[str, np.ndarray]]:
                A tuple
                containing the unique destination paths and the unique source paths and the performance arrays
                for each benchmark.
        """

        return self.format_result_dictionary_to_plot(
            *self.extract_and_sort_unique_elements_from_result_dictionary(result_dictionary)
        )

    @staticmethod
    def extract_and_sort_unique_elements_from_result_dictionary(
            result_dictionary: dict[str, dict[tuple[str, str], dict[str, dict[str, float]]]]
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, np.ndarray]]:
        """
        Extracts and sorts the unique destination paths and the unique source paths and the performance arrays for each
        benchmark.

        Args:
            result_dictionary (dict[str, dict[tuple[str, str], dict[str, float]]]):
                The dictionary containing the results for each benchmark.

        Returns:
            tuple[dict[str, list[str]], dict[str, list[str]], dict[str, np.ndarray]]:
                A tuple containing the unique destination paths and the unique source paths and the performance arrays
                for each benchmark.
        """

        # Helper function to extract tuples from string
        def extract_tuples(s):
            return re.findall(r"\('(.*?)', '(.*?)', '(.*?)'\)", s)

        # Helper function to extract the first number found in a string
        def extract_first_number(s):
            match = re.search(r'\d+', s)
            return int(match.group()) if match else float('inf')

        task_specific_first_elements = {}
        task_specific_second_elements = {}
        performance_arrays = {}

        # Collecting task-specific unique grouped elements and their order
        for task, task_dict in result_dictionary.items():
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

            # Sort first and second groups based on the first number present in them
            first_groups.sort(key=lambda x: extract_first_number(x))
            second_groups.sort(key=lambda x: extract_first_number(x))

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

    def format_result_dictionary_to_plot(
            self,
            destination_paths: dict[str, list[str]],
            source_paths: dict[str, list[str]],
            performance_arrays: dict[str, np.ndarray]
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, np.ndarray]]:
        """
        Formats the result dictionary to be plotted.

        Args:
            destination_paths (dict[str, list[str]]):
                The dictionary containing the destination paths for each benchmark.
            source_paths (dict[str, list[str]]):
                The dictionary containing the source paths for each benchmark.
            performance_arrays (dict[str, np.ndarray]):
                The dictionary containing the performance arrays for each benchmark.

        Returns:
            tuple[dict[str, list[str]], dict[str, list[str]], dict[str, np.ndarray]]:
                A tuple containing the formatted destination paths, the formatted source paths and the performance
                arrays for each benchmark.
        """

        formatted_elements = ({}, {})
        elements = (destination_paths, source_paths)
        for element, formatted_element in zip(elements, formatted_elements):
            for benchmark_id in element.keys():
                formatted_element[benchmark_id] = [group[1:-1].replace("), (", ")\n(") for group in element[benchmark_id]]

        return formatted_elements[0], formatted_elements[1], performance_arrays


class AllLayerCouplesReplacementAnalysis(LayerReplacementAnalysis):
    def get_layers_replacement_mapping(
            self
    ) -> list[dict[tuple, tuple]]:
        """
        Returns the mapping on which the analysis has to be performed.
        In the list of mappings to be analyzed, each one is a dictionary where the keys are the paths to the layers to
        be replaced and the values are the paths to the layers that will replace them.

        Returns:
            list[dict[tuple, tuple]]:
                The list of mappings to be analyzed.
        """

        targets_lists = self.config.get("targets")
        num_layers = self.config.get("num_layers")

        return [
            {
                tuple(el if el != "layer_index" else f"{i}" for el in targets):
                    tuple(el if el != "layer_index" else f"{j}" for el in targets) for targets in targets_lists
            } for i in range(num_layers) for j in range(num_layers) if i != j
        ]

    @override
    def _postprocess_results(
            self
    ) -> None:
        """
        Post-processes the results of the analysis before plotting them.
        """

        destination_layer_path_source_layer_path_mapping_list, performance_dict = self.data

        self.log(f"Post-processing the results...")
        targets_lists = self.config.get("targets")
        num_layers = self.config.get("num_layers")
        redundant_mappings  = [
            {
                tuple(el if el != "layer_index" else f"{i}" for el in targets):
                    tuple(el if el != "layer_index" else f"{j}" for el in targets) for targets in targets_lists
            } for i in range(num_layers) for j in range(num_layers) if i == j
        ]
        for benchmark_id in performance_dict.keys():
            for mapping in redundant_mappings:
                performance_dict[benchmark_id][self.get_performance_dict_key_from_mapping(mapping)] = performance_dict[benchmark_id][("original", "original")]

        self.data = (destination_layer_path_source_layer_path_mapping_list, performance_dict)

        self.store_data()
        self.log("The results have been post-processed and stored.")


class SameLayerCouplesReplacementAnalysis(LayerReplacementAnalysis):
    def get_layers_replacement_mapping(
            self
    ) -> list[dict[tuple, tuple]]:
        """
        Returns the mapping on which the analysis has to be performed.
        In the list of mappings to be analyzed, each one is a dictionary where the keys are the paths to the layers to
        be replaced and the values are the paths to the layers that will replace them.

        Returns:
            list[dict[tuple, tuple]]:
                The list of mappings to be analyzed.
        """

        targets_lists = self.config.get("targets")
        num_layers = self.config.get("num_layers")

        return [
            {
                tuple(el if el != "layer_index" else f"{i}" for el in targets):
                    tuple(el if el != "layer_index" else f"{j}" for el in targets) for targets in targets_lists
            } for i in range(num_layers) for j in range(num_layers) if i == j
        ]


class AllLayersReplacementAnalysis(LayerReplacementAnalysis):
    def get_layers_replacement_mapping(
            self
    ) -> list[dict[tuple, tuple]]:
        """
        Returns the mapping on which the analysis has to be performed.
        In the list of mappings to be analyzed, each one is a dictionary where the keys are the paths to the layers to
        be replaced and the values are the paths to the layers that will replace them.

        Returns:
            list[dict[tuple, tuple]]:
                The list of mappings to be analyzed.
        """

        targets_lists = self.config.get("targets")
        num_layers = self.config.get("num_layers")

        return [
            {
                tuple(el if el != "layer_index" else f"{j}" for el in targets):
                    tuple(el if el != "layer_index" else f"{i}" for el in targets)
                for targets in targets_lists for j in range(num_layers)
            } for i in range(num_layers)
        ]

    # TODO documentaion
    @override
    def format_result_dictionary_to_plot(
            self,
            destination_paths: dict[str, list[str]],
            source_paths: dict[str, list[str]],
            performance_arrays: dict[str, np.ndarray]
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, np.ndarray]]:
        """
        Formats the result dictionary to be plotted.

        Args:
            destination_paths (dict[str, list[str]]):
                The dictionary containing the destination paths for each benchmark.
            source_paths (dict[str, list[str]]):
                The dictionary containing the source paths for each benchmark.
            performance_arrays (dict[str, np.ndarray]):
                The dictionary containing the performance arrays for each benchmark.

        Returns:
            tuple[dict[str, list[str]], dict[str, list[str]], dict[str, np.ndarray]]:
                A tuple containing the formatted destination paths, the formatted source paths and the performance
                arrays for each benchmark.
        """

        formatted_fist_element = {}
        for benchmark_id in destination_paths.keys():
            formatted_fist_element[benchmark_id] = [group[1:-1].replace("), (", ")\n(") for group in destination_paths[benchmark_id]]
        formatted_second_element = {}
        for benchmark_id in source_paths.keys():
            formatted_second_element[benchmark_id] = [group[1:].split("), (")[0] + ")" for group in source_paths[benchmark_id]]

        return formatted_fist_element, formatted_second_element, performance_arrays


class SubsequentLayerReplacementAnalysis(AllLayerCouplesReplacementAnalysis):
    def get_layers_replacement_mapping(
            self
    ) -> list[dict[tuple, tuple]]:
        """
        Returns the mapping on which the analysis has to be performed.
        In the list of mappings to be analyzed, each one is a dictionary where the keys are the paths to the layers to
        be replaced and the values are the paths to the layers that will replace them.

        Returns:
            list[dict[tuple, tuple]]:
                The list of mappings to be analyzed.
        """

        targets_lists = self.config.get("targets")
        num_layers = self.config.get("num_layers")

        return [
            {
                tuple(el if el != "layer_index" else f"{i}" for el in targets):
                    tuple(el if el != "layer_index" else f"{i + 1}" for el in targets) for targets in targets_lists
            } for i in range(num_layers - 1)
        ]