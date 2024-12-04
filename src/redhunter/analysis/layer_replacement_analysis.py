from __future__ import annotations

from abc import abstractmethod
import copy
import gc
import logging
import os
import re
from tqdm import tqdm
from typing import Any, override

import math
import numpy as np

import torch
import pytorch_lightning as pl

import transformers

from exporch import Config, get_available_device

from exporch.utils.causal_language_modeling import load_model_for_causal_lm, load_tokenizer_for_causal_lm
from exporch.experiment import benchmark_id_metric_name_mapping, evaluate_model_on_benchmark
from exporch.utils.general_framework_utils import get_datamodule
from exporch.utils.plot_utils.heatmap import plot_heatmap

from redhunter.utils.layer_replacement_wrapper.layer_replacement_wrapper import LayerReplacingModelWrapper, \
    NullLayerReplacingModelWrapper
from redhunter.analysis_experiment import AnalysisExperiment


class LayerReplacementAnalysis(AnalysisExperiment):
    """
    The class for the layer replacement analysis experiments.
    """

    mandatory_keys = ["num_layers"]

    @override
    def _run_experiment(
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

        config = self.config

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
        model_wrapper = self.wrap_model(base_model, None)
        self.log(f"Model wrapped.")

        for benchmark_id in remaining_destination_layer_path_source_layer_path_mapping_list.keys():
            logging.info("Evaluating the original model")
            print("Evaluating the original model")

            # Defining the evaluation parameters
            benchmark_evaluation_args = evaluation_args[benchmark_id]
            self.log(f"Chosen evaluation args: {benchmark_evaluation_args}")

            if ("original", "original") not in performance_dict[benchmark_id].keys():
                original_model_results = self._evaluate_model(
                    model_wrapper.get_model(), tokenizer, benchmark_id, benchmark_evaluation_args, device)
                #original_model_results = {benchmark_id: {"acc_norm,none": 0.7}} # Testing
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

                # Evaluating the model
                self.log(f"Starting the evaluation of the model on the device {model_wrapper.get_model().device}.")
                results = self._evaluate_model(
                    model_wrapper.get_model(), tokenizer, benchmark_id, benchmark_evaluation_args, device)
                #results = {benchmark_id: {"acc_norm,none": 0.1}} # Testing
                self.log(f"Results of the modified model: {results}")
                print(f"Results of the modified model: {results}")

                # The key in the performance dictionary is a tuple containing the overwritten layers as the first
                # element and the ones used to overwrite the destination as second elements
                performance_dict[benchmark_id][
                    self.get_performance_dict_key_from_mapping(destination_layer_path_source_layer_path_mapping)
                ] = results
                self.log(f"Performance dictionary updated with the results.")

                model_wrapper.reset_replacement()
                self.log(f"Layers reset.")

                self.data = (destination_layer_path_source_layer_path_mapping_list, performance_dict)
                # Storing the data
                self.log(f"Trying to store the data for benchmark {benchmark_id}...")
                self.set_data((destination_layer_path_source_layer_path_mapping_list, performance_dict))
                self.log(f"Partial data stored.")

            self.data = (destination_layer_path_source_layer_path_mapping_list, performance_dict)

            # Storing the data
            self.log(f"Trying to store the data for benchmark {benchmark_id}...")
            self.set_data((destination_layer_path_source_layer_path_mapping_list, performance_dict))
            self.log(f"Stored data up to benchmark {benchmark_id}.")

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
    def wrap_model(
            model,
            *args,
            **kwargs
    ) -> LayerReplacingModelWrapper:
        """
        Returns the model wrapper to be used for the analysis.

        Args:
            model:
                The model to be wrapped.
            *args:
                The additional arguments to be passed to the model wrapper.
            **kwargs:
                The additional keyword arguments to be passed to the model wrapper.

        Returns:
            LayerReplacingModelWrapper:
                The model wrapper to be used for the analysis.
        """

        return LayerReplacingModelWrapper(model, *args, **kwargs)

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

    def _evaluate_model(
            self,
            model: [torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel],
            tokenizer: [transformers.AutoTokenizer | transformers.PreTrainedTokenizer],
            benchmark_id: str,
            benchmark_evaluation_args: dict,
            device: str
    ) -> dict[str, dict[str, float]]:
        """
        Evaluates the model on a benchmark.

        Args:
            model (torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel):
                The model to be evaluated.
            tokenizer (transformers.AutoTokenizer | transformers.PreTrainedTokenizer):
                The tokenizer of the model.
            benchmark_id (str):
                The benchmark on which the model has to be evaluated.
            benchmark_evaluation_args (dict):
                The arguments to use in the benchmark.
            device (str):
                The device to use in the evaluation.

        Returns:
            dict[str, dict[str, float]]:
                The dictionary containing the results of the evaluation.
        """

        return evaluate_model_on_benchmark(
            model,
            tokenizer,
            benchmark_id,
            benchmark_evaluation_args,
            device
        )

    def _postprocess_results(
            self
    ) -> None:
        """
        Post-processes the results obtained from the analysis.
        The performed analysis will depend on the specific subclass of Analysis
        """

        pass

    @override
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

        # Post-processing the results if they have not been post-processed yet
        self._postprocess_results()

        fig_size = config.get("figure_size") if config.contains("figure_size") else (36, 36)
        destination_layer_path_source_layer_path_mapping_list, performance_dict = data

        # Extracting the results for the original model
        original_model_performance = {benchmark_id: all_benchmark_results.pop(("original", "original")) for benchmark_id, all_benchmark_results in performance_dict.items()}
        self.log(f"Original model performance: {original_model_performance}")

        # Plotting the results
        duplicated_layers_labels_list, overwritten_layers_labels_list, post_processed_results_list = self._format_result_dictionary_to_plot(performance_dict, original_model_performance)
        for benchmark_id in post_processed_results_list.keys():
            self.log(f"Printing the results for task: {benchmark_id}")
            plot_heatmap(
                [[post_processed_results_list[benchmark_id]]],
                os.path.join(config.get("directory_path"), f"heatmap_{benchmark_id}.png"),
                f"Results for the model {config.get('model_id').split('/')[-1]} on the task {benchmark_id}",
                axis_titles=[f"Metric: {benchmark_id_metric_name_mapping[benchmark_id]}"],
                x_title="Overwritten layers labels",
                y_title="Duplicated layers labels",
                x_labels=[overwritten_layers_labels_list[benchmark_id]],
                y_labels=[duplicated_layers_labels_list[benchmark_id]],
                fig_size=fig_size,
                edge_color="white",
                fontsize=22,
                precision=3
            )
            plot_heatmap(
                [[(post_processed_results_list[benchmark_id] >= post_processed_results_list[benchmark_id][0, 0]) *
                  post_processed_results_list[benchmark_id]]],
                os.path.join(config.get("directory_path"), f"heatmap_{benchmark_id}_greater_baseline.png"),
                f"Models with better performance than the original one ({config.get('model_id').split('/')[-1]}) on the task {benchmark_id}",
                axis_titles=["Coloured cells represent models with better performance than the original one"],
                x_title="Overwritten layers labels",
                y_title="Duplicated layers labels",
                x_labels=[overwritten_layers_labels_list[benchmark_id]],
                y_labels=[duplicated_layers_labels_list[benchmark_id]],
                fig_size=fig_size,
                edge_color="white",
                fontsize=22,
                precision=3
            )

    def _format_result_dictionary_to_plot(
            self,
            result_dictionary: dict[str, dict[tuple[str, str], dict[str, dict[str, float]]]],
            original_model_performance_dictionary: dict[str, dict[str, float]]
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, np.ndarray]]:
        """
        Formats the result dictionary to be plotted extracting the unique source paths and the unique destination paths
        and the performance arrays for each benchmark.
        After the extraction of these elements, they can be formatted in different ways by the specific analysis
        overriding the method 'format_result_dictionary_to_plot'.

        Args:
            result_dictionary (dict[str, dict[tuple[str, str], dict[str, float]]]):
                The dictionary containing the results for each benchmark.
            original_model_performance_dictionary (dict[str, dict[str, float]]):
                The dictionary containing the performance of the original model for each benchmark.

        Returns:
            dict[str, list[tuple[str, str]]]:
                Unique source paths for each benchmark.
            dict[str, list[tuple[str, str]]]:
                Unique destination paths for each benchmark.
            dict[str, np.ndarray]:
                Performance arrays for each benchmark.
        """

        return self.format_result_dictionary_to_plot(
            *self.extract_and_sort_destination_source_labels_from_result_dictionary(result_dictionary), original_model_performance_dictionary
        )

    @staticmethod
    def extract_and_sort_destination_source_labels_from_result_dictionary(
            result_dictionary: dict[str, dict[tuple[str, str], dict[str, dict[str, float]]]]
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, np.ndarray]]:
        """
        Formats the result dictionary to be plotted extracting the unique source paths and the unique destination paths
        and the performance arrays for each benchmark.

        Args:
            result_dictionary (dict[str, dict[tuple[str, str], dict[str, float]]]):
                The dictionary containing the results for each benchmark.

        Returns:
            dict[str, list[tuple[str, str]]]:
                Unique source paths for each benchmark.
            dict[str, list[tuple[str, str]]]:
                Unique destination paths for each benchmark.
            dict[str, np.ndarray]:
                Performance arrays for each benchmark.
        """

        # Helper function to extract tuples from string
        def extract_tuples(s):
            # Match tuples of the form ('element1', 'element2', ...) with any number of elements
            return re.findall(r"\('(.*?)'(?:, '(.*?)')*\)", s)

        # Helper function to extract the first number found in a string
        def extract_first_number(s):
            match = re.search(r'\d+', s)
            return int(match.group()) if match else float('inf')

        destination_paths_all_benchmarks = {}
        source_paths_all_benchmarks = {}
        performance_arrays = {}

        # Collecting task-specific unique grouped elements and their order
        for benchmark_id, benchmark_dict in result_dictionary.items():
            benchmark_destination_paths = []
            benchmark_source_paths = []

            # Collecting grouped elements for each task
            for (destination_paths_str, source_paths_str), _ in benchmark_dict.items():
                a = extract_tuples(destination_paths_str)
                b = extract_tuples(source_paths_str)
                destination_paths = str(extract_tuples(destination_paths_str))
                source_paths = str(extract_tuples(source_paths_str))

                # Adding unique groups for the first and second elements
                if destination_paths not in benchmark_destination_paths:
                    benchmark_destination_paths.append(destination_paths)
                if source_paths not in benchmark_source_paths:
                    benchmark_source_paths.append(source_paths)

            # Sort first and second groups based on the first number present in them
            benchmark_destination_paths.sort(key=lambda x: extract_first_number(x))
            benchmark_source_paths.sort(key=lambda x: extract_first_number(x))

            destination_paths_all_benchmarks[benchmark_id] = benchmark_destination_paths
            source_paths_all_benchmarks[benchmark_id] = benchmark_source_paths

            performance_array = np.full((len(benchmark_source_paths), len(benchmark_destination_paths)), np.nan)
            # Filling the performance array with the metrics for this task
            for (destination_paths_str, source_paths), value in benchmark_dict.items():
                destination_paths = str(extract_tuples(destination_paths_str))
                source_paths = str(extract_tuples(source_paths))

                # Finding the correct row and column by the group index
                col = benchmark_destination_paths.index(destination_paths)
                row = benchmark_source_paths.index(source_paths)
                performance_array[row, col] = value[benchmark_id][benchmark_id_metric_name_mapping[benchmark_id]]

            performance_arrays[benchmark_id] = performance_array

        return source_paths_all_benchmarks, destination_paths_all_benchmarks, performance_arrays

    @staticmethod
    def format_result_dictionary_to_plot(
            source_paths: dict[str, list[str]],
            destination_paths: dict[str, list[str]],
            performance_arrays: dict[str, np.ndarray],
            original_model_performance_dictionary: dict[str, dict[str, float]]
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, np.ndarray]]:
        """
        Formats the result dictionary to be plotted.

        Args:
            source_paths (dict[str, list[str]]):
                The dictionary containing the source paths for each benchmark.
            destination_paths (dict[str, list[str]]):
                The dictionary containing the destination paths for each benchmark.
            performance_arrays (dict[str, np.ndarray]):
                The dictionary containing the performance arrays for each benchmark.
            original_model_performance_dictionary (dict[str, dict[str, float]]):
                The dictionary containing the performance of the original model for each benchmark.

        Returns:
            dict[str, list[tuple[str, str]]]:
                Unique source paths for each benchmark.
            dict[str, list[tuple[str, str]]]:
                Unique destination paths for each benchmark.
            dict[str, np.ndarray]:
                Performance arrays for each benchmark.
        """

        formatted_elements = ({}, {})
        elements = (source_paths, destination_paths)
        for element, formatted_element in zip(elements, formatted_elements):
            for benchmark_id in element.keys():
                formatted_element[benchmark_id] = [group[1:-1].replace("), (", ")\n(") for group in element[benchmark_id]]

        return formatted_elements[0], formatted_elements[1], performance_arrays


class SingleNullLayersReplacementAnalysis(LayerReplacementAnalysis):
    """
    The class for the layer replacement analysis experiments.
    It performs the analysis by replacing a single layer with a null layer.
    """

    @override
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
                tuple(el if el != "block_index" else f"{i}" for el in targets): ("all_zeros",) for targets in targets_lists
            } for i in range(num_layers)
        ]

    @staticmethod
    @override
    def wrap_model(
            model,
            *args,
            **kwargs
    ) -> LayerReplacingModelWrapper:
        """
        Returns the model wrapper to be used for the analysis.

        Args:
            model:
                The model to be wrapped.
            *args:
                The additional arguments to be passed to the model wrapper.
            **kwargs:
                The additional keyword arguments to be passed to the model wrapper.

        Returns:
            LayerReplacingModelWrapper:
                The model wrapper to be used for the analysis.
        """

        return NullLayerReplacingModelWrapper(model, *args, **kwargs)

    @staticmethod
    @override
    def format_result_dictionary_to_plot(
            source_paths: dict[str, list[str]],
            destination_paths: dict[str, list[str]],
            performance_arrays: dict[str, np.ndarray],
            original_model_performance_dictionary: dict[str, dict[str, dict[str, float]]]
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, np.ndarray]]:
        """
        Formats the result dictionary to be plotted.

        Args:
            source_paths (dict[str, list[str]]):
                The dictionary containing the source paths for each benchmark.
            destination_paths (dict[str, list[str]]):
                The dictionary containing the destination paths for each benchmark.
            performance_arrays (dict[str, np.ndarray]):
                The dictionary containing the performance arrays for each benchmark.
            original_model_performance_dictionary (dict[str, dict[str, float]]):
                The dictionary containing the performance of the original model for each benchmark.

        Returns:
            dict[str, list[tuple[str, str]]]:
                Unique source paths for each benchmark.
            dict[str, list[tuple[str, str]]]:
                Unique destination paths for each benchmark.
            dict[str, np.ndarray]:
                Performance arrays for each benchmark.
        """

        for benchmark_id in performance_arrays.keys():
            original_model_performance = original_model_performance_dictionary[benchmark_id][benchmark_id][benchmark_id_metric_name_mapping[benchmark_id]]
            performance_arrays[benchmark_id] = np.concatenate(
                (np.array([[original_model_performance]]), performance_arrays[benchmark_id]), axis=1
            )
            destination_paths[benchmark_id] = ["Original model"] + destination_paths[benchmark_id]

        return source_paths, destination_paths, performance_arrays


class AllLayerCouplesReplacementAnalysis(LayerReplacementAnalysis):
    """
    The class for the layer replacement analysis experiments.
    It performs the analysis by replacing all the couples of layers.
    """

    @override
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
                tuple(el if el != "block_index" else f"{i}" for el in targets):
                    tuple(el if el != "block_index" else f"{j}" for el in targets) for targets in targets_lists
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
                tuple(el if el != "block_index" else f"{i}" for el in targets):
                    tuple(el if el != "block_index" else f"{j}" for el in targets) for targets in targets_lists
            } for i in range(num_layers) for j in range(num_layers) if i == j
        ]
        for benchmark_id in performance_dict.keys():
            for mapping in redundant_mappings:
                performance_dict[benchmark_id][self.get_performance_dict_key_from_mapping(mapping)] = performance_dict[benchmark_id][("original", "original")]

        self.data = (destination_layer_path_source_layer_path_mapping_list, performance_dict)

        self.set_data((destination_layer_path_source_layer_path_mapping_list, performance_dict))
        self.log("The results have been post-processed and stored.")


class AllLayerCouplesDisplacementBasedReplacementAnalysis(AllLayerCouplesReplacementAnalysis):
    """
    The class for the layer replacement analysis experiments.
    It performs the analysis by replacing all the couples of layers following an order based on a displacement.
    """

    @override
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
        displacements = [(i+1)//2 if i % 2 != 0 else -i//2 for i in range(1, 2*num_layers+1)]

        return [
            {
                tuple(el if el != "block_index" else f"{i}" for el in targets):
                    tuple(el if el != "block_index" else f"{i + displacement}" for el in targets)
                for targets in targets_lists
            }
            for displacement in displacements
            for i in range(max(-displacement, 0), min(num_layers - displacement, num_layers))
    ]


class SpecificDisplacementLayerReplacementAnalysis(AllLayerCouplesReplacementAnalysis):
    """
    The class for the layer replacement analysis experiments with a specific displacement between the replaced and the
    replacing layers.
    """

    mandatory_keys = ["displacement"]

    @override
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
        displacement = self.config.get("displacement")

        return [
            {
                tuple(el if el != "block_index" else f"{i}" for el in targets):
                    tuple(el if el != "block_index" else f"{i + displacement}" for el in targets) for targets in targets_lists
            } for i in range(max(-displacement, 0), min(num_layers - displacement, num_layers))
        ]


class SubsequentLayerReplacementAnalysis(AllLayerCouplesReplacementAnalysis):
    """
    The class for the layer replacement analysis experiments. It performs the analysis by replacing subsequent layers.
    """

    @override
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
                tuple(el if el != "block_index" else f"{i}" for el in targets):
                    tuple(el if el != "block_index" else f"{i + 1}" for el in targets) for targets in targets_lists
            } for i in range(num_layers - 1)
        ]


class PreviousLayerReplacementAnalysis(AllLayerCouplesReplacementAnalysis):
    """
    The class for the layer replacement analysis experiments. It performs the analysis by replacing previous layers.
    """

    @override
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
                tuple(el if el != "block_index" else f"{i + 1}" for el in targets):
                    tuple(el if el != "block_index" else f"{i}" for el in targets) for targets in targets_lists
            } for i in range(num_layers - 1)
        ]


class SpecificReplacedLayerReplacementAnalysis(AllLayerCouplesReplacementAnalysis):
    """
    The class for the layer replacement analysis experiments. It performs the analysis by replacing a specific layer with
    all the other layers.
    """

    mandatory_keys = ["replaced_block_index"]

    @override
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
        replaced_block_index = self.config.get("replaced_block_index")

        return [
            {
                tuple(el if el != "block_index" else f"{replaced_block_index}" for el in targets):
                    tuple(el if el != "block_index" else f"{i}" for el in targets) for targets in targets_lists
            } for i in range(num_layers) if i != replaced_block_index
        ]


class SpecificReplacingLayerReplacementAnalysis(AllLayerCouplesReplacementAnalysis):
    """
    The class for the layer replacement analysis experiments. It performs the analysis by replacing all the layers with
    a specific layer.
    """

    mandatory_keys = ["replacing_block_index"]

    @override
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
        replacing_block_index = self.config.get("replacing_block_index")

        return [
            {
                tuple(el if el != "block_index" else f"{i}" for el in targets):
                    tuple(el if el != "block_index" else f"{replacing_block_index}" for el in targets) for targets in targets_lists
            } for i in range(num_layers) if i != replacing_block_index
        ]


class SameLayerCouplesReplacementAnalysis(LayerReplacementAnalysis):
    """
    The class for the layer replacement analysis experiments. It performs the analysis by replacing a layer with the
    itself. It is a testing analysis.
    """

    @override
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
                tuple(el if el != "block_index" else f"{i}" for el in targets):
                    tuple(el if el != "block_index" else f"{j}" for el in targets) for targets in targets_lists
            } for i in range(num_layers) for j in range(num_layers) if i == j
        ]


class AllLayersReplacementAnalysis(LayerReplacementAnalysis):
    """
    The class for the layer replacement analysis experiments. It performs the analysis by replacing all the layers with
    a given one.
    """

    @override
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
                tuple(el if el != "block_index" else f"{j}" for el in targets):
                    tuple(el if el != "block_index" else f"{i}" for el in targets)
                for targets in targets_lists for j in range(num_layers)
            } for i in range(num_layers)
        ]

    @staticmethod
    @override
    def format_result_dictionary_to_plot(
            source_paths: dict[str, list[str]],
            destination_paths: dict[str, list[str]],
            performance_arrays: dict[str, np.ndarray],
            original_model_performance_dictionary: dict[str, dict[str, float]]
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, np.ndarray]]:
        """
        Formats the result dictionary to be plotted.

        Args:
            source_paths (dict[str, list[str]]):
                The dictionary containing the source paths for each benchmark.
            destination_paths (dict[str, list[str]]):
                The dictionary containing the destination paths for each benchmark.
            performance_arrays (dict[str, np.ndarray]):
                The dictionary containing the performance arrays for each benchmark.
            original_model_performance_dictionary (dict[str, dict[str, float]]):
                The dictionary containing the performance of the original model for each benchmark.

        Returns:
            dict[str, list[tuple[str, str]]]:
                Unique source paths for each benchmark.
            dict[str, list[tuple[str, str]]]:
                Unique destination paths for each benchmark.
            dict[str, np.ndarray]:
                Performance arrays for each benchmark.
        """

        formatted_source_paths = {}
        for benchmark_id in source_paths.keys():
            formatted_source_paths[benchmark_id] = [group[1:].split("), (")[0] + ")" for group in source_paths[benchmark_id]]
        formatted_destination_paths = {}
        for benchmark_id in destination_paths.keys():
            formatted_destination_paths[benchmark_id] = [group[1:-1].replace("), (", ")\n(") for group in destination_paths[benchmark_id]]

        return formatted_source_paths, formatted_destination_paths, performance_arrays


class PerplexityLayerReplacementAnalysis(AllLayerCouplesDisplacementBasedReplacementAnalysis):
    """
    The class for the layer replacement analysis experiments where models are compared based on their perplexity on a
    benchmark.

    Attributes:
        benchmarks (dict[str, pl.LightningDataModule]):
            The benchmarks to be used in the analysis
    """

    def __init__(
            self,
            *args,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.benchmarks = None


    def initialize_benchmark(
            self,
            benchmark_id: str,
            tokenizer: transformers.AutoTokenizer | transformers.PreTrainedTokenizer,
            benchmarks_args: dict[str, Any]
    ) -> None:
        """
        Initializes the benchmark to be used in the analysis.

        Args:
            benchmark_id (list[str]):
                The benchmark to be initialized.
            tokenizer (transformers.AutoTokenizer | transformers.PreTrainedTokenizer):
                The tokenizer to be used in the benchmarks.
            benchmarks_args (dict[str, dict[str, Any]]):
                The arguments to be used in the benchmarks.
        """

        if self.benchmarks is None:
            self.benchmarks = {}

        benchmarks_args_copy = benchmarks_args.copy()
        max_length = benchmarks_args_copy.pop("max_length") if "max_length" in benchmarks_args_copy.keys() else 512
        self.benchmarks.update(
            {benchmark_id: get_datamodule(
                benchmark_id,
                tokenizer,
                max_length,
                **benchmarks_args_copy)})

    @override
    def _evaluate_model(
            self,
            model: [torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel],
            tokenizer: [transformers.AutoTokenizer | transformers.PreTrainedTokenizer],
            benchmark_id: str,
            benchmark_evaluation_args: dict,
            device: str
    ) -> dict[str, dict[str, float]]:
        """
        Evaluates the model on a benchmark computing the perplexity.

        Args:
            model (torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel):
                The model to be evaluated.
            tokenizer (transformers.AutoTokenizer | transformers.PreTrainedTokenizer):
                The tokenizer of the model.
            benchmark_id (str):
                The benchmark on which the model has to be evaluated.
            benchmark_evaluation_args (dict):
                The arguments to use in the benchmark.
            device (str):
                The device to use in the evaluation.

        Returns:
            dict[str, dict[str, float]]:
                The dictionary containing the results of the evaluation.
                The structure of the dictionary is the following:
                {
                    benchmark_id: {
                        f"{evaluation_metric_name}": perplexity
                    }
                }
        """

        if self.benchmarks is None or benchmark_id not in self.benchmarks.keys():
            self.initialize_benchmark(benchmark_id, tokenizer, benchmark_evaluation_args)

        model.eval()

        datamodule = self.benchmarks[benchmark_id]
        datamodule.setup()

        # Moving model to GPU if available
        device = get_available_device(device)
        try:
            model.to(device)
        except Exception:
            pass

        # Computing loss over validation set
        total_loss = 0.0
        total_tokens = 0
        val_loader = datamodule.val_dataloader()
        with torch.no_grad():
            for batch in tqdm(val_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Performing the forward pass and computing the loss
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                total_loss += loss.item() * input_ids.size(0)
                total_tokens += torch.sum(attention_mask).item()

        # Computing perplexity
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        return {
            f"{benchmark_id}": {"perplexity": perplexity}
        }
