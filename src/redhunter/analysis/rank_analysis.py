from abc import ABC, abstractmethod
import copy
import gc
import os
from typing import Any, override

from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt

import numpy as np

import torch

from exporch import Config
from exporch.utils.causal_language_modeling import load_model_for_causal_lm
from exporch.utils.plot_utils import plot_heatmap

from redhunter.analysis.analysis_utils import AnalysisTensorDict, AnalysisTensorWrapper, extract, \
    layer_name_matrix_name_mapping, extract_number
from redhunter.analysis_experiment import AnalysisExperiment


class RankAnalysis(AnalysisExperiment, ABC):
    """
    Class to perform the rank analysis of the layers of a model.
    """


    def _run_experiment(
            self
    ) -> None:
        """
        Runs the experiment.

        It computes the singular values and fraction of explained variance of the layers of the model and the
        approximated rank of the matrices.
        """

        if self.get_data() is None:
            if self.is_low_memory_mode():
                original_configuration = self.get_config()
                for target in original_configuration.get("targets"):
                    modified_configuration = copy.deepcopy(original_configuration)
                    modified_configuration.set("targets", [target])
                    self.set_config(modified_configuration)

                    # Extracting the matrices from the model
                    extracted_layers = self._extract_layers()

                    # Processing the extracted matrices
                    preprocessed_tensors = self._preprocess_extracted_tensors(extracted_layers)

                    # Computing the singular values and fraction of explained variance
                    results = self._compute_singular_values_on_analysis_dict(preprocessed_tensors)

                    # Saving the singular values and explained variance
                    if self.get_data() is not None:
                        data = self.get_data()[0].copy()
                        data.update(results)
                        results = data
                    self.set_data(results, position=0)
                self.set_config(original_configuration)
            else:
                # Extracting the matrices from the model
                extracted_layers = self._extract_layers()

                # Processing the extracted matrices
                preprocessed_tensors = self._preprocess_extracted_tensors(extracted_layers)

                # Computing the singular values and fraction of explained variance
                results = self._compute_singular_values_on_analysis_dict(preprocessed_tensors)

                # Saving the singular values and explained variance
                self.set_data(results, position=0)

        # Computing the rank
        results = self._compute_rank()
        self.set_data(results, position=0)

    def _extract_layers(
            self
    ) -> AnalysisTensorDict:
        """
        Extracts the layers to be analyzed from the model.

        Returns:
            AnalysisTensorDict:
                The extracted layers
        """

        self.log("Extracting the matrices from the model.")
        config = self.get_config()

        # Loading the model
        model = load_model_for_causal_lm(config)

        # Extracting the layers to analyze
        extracted_layers = AnalysisTensorDict()
        extract(
            model,
            config.get("targets"),
            extracted_layers,
            black_list=config.get("black_list") if config.contains("black_list") else (),
            verbose=config.get_verbose()
        )
        self.log("Layers extracted.")
        del model
        gc.collect()

        return extracted_layers

    def _preprocess_extracted_tensors(
            self,
            extracted_tensors: AnalysisTensorDict
    ) -> AnalysisTensorDict:
        """
        Processes the extracted matrices.

        Args:
            extracted_tensors (AnalysisTensorDict):
                The extracted matrices.

        Returns:
            AnalysisTensorDict:
                The preprocessed tensors.
        """

        self.log("Preprocessing the extracted tensors.")
        preprocessed_tensors = self.preprocess_extracted_tensors(extracted_tensors)

        return preprocessed_tensors

    @abstractmethod
    def preprocess_extracted_tensors(
            self,
            extracted_tensors: AnalysisTensorDict
    ) -> AnalysisTensorDict:
        """
        Preprocesses the extracted matrices.
        The operations to perform will depend on the specific subclass of RankAnalysis.

        Args:
            extracted_tensors (AnalysisTensorDict):
                The extracted matrices.

        Returns:
            AnalysisTensorDict:
                The preprocessed tensors.
        """

        return extracted_tensors

    def _compute_singular_values_on_analysis_dict(
            self,
            analyzed_tensors: AnalysisTensorDict
    ) -> dict:
        """
        Computes the singular values of the matrices in the analysis dictionary.

        Args:
            analyzed_tensors (AnalysisTensorDict):
                The analyzed tensors.

        Returns:
            dict:
                The singular values of the tensors.
        """

        self.log("Computing the singular values of the layers.", print_message=True)
        results = analyzed_tensors.compute_singular_values()

        return results

    def _compute_rank(
            self
    ) -> dict:
        """
        Computes the rank of the matrices in the analysis dictionary.

        Returns:
            dict:
                The rank of the tensors.
        """

        if self.get_data() is None:
            raise Exception("The singular values of the layers must be computed before computing the rank.")

        self.log("Computing the rank of the layers.", print_message=True)

        explained_variance_threshold = self.get_config().get("explained_variance_threshold") if self.get_config().contains("explained_variance_threshold") else 1.
        singular_values_threshold = self.get_config().get("singular_values_threshold") if self.get_config().contains("singular_values_threshold") else 0.

        if explained_variance_threshold <= 0. or explained_variance_threshold > 1.:
            raise ValueError("The threshold on the explained variance must be between 0 and 1.")

        singular_values_explained_variance = self.get_data()[0]
        for label in singular_values_explained_variance.keys():
            self.log(f"Computing the rank of the tensors with label '{' '.join(label)}'.", print_message=True)
            for tensor_key in singular_values_explained_variance[label]:
                singular_values = singular_values_explained_variance[label][tensor_key]["singular_values"]
                explained_variance = singular_values_explained_variance[label][tensor_key]["explained_variance"]

                rank_based_on_explained_variance = np.argmax(explained_variance >= explained_variance_threshold) + 1

                if singular_values[-1] > singular_values_threshold:
                    rank_based_on_singular_values = len(singular_values)
                else:
                    rank_based_on_singular_values = np.argmax(singular_values < singular_values_threshold)

                rank = np.minimum(rank_based_on_explained_variance, rank_based_on_singular_values)

                singular_values_explained_variance[label][tensor_key].update({"rank": rank})

        return singular_values_explained_variance

    def _postprocess_results(
            self
    ) -> None:
        """
        Post-processes the results obtained from the experiment.
        By default, it does nothing.
        """

        pass

    def _plot_results(
            self,
            config: Config,
            data: Any
    ) -> None:
        """
        Plots the results obtained from the experiment.
        It plots the singular values and fraction of explained variance of the layers on figure and the approximated
        rank of the matrices on another.

        Args:
            config (Config):
                The configuration of the experiment.
            data (Any):
                The data obtained from the analysis.
        """

        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.it'] = 'STIXGeneral:italic'
        plt.rcParams['mathtext.bf'] = 'STIXGeneral:bold'

        kwargs_plot_singular_values_distribution = self.get_kwargs_plot_singular_values_distribution(config, data)
        self._plot_singular_values_distribution(config, data, kwargs_plot_singular_values_distribution)
        kwargs_plot_rank_analysis = self.get_kwargs_plot_rank_analysis(config, data)
        self._plot_rank_analysis(config, data, kwargs_plot_rank_analysis)

    def get_kwargs_plot_singular_values_distribution(
            self,
            config: Config,
            data: Any
    ) -> dict:
        """
        Gets the keyword arguments to plot the singular values distribution.

        Args:
            config (Config):
                The configuration of the experiment.
            data (Any):
                The data obtained from the analysis.

        Returns:
            dict:
                The keyword arguments.
        """

        return {}

    def _plot_singular_values_distribution(
            self,
            config: Config,
            data: Any,
            kwargs: dict
    ) -> None:
        """
        Plots the singular values distribution.

        Args:
            config (Config):
                The configuration of the experiment.
            data (Any):
                The data obtained from the analysis.
            kwargs (dict):
                Additional keyword arguments.
                Structure:
                    >> { label: { key: value, ... }, ... }
        """

        self.log("Plotting singular values and fraction of explained variance.")
        results = data[0]
        fig_size = config.get("figure_size") if config.contains("figure_size") else (10, 10)
        key_arguments = {label: {
            "title": f"Singular values and fraction of explained variance of the tensors with label '{" ".join(label)}' of the model '{config.get("model_id")}'",
            "sv_title": f"Singular values of the tensors with label '{" ".join(label)}'",
            "ev_title": f"Fraction of explained variance of the tensors with label '{" ".join(label)}'"
        } for label in results.keys()}
        key_arguments.update(kwargs)

        for label in results.keys():
            # Defining the colormap
            colormap = get_cmap("viridis")
            num_tensors = len(results[label].keys())
            line_colors = colormap(np.linspace(0, 1, num_tensors))

            # Creating the plot
            fig, ax = plt.subplots(1, 2, figsize=(fig_size[1], fig_size[1] / 2))
            #fig.suptitle(key_arguments[label]["title"], fontsize=16)

            for idx, (tensor_key, color) in enumerate(zip(results[label].keys(), line_colors)):
                singular_values = results[label][tensor_key]["singular_values"]
                explained_variance = results[label][tensor_key]["explained_variance"]

                plot_color = self.config.get("first_color", color) if idx == 0 else color
                plot_color = self.config.get("last_color", plot_color) if idx == len(results[label].keys()) - 1 else plot_color

                key_elements = tensor_key.split(" ")
                layer_idx = extract_number(tensor_key)
                plot_label = layer_name_matrix_name_mapping[key_elements[-1]]

                plot_label = plot_label[:-2] + rf"\, , {layer_idx}" + "}$"
                ax[0].plot(singular_values, label=plot_label, color=plot_color)
                ax[1].plot(explained_variance, label=plot_label, color=plot_color)

                position = int(len(singular_values) * (idx + 1) / (num_tensors + 1))

                # Annotating the block number on chosen position in the singular values plot
                ax[0].text(
                    position,
                    singular_values[position],
                    str(idx),
                    fontsize=10,
                    color=plot_color,
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor=plot_color, facecolor="white"),
                )

                # Annotating the block number on chosen position in the explained variance plot
                ax[1].text(
                    position,
                    explained_variance[position],
                    str(idx),
                    fontsize=10,
                    color=plot_color,
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor=plot_color, facecolor="white"),
                )

            # Setting the singular values plot properties
            #ax[0].set_title(key_arguments[label]["sv_title"])
            ax[0].set_xlabel("Singular value index", fontsize=14)
            ax[0].set_ylabel("Singular value", fontsize=14)
            ax[0].legend(fontsize=12)
            ax[0].grid(True)

            # Setting the explained variance plot properties
            #ax[1].set_title(key_arguments[label]["ev_title"])
            ax[1].set_xlabel("Number of singular values considered", fontsize=14)
            ax[1].set_ylabel("Explained variance", fontsize=14)
            ax[1].legend(fontsize=12)
            ax[1].grid(True)

            plt.tight_layout()
            # Saving the plot
            storage_path = str(os.path.join(self.get_experiment_path(), f"{config.get("model_id").split("/")[-1]}_singular_values_distribution_{"_".join(label)}.pdf"))
            plt.savefig(storage_path, format="pdf")

    def _plot_rank_analysis(
            self,
            config: Config,
            data: Any,
            kwargs: dict
    ) -> None:
        """
        Plots a heatmap containing the approximated ranks of the model layers.

        Args:
            config (Config):
                The configuration of the experiment.
            data (Any):
                The data obtained from the analysis.
            kwargs (dict):
                Additional keyword arguments.
                Structure:
                    >> { label: { key: value, ... }, ... }
        """

        self.log("Plotting the rank analysis.")

        results = data[0]
        layer_types = list(results.keys())
        number_of_blocks = len(list(results.values())[0].keys())

        explained_variance_threshold = config.get("explained_variance_threshold") if config.contains("explained_variance_threshold") else 1.
        singular_values_threshold = config.get("singular_values_threshold") if config.contains("singular_values_threshold") else 0.
        relative_rank = config.get("relative_rank") if config.contains("relative_rank") else False

        fig_size_ranks = config.get("fig_size_ranks") if config.contains("fig_size_ranks") else (10, 8)

        heatmap_name = config.get("model_id").split("/")[-1]
        heatmap_name += "_" + config.get("heatmap_name") if config.contains("heatmap_name") else "_heatmap"
        heatmap_name += "_expvar_" + str(explained_variance_threshold).replace('.', '_') + "_sv_" + str(singular_values_threshold).replace('.', '_')

        key_arguments = {
            #"title": "Rank analysis of the matrices of the model" + f" (explained variance threshold: {explained_variance_threshold})",
            "title": None,
            "axes_displacement" : "column",
            #"axis_titles" : [f"Rank values for matrices with label {label}" for label in layer_types],
            "axis_titles": None,
            "x_title" : "Block index",
            "y_title" : "Layer type",
            "x_labels" : [[str(i) for i in range(number_of_blocks)],],
            "y_labels" : [[layer_name_matrix_name_mapping[label[-1]] for label in layer_types],],
            "x_rotation": 0,
            "fig_size" : fig_size_ranks,
            "precision" : 2,
            "fontsize" : 10
        }
        key_arguments.update(kwargs)

        rank_matrix = np.zeros((len(layer_types), number_of_blocks))
        relative_rank_matrix = np.zeros((len(layer_types), number_of_blocks))

        tensor_shapes_list = []
        for index_label, label in enumerate(layer_types):
            tensor_shapes_list.append([])
            for index_block, tensor_key in enumerate(results[label].keys()):
                rank = results[label][tensor_key]["rank"]
                shape = results[label][tensor_key]["shape"]
                tensor_shapes_list[-1].append(shape)
                rank_matrix[index_label, index_block] = rank
                if config.contains("rank_related_to_parameters") and config.get("rank_related_to_parameters"):
                    relative_rank_matrix[index_label, index_block] = rank / (torch.sqrt(torch.tensor(shape[0]) * torch.tensor(shape[1]))).item()
                else:
                    relative_rank_matrix[index_label, index_block] = rank / min(shape[0], shape[1])

        if relative_rank:
            all_ranks_to_plot = [rank_matrix, relative_rank_matrix]
        else:
            all_ranks_to_plot = [rank_matrix,]

        plot_heatmap(
            [all_ranks_to_plot,],
            save_path=str(os.path.join(config.get("experiment_root_path"), f"{heatmap_name}.pdf")),
            index_colormesh=1 if relative_rank else 0,
            **key_arguments,
            vmin=[0,],
            vmax=[1,],
            use_custom_font=False
        )

    def get_kwargs_plot_rank_analysis(
            self,
            config: Config,
            data: Any
    ) -> dict:
        """
        Gets the keyword arguments to plot the rank analysis.

        Args:
            config (Config):
                The configuration of the experiment.
            data (Any):
                The data obtained from the analysis.

        Returns:
            dict:
                The keyword arguments.
        """

        return {}


class TypeSortedRankAnalysis(RankAnalysis, ABC):
    """
    Class to perform the rank analysis of the layers of a model.
    """

    @override
    def _preprocess_extracted_tensors(
            self,
            extracted_tensors: AnalysisTensorDict
    ) -> AnalysisTensorDict:
        """
        Preprocesses the extracted matrices.
        The operations to perform will depend on the specific subclass of RankAnalysis.

        Args:
            extracted_tensors (AnalysisTensorDict):
                The extracted matrices.
        """
        self.log("Preprocessing the extracted tensors.")
        sorted_tensors = extracted_tensors.group_by_name()

        preprocessed_tensors = self.preprocess_extracted_tensors(sorted_tensors)

        return preprocessed_tensors


class OriginalLayersRankAnalysis(TypeSortedRankAnalysis):
    """
    Class to perform the rank analysis of the layers of a model.
    """

    @override
    def preprocess_extracted_tensors(
            self,
            extracted_tensors: AnalysisTensorDict
    ) -> AnalysisTensorDict:
        """
        Preprocesses the extracted matrices. It does nothing to them.

        Args:
            extracted_tensors (AnalysisTensorDict):
                The extracted matrices.
        """

        return extracted_tensors


class ConcatenatedLayersRankAnalysis(TypeSortedRankAnalysis):
    """
    Class to perform the rank analysis of the layers of a model.
    """

    @override
    def preprocess_extracted_tensors(
            self,
            extracted_tensors: AnalysisTensorDict
    ) -> AnalysisTensorDict:
        """
        Preprocesses the extracted matrices.
        It concatenates the matrices with the same name.

        Args:
            extracted_tensors (AnalysisTensorDict):
                The extracted matrices.
        """

        preprocessed_tensors = AnalysisTensorDict()
        for label in extracted_tensors.get_keys():
            extracted_tensors_label = extracted_tensors.get_tensor_list(label)
            for extracted_tensor in extracted_tensors_label:
                a = extracted_tensor.get_tensor()

            concatenated_tensors_row_wise = torch.cat(
                [extracted_tensor.get_tensor() for extracted_tensor in extracted_tensors_label], dim=0)
            concatenated_tensors_column_wise = torch.cat(
                [extracted_tensor.get_tensor() for extracted_tensor in extracted_tensors_label], dim=1)
            paths = [["block index" if item.isdigit() else item for item in extracted_tensor.get_path()] for extracted_tensor in extracted_tensors_label]
            paths = [list(x) for x in set(tuple(sublist) for sublist in paths)]
            path = sum(([item] if i == 0 else ["&", item] for i, sublist in enumerate(paths) for item in sublist), [])
            label_row_wise = ["row-wise concatenation",]
            label_column_wise = ["column-wise concatenation",]
            label_row_wise.extend(label)
            label_column_wise.extend(label)
            preprocessed_tensors.append_tensor(label_row_wise, [AnalysisTensorWrapper(concatenated_tensors_row_wise, path=path, name=label[0], label=f"concatenated {label[0]}", block_index=0)])
            preprocessed_tensors.append_tensor(label_column_wise, [AnalysisTensorWrapper(concatenated_tensors_column_wise, path=path, name=label[0], label=f"concatenated {label[0]}", block_index=0)])

        return preprocessed_tensors

    def _plot_singular_values_distribution(
            self,
            config: Config,
            data: Any,
            kwargs: dict
    ) -> None:
        """
        Plots the singular values distribution.

        Args:
            config (Config):
                The configuration of the experiment.
            data (Any):
                The data obtained from the analysis.
            kwargs (dict):
                Additional keyword arguments.
                Structure:
                    >> { label: { key: value, ... }, ... }
        """

        self.log("Plotting singular values and fraction of explained variance.")
        results = data[0]
        fig_size = config.get("figure_size") if config.contains("figure_size") else (10, 10)
        key_arguments = {label: {
            "title": f"Singular values and fraction of explained variance of the tensors with label '{" ".join(label)}' of the model '{config.get("model_id")}'",
            "sv_title": f"Singular values of the tensors with label '{" ".join(label)}'",
            "ev_title": f"Fraction of explained variance of the tensors with label '{" ".join(label)}'"
        } for label in results.keys()}
        key_arguments.update(kwargs)

        for label in results.keys():
            # Defining the colormap
            colormap = get_cmap("viridis")
            num_tensors = len(results[label].keys())
            line_colors = colormap(np.linspace(0, 1, num_tensors))

            # Creating the plot
            fig, ax = plt.subplots(1, 2, figsize=(fig_size[1], fig_size[1] / 2))
            #fig.suptitle(key_arguments[label]["title"], fontsize=16)

            for idx, (tensor_key, color) in enumerate(zip(results[label].keys(), line_colors)):
                singular_values = results[label][tensor_key]["singular_values"]
                explained_variance = results[label][tensor_key]["explained_variance"]

                plot_color = self.config.get("first_color", color) if idx == 0 else color
                plot_color = self.config.get("last_color", plot_color) if idx == len(results[label].keys()) - 1 else plot_color

                ax[0].plot(singular_values, label=layer_name_matrix_name_mapping[tensor_key.split(" ")[-1]], color=plot_color)
                ax[1].plot(explained_variance, label=layer_name_matrix_name_mapping[tensor_key.split(" ")[-1]], color=plot_color)

            # Setting the singular values plot properties
            #ax[0].set_title(key_arguments[label]["sv_title"])
            ax[0].set_xlabel("Singular value index")
            ax[0].set_ylabel("Singular value")
            ax[0].legend()
            ax[0].grid(True)

            # Setting the explained variance plot properties
            #ax[1].set_title(key_arguments[label]["ev_title"])
            ax[1].set_xlabel("Number of considered singular values")
            ax[1].set_ylabel("Explained variance")
            ax[1].legend()
            ax[1].grid(True)

            plt.tight_layout()
            # Saving the plot
            storage_path = str(os.path.join(self.get_experiment_path(), f"{config.get("model_id").split("/")[-1]}_singular_values_distribution_{"_".join(label)}.pdf"))
            plt.savefig(storage_path, format="pdf")

    def _plot_rank_analysis(
            self,
            config: Config,
            data: Any,
            kwargs: dict
    ) -> None:
        """
        Plots a heatmap containing the approximated ranks of the model layers.

        Args:
            config (Config):
                The configuration of the experiment.
            data (Any):
                The data obtained from the analysis.
            kwargs (dict):
                Additional keyword arguments.
                Structure:
                    >> { label: { key: value, ... }, ... }
        """

        self.log("Plotting the rank analysis.")

        results = data[0]

        explained_variance_threshold = config.get("explained_variance_threshold") if config.contains("explained_variance_threshold") else 1.
        singular_values_threshold = config.get("singular_values_threshold") if config.contains("singular_values_threshold") else 0.
        relative_rank = config.get("relative_rank") if config.contains("relative_rank") else False

        fig_size_ranks = config.get("fig_size_ranks") if config.contains("fig_size_ranks") else (10, 1)

        heatmap_name = config.get("model_id").split("/")[-1]
        heatmap_name += "_" + config.get("heatmap_name") if config.contains("heatmap_name") else "_heatmap"
        heatmap_name += "_expvar_" + str(explained_variance_threshold).replace('.', '_') + "_sv_" + str(singular_values_threshold).replace('.', '_')

        results_row_concat = {key: item for key, item in results.items() if "row-wise concatenation" in key}
        results_column_concat = {key: item for key, item in results.items() if "column-wise concatenation" in key}
        split_results_list = {"row_wise_concatenation": results_row_concat,  "column_wise_concatenation": results_column_concat}

        for key, split_results in split_results_list.items():
            layer_types = list(split_results.keys())

            key_arguments = {
                # "title": "Rank analysis of the matrices of the model" + f" (explained variance threshold: {explained_variance_threshold})",
                "title": None,
                "axes_displacement": "column",
                # "axis_titles" : [f"Rank values for matrices with label {label}" for label in layer_types],
                "axis_titles": None,
                "x_title": "Layer type",
                "y_title": None,
                "x_labels": [[layer_name_matrix_name_mapping[label[-1]] for label in layer_types], ],
                "y_labels": ["",],
                "x_rotation": 0,
                "fig_size": fig_size_ranks,
                "precision": 2,
                "fontsize": 10
            }
            key_arguments.update(kwargs)

            rank_matrix = np.zeros((1, len(layer_types)))
            relative_rank_matrix = np.zeros((1, len(layer_types)))

            for index_label, label in enumerate(layer_types):
                res = split_results[label][list(split_results[label].keys())[0]]
                rank = res["rank"]
                shape = res["shape"]
                rank_matrix[0, index_label] = rank
                if config.contains("rank_related_to_parameters") and config.get("rank_related_to_parameters"):
                    relative_rank_matrix[0, index_label] = rank / (
                        torch.sqrt(torch.tensor(shape[0]) * torch.tensor(shape[1]))).item()
                else:
                    relative_rank_matrix[0, index_label] = rank / min(shape[0], shape[1])

            if relative_rank:
                all_ranks_to_plot = [rank_matrix, relative_rank_matrix]
            else:
                all_ranks_to_plot = [rank_matrix,]

            plot_heatmap(
                [all_ranks_to_plot,],
                save_path=str(os.path.join(config.get("experiment_root_path"), f"{heatmap_name}_{key}.pdf")),
                index_colormesh=1 if relative_rank else 0,
                **key_arguments,
                vmin=[0,],
                vmax=[1,],
                use_custom_font=False
            )

class DeltaLayersRankAnalysis(TypeSortedRankAnalysis, ABC):
    """
    Class to perform the rank analysis of the delta between layers of a model.
    """

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
            delta_path = minuend_matrices[i].get_path().copy()
            subtrahend_path = subtrahend_matrices[i].get_path().copy()
            delta_path.append("-")
            delta_path.extend(subtrahend_path)
            delta_matrices.append(
                AnalysisTensorWrapper(
                    delta_matrix,
                    name=f"{minuend_matrices[i].get_name()} - {subtrahend_matrices[i].get_name()}",
                    label=f"{minuend_matrices[i].get_label()} - {subtrahend_matrices[i].get_label()}",
                    path=delta_path,
                    block_index=minuend_matrices[i].get_block_index()
                )
            )

        return delta_matrices


class DeltaConsecutiveLayersRankAnalysis(DeltaLayersRankAnalysis):
    """
    Class to perform the rank analysis of the layers of a model.
    """

    def compute_delta_consecutive_matrices(
            self,
            tensors: list[AnalysisTensorWrapper],
    ) -> list[AnalysisTensorWrapper]:
        """
        Compute the delta between consecutive matrices.

        Args:
            tensors (list[AnalysisTensorWrapper]):
                List of matrices.

        Returns:
            list:
                List of delta matrices.
        """

        self.log("Computing delta consecutive matrices...")

        minuend_matrices = tensors[1:].copy()
        subtrahend_matrices = tensors[:-1].copy()

        return self.compute_delta_matrices(minuend_matrices, subtrahend_matrices)

    @override
    def preprocess_extracted_tensors(
            self,
            extracted_tensors: AnalysisTensorDict
    ) -> AnalysisTensorDict:
        """
        Preprocesses the extracted matrices.
        It computes the delta matrices between consecutive layers.

        Args:
            extracted_tensors (AnalysisTensorDict):
                The extracted matrices.
        """

        preprocessed_tensors = AnalysisTensorDict()
        for label in extracted_tensors.get_keys():
            tensor_list = extracted_tensors.get_tensor_list(label)
            delta_tensors = self.compute_delta_consecutive_matrices(tensor_list)

            preprocessed_tensors.set_tensor(label, delta_tensors)

        return preprocessed_tensors

    def get_kwargs_plot_singular_values_distribution(
            self,
            config: Config,
            data: Any
    ) -> dict:
        """
        Gets the keyword arguments to plot the singular values distribution.

        Args:
            config (Config):
                The configuration of the experiment.
            data (Any):
                The data obtained from the analysis.

        Returns:
            dict:
                The keyword arguments.
        """

        results = data[0]
        kwargs = super().get_kwargs_plot_singular_values_distribution(config, data)
        kwargs.update({label: {
            "title": f"Singular values and fraction of explained variance difference between consecutive layers with label '{" ".join(label)}' of the model '{config.get("model_id")}",
            "sv_title": f"Singular values of the tensors with label '{" ".join(label)}'",
            "ev_title": f"Fraction of explained variance of the tensors with label '{" ".join(label)}'"
        } for label in results.keys()})

        return kwargs


class DeltaLayersWRTAverageLayerRankAnalysis(DeltaLayersRankAnalysis):
    """
    Class to perform the rank analysis of the deltas between layers of a model and their average.
    """

    def compute_delta_wrt_average_matrices(
            self,
            tensors: list[AnalysisTensorWrapper],
    ) -> list[AnalysisTensorWrapper]:
        """
        Computes the delta between the average matrix and the rest of the matrices.

        Args:
            tensors (list[AnalysisTensorWrapper]):
                List of matrices.

        Returns:
            list[AnalysisTensorWrapper]:
                List of delta matrices.
        """

        self.log("Computing delta matrices with respect to the average matrix...")

        minuend_matrices = tensors.copy()
        stacked_tensors = torch.stack([matrix.get_tensor() for matrix in tensors])
        average_tensor = torch.mean(stacked_tensors, dim=0)
        layer_name = f"{tensors[0].get_name()}"

        for i in range(len(tensors)):
            layer_name += "_"
            layer_name += tensors[i].get_name()

        average_matrix = AnalysisTensorWrapper(
            average_tensor,
            name=layer_name,
            label=f"avg {layer_name}",
            block_index=tensors[0].get_block_index(),
            path=tensors[0].get_path()
        )

        subtrahend_matrices = [average_matrix] * len(minuend_matrices)

        return self.compute_delta_matrices(minuend_matrices, subtrahend_matrices)

    @override
    def preprocess_extracted_tensors(
            self,
            extracted_tensors: AnalysisTensorDict
    ) -> AnalysisTensorDict:
        """
        Preprocesses the extracted matrices.
        It computes the delta matrices with respect to the average matrix.

        Args:
            extracted_tensors (AnalysisTensorDict):
                The extracted matrices.
        """

        preprocessed_tensors = AnalysisTensorDict()
        for label in extracted_tensors.get_keys():
            tensor_list = extracted_tensors.get_tensor_list(label)
            delta_tensors = self.compute_delta_wrt_average_matrices(tensor_list)

            preprocessed_tensors.append_tensor(label, delta_tensors)

        return preprocessed_tensors

    def get_kwargs_plot_singular_values_distribution(
            self,
            config: Config,
            data: Any
    ) -> dict:
        """
        Gets the keyword arguments to plot the singular values distribution.

        Args:
            config (Config):
                The configuration of the experiment.
            data (Any):
                The data obtained from the analysis.

        Returns:
            dict:
                The keyword arguments.
        """

        results = data[0]
        kwargs = super().get_kwargs_plot_singular_values_distribution(config, data)
        kwargs.update({label: {
            "title": f"Singular values and fraction of explained variance of the deltas between layers with label '{" ".join(label)}' and the other layers with the same label of the model '{config.get("model_id")}",
            "sv_title": f"Singular values of the tensors with label '{" ".join(label)}'",
            "ev_title": f"Fraction of explained variance of the tensors with label '{" ".join(label)}'"
        } for label in results.keys()})

        return kwargs


class AllDeltaLayersRankAnalysis(DeltaLayersRankAnalysis):
    """
    Class to perform the rank analysis of the deltas between all the layers of a model and the others.
    """

    def compute_all_delta_matrices(
            self,
            all_delta_tensors: AnalysisTensorDict,
            tensors: list[AnalysisTensorWrapper],
    ) -> AnalysisTensorDict:
        """
        Compute the delta between each matrix and all the others.

        Args:
            all_delta_tensors (AnalysisTensorDict):
                Dictionary containing the delta matrices organized by the path of the minuend.
            tensors (list[AnalysisTensorWrapper]):
                List of matrices.

        Returns:
            AnalysisTensorDict:
                Dictionary containing the delta matrices organized by the index of the minuend
        """

        self.log("Computing all delta matrices...")

        for i in range(len(tensors)):
            minuend_tensors = [tensors[i]] * len(tensors)
            subtrahend_tensors = tensors.copy()

            delta_tensors_minuend_i = self.compute_delta_matrices(minuend_tensors, subtrahend_tensors)
            key = minuend_tensors[0].get_path().copy()
            all_delta_tensors.append_tensor(key, delta_tensors_minuend_i)

        return all_delta_tensors

    @override
    def preprocess_extracted_tensors(
            self,
            extracted_tensors: AnalysisTensorDict
    ) -> AnalysisTensorDict:
        """
        Preprocesses the extracted matrices.
        It computes the delta matrices between all the layers.

        Args:
            extracted_tensors (AnalysisTensorDict):
                The extracted matrices.
        """

        # If the deltas have to be done between elements with different labels, the code has to be modified here by
        # grouping layers differently.

        preprocessed_tensors = AnalysisTensorDict()
        for label in extracted_tensors.get_keys():
            tensor_list = extracted_tensors.get_tensor_list(label)
            self.compute_all_delta_matrices(preprocessed_tensors, tensor_list)

        return preprocessed_tensors

    def get_kwargs_plot_singular_values_distribution(
            self,
            config: Config,
            data: Any
    ) -> dict:
        """
        Gets the keyword arguments to plot the singular values distribution.

        Args:
            config (Config):
                The configuration of the experiment.
            data (Any):
                The data obtained from the analysis.

        Returns:
            dict:
                The keyword arguments.
        """

        results = data[0]
        kwargs = super().get_kwargs_plot_singular_values_distribution(config, data)
        kwargs.update({label: {
            "title": f"Singular values and fraction of explained variance of the difference between layer {" ".join(label)} in block at index {" ".join(label)} and the other layers of the model '{config.get("model_id")}",
            "sv_title": f"Singular values of the difference between layer {" ".join(label)} and the other layers of the model '{config.get("model_id")}",
            "ev_title": f"Fraction of explained variance of the difference between layer {" ".join(label)} and the other layers of the model '{config.get("model_id")}",
        } for label in results.keys()})

        return kwargs