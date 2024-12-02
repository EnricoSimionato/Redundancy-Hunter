import os

import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Any, override

from matplotlib.cm import get_cmap

import numpy as np

import torch

from exporch import Config
from exporch.utils.causal_language_modeling import load_model_for_causal_lm
from exporch.utils.plot_utils import plot_heatmap

from redhunter.analysis.analysis_utils import AnalysisTensorDict, extract
from redhunter.analysis_experiment import AnalysisExperiment


class RankAnalysis(AnalysisExperiment, ABC):
    """
    Class to perform the rank analysis of the layers of a model.
    """

    def _run_experiment(
            self
    ) -> None:
        """
        Runs the experiment. Performing the operations that are defined in the specific subclass of
        GeneralPurposeExperiment. This method is abstract and must be implemented in the specific subclass.
        """

        if self.get_data() is None:
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
        Extracts the layers from the model.
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

        self.log("Computing the singular values of the layers.")
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

        self.log("Computing the rank of the layers.")

        explained_variance_threshold = self.get_config().get("explained_variance_threshold") if self.get_config().contains("explained_variance_threshold") else 1.
        singular_values_threshold = self.get_config().get("singular_values_threshold") if self.get_config().contains("singular_values_threshold") else 0.

        if explained_variance_threshold <= 0. or explained_variance_threshold > 1.:
            raise ValueError("The threshold on the explained variance must be between 0 and 1.")

        #results = {}
        singular_values_explained_variance = self.get_data()[0]
        for label in singular_values_explained_variance.keys():
            #results[label] = {}
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
        """

        pass

    def _plot_results(
            self,
            config: Config,
            data: Any
    ) -> None:
        """
        Plots the results obtained from the experiment.
        The performed operations will depend on the specific subclass of GeneralPurposeExperiment.

        Args:
            config (Config):
                The configuration of the experiment.
            data (Any):
                The data obtained from the analysis.
        """

        self.log("Plotting singular values and fraction of explained variance.")
        fig_size = config.get("figure_size") if config.contains("figure_size") else (10, 10)
        configuration = self.get_config()

        # Plotting the singular values and fraction of explained variance
        results = self.get_data()[0]
        for label in results.keys():
            colormap = get_cmap("viridis")
            num_tensors = len(results[label].keys())
            line_colors = colormap(np.linspace(0, 1, num_tensors))

            fig, ax = plt.subplots(1, 2, figsize=(fig_size[1], fig_size[1]/2))
            fig.suptitle(f"Singular values and fraction of explained variance of the tensors with label '{" ".join(label)}' of the model '{config.get("model_id")}'", fontsize=16)
            for idx, (tensor_key, color) in enumerate(zip(results[label].keys(), line_colors)):
                singular_values = results[label][tensor_key]["singular_values"]
                explained_variance = results[label][tensor_key]["explained_variance"]

                plot_color = "red" if idx == 0 else color

                ax[0].plot(singular_values, label=tensor_key, color=plot_color)
                ax[1].plot(explained_variance, label=tensor_key, color=plot_color)

                position = int(len(singular_values) * (idx + 1) / (num_tensors + 1))

                # Annotate the chosen position on the singular values plot
                ax[0].text(
                    position,
                    singular_values[position],
                    str(idx),
                    fontsize=10,
                    color=plot_color,
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor=plot_color, facecolor="white"),
                )

                # Annotate the chosen position on the explained variance plot
                ax[1].text(
                    position,
                    explained_variance[position],
                    str(idx),
                    fontsize=10,
                    color=plot_color,
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor=plot_color, facecolor="white"),
                )

            ax[0].set_title(f"Singular values of the tensors with label '{" ".join(label)}'")
            ax[0].set_xlabel("Singular value index")
            ax[0].set_ylabel("Singular value")
            ax[0].legend()
            ax[0].grid(True)

            ax[1].set_title(f"Fraction of explained variance of the tensors with label '{" ".join(label)}'")
            ax[1].set_xlabel("Singular value index")
            ax[1].set_ylabel("Fraction of explained variance")
            ax[1].legend()
            ax[1].grid(True)

            plt.tight_layout()
            # Saving the plot
            plt.savefig(str(os.path.join(config.get("experiment_root_path"), f"singular_values_distribution_{"_".join(label)}.pdf")), format="pdf")

        # Plotting the rank analysis
        explained_variance_threshold = configuration.get("explained_variance_threshold")
        singular_values_threshold = configuration.get("singular_values_threshold")
        heatmap_name = configuration.get("heatmap_name") if configuration.contains("heatmap_name") else "heatmap"
        heatmap_name += "_expvar_" + str(explained_variance_threshold).replace('.', '_') + "_sv_" + str(singular_values_threshold).replace('.', '_')
        layer_types = list(results.keys())
        number_of_blocks = len(list(results.values())[0].keys())

        tensor_ranks_list = []
        tensor_shapes_list = []
        for index_label, label in enumerate(layer_types):
            ranks = np.zeros((1, number_of_blocks))
            relative_ranks = np.zeros((1, number_of_blocks))

            tensor_shapes_list.append([])
            for index_block, tensor_key in enumerate(results[label].keys()):
                rank = results[label][tensor_key]["rank"]
                shape = results[label][tensor_key]["shape"]
                tensor_shapes_list[-1].append(shape)
                ranks[0, index_block] = rank

                if configuration.contains("relative_rank") and configuration.get("relative_rank"):

                    relative_rank = round(rank / (torch.sqrt(torch.tensor(shape[0]) * torch.tensor(shape[1]))).item(), 2)
                    relative_ranks[0, index_block] = relative_rank

            if configuration.contains("relative_rank") and configuration.get("relative_rank"):
                tensor_ranks_list.append([ranks, relative_ranks])
            else:
                tensor_ranks_list.append([ranks])

        plot_heatmap(
            tensor_ranks_list,
            str(os.path.join(config.get("experiment_root_path"), f"{heatmap_name}.pdf")),
            "Rank analysis of the matrices of the model" + f" (explained variance threshold: {explained_variance_threshold})",

            axes_displacement="column",
            axis_titles=[f"Rank values for matrices with label {label}" for label in layer_types],
            #interval=[{"min": 0, "max": min(shape)} for label in layer_types],
            x_title="Block indexes",
            y_title="Layer type",
            x_labels=[[str(i) for i in range(number_of_blocks)] for _ in layer_types],
            y_labels=[[label] for label in layer_types],
            fig_size=(fig_size[1], fig_size[1]/4 * len(layer_types)),
            precision=2,
            fontsize=10,
            vmin=[0 for _ in layer_types],
            vmax=[min(min(tuple(shape)) for shape in shape_list) for shape_list in tensor_shapes_list]
        )


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
        preprocessed_tensors = self.preprocess_extracted_tensors(extracted_tensors)

        sorted_tensors = preprocessed_tensors.group_by_name()

        return sorted_tensors


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
