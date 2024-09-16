from __future__ import annotations

from typing import Any
import pickle as pkl

import numpy as np

import torch
import torch.nn as nn

import transformers

import re

from exporch import Config, Verbose

from exporch.utils.causal_language_modeling import load_model_for_causal_lm

from redhunter.analysis.rank_analysis_utils import (
    compute_explained_variance,
    compute_singular_values,
    RankAnalysisResult
)


# Definition of the classes to perform the rank analysis

class AnalysisTensorWrapper:
    """
    Wrapper for the analysis of a tensor.

    Args:
        tensor (torch.Tensor):
            The tensor.
        name (str, optional):
            The name of the tensor. Defaults to None.
        label (str, optional):
            The label of the tensor. Defaults to None.
        path (str, optional):
            The path of the tensor. Defaults to None.
        block_index (int, optional):
            The block index of the tensor. Defaults to None.
        layer (nn.Module, optional):
            The layer of the tensor. Defaults to None.
        precision (int, optional):
            The precision of the relative rank of the tensor. Default to 2.
        verbose (Verbose, optional):
            The verbosity level. Default to Verbose.INFO.

    Attributes:
        tensor (torch.Tensor):
            The tensor.
        name (str):
            The name of the tensor.
        label (str):
            The label of the tensor.
        path (str):
            The path of the tensor.
        block_index (int):
            The block index of the tensor.
        layer (nn.Module):
            The layer of the tensor.
        singular_values (np.ndarray):
            The singular values of the tensor.
        rank_analysis_results (list):
            The list of rank analysis
    """

    def __init__(
            self,
            tensor: torch.Tensor,
            name: str = None,
            label: str = None,
            path: str = None,
            block_index: int = None,
            layer: nn.Module = None,
            precision: int = 2,
            verbose: Verbose = Verbose.INFO
    ) -> None:

        self.tensor = tensor
        self.name = name
        self.label = label
        self.path = path
        self.block_index = block_index
        self.layer = layer

        self.precision = precision

        self.verbose = verbose

        self.singular_values = None
        self.rank_analysis_results = []
        
        self.norm = None

        self.attributes = {}

    def get_tensor(
            self,
            numpy_array: bool = False
    ) -> [np.ndarray | torch.Tensor]:
        """
        Returns the tensor.

        Returns:
            [np.ndarray | torch.Tensor]:
                The tensor.
        """

        if numpy_array:
            return self.tensor.detach().numpy()
        else:
            return self.tensor.detach()

    def get_name(
            self
    ) -> str:
        """
        Returns the name of the tensor.

        Returns:
            str:
                The name of the tensor.
        """

        return self.name

    def get_label(
            self
    ) -> str:
        """
        Returns the label of the tensor.

        Returns:
            str:
                The label of the tensor.
        """

        return self.label

    def get_path(
            self
    ) -> str:
        """
        Returns the path of the tensor.

        Returns:
            str:
                The path of the tensor.
        """

        return self.path

    def get_block_index(
            self
    ) -> int:
        """
        Returns the block index of the tensor.

        Returns:
            int:
                The block index of the tensor.
        """

        return self.block_index

    def get_layer(
            self
    ) -> nn.Module:
        """
        Returns the layer of the tensor.

        Returns:
            nn.Module:
                The layer of the tensor.
        """

        return self.layer

    def get_precision(
            self
    ) -> int:
        """
        Returns the precision of the relative rank of the tensor.

        Returns:
            int:
                The precision of the relative rank of the tensor.
        """

        return self.precision

    def get_verbose(
            self
    ) -> Verbose:
        """
        Returns the verbosity level.

        Returns:
            Verbose:
                The verbosity level.
        """

        return self.verbose

    def get_singular_values(
            self
    ) -> np.ndarray:
        """
        Returns the singular values of the tensor.

        Returns:
            np.ndarray:
                The singular values of the tensor.
        """

        return self.singular_values

    def get_explained_variance(
            self
    ) -> None:
        """
        Returns the explained variance of the tensor.

        Returns:
            np.ndarray:
                The explained variance of the tensor.
        """

        return compute_explained_variance(self.singular_values)

    def get_rank(
            self,
            explained_variance_threshold: float = 0,
            singular_values_threshold: float = 0,
            relative: bool = True
    ) -> [int | float]:
        """
        Returns the rank of the tensor.

        Args:
            explained_variance_threshold (float, optional):
                The threshold on the explained variance to use to compute the rank. Rank is computed as the number of
                singular values that explain the threshold fraction of the total variance. Defaults to 0.
            singular_values_threshold (float, optional):
                The threshold to use to compute the rank based on singular values. Rank is computed as the number of
                singular values that are greater than the threshold. Defaults to 0.
            relative (bool, optional):
                Whether to return the relative rank. Defaults to True.

        Returns:
            [int | float]:
                The rank of the tensor.
        """

        for rank_analysis_result in self.rank_analysis_results:
            if rank_analysis_result.get_explained_variance_threshold() == explained_variance_threshold and \
                    rank_analysis_result.get_singular_values_threshold() == singular_values_threshold:
                rank = rank_analysis_result.get_rank()
                if relative:
                    shape = self.get_shape()
                    rank = round(rank / (torch.sqrt(torch.tensor(shape[0]) * torch.tensor(shape[1]))).item(), self.precision)
                return rank

        self._perform_rank_analysis(explained_variance_threshold, singular_values_threshold)

        rank = self.get_rank(explained_variance_threshold, singular_values_threshold, relative)

        return rank

    def get_shape(
            self
    ) -> torch.Size:
        """
        Returns the shape of the tensor.

        Returns:
            torch.Size:
                The shape of the tensor.
        """

        return self.tensor.shape

    def get_norm(
            self
    ) -> torch.Tensor:
        """
        Returns the norm of the tensor.

        Returns:
            torch.Tensor:
                The norm of the tensor.
        """

        if self.norm is None:
            self.norm = torch.norm(self.tensor)
        return self.norm
    
    def get_attribute(
            self,
            key: str
    ) -> Any:
        """
        Returns the attribute given the key.

        Args:
            key (str):
                The key of the attribute.

        Returns:
            Any:
                The attribute.
        """

        return self.attributes[key]

    def get_parameters_count(
            self
    ) -> int:
        """
        Returns the number of parameters of the tensor.

        Returns:
            int:
                The number of parameters of the tensor.
        """

        return self.tensor.numel()

    def get_parameters_count_thresholded(
            self,
            explained_variance_threshold: float = 0,
            singular_values_threshold: float = 0
    ) -> int:
        """
        Returns the number of parameters of the tensor.

        Args:
            explained_variance_threshold (float, optional):
                The threshold on the explained variance to use to compute the rank. Rank is computed as the number of
                singular values that explain the threshold fraction of the total variance. Defaults to 0.
            singular_values_threshold (float, optional):
                The threshold to use to compute the rank based on singular values. Rank is computed as the number of
                singular values that are greater than the threshold. Defaults to 0.

        Returns:
            int:
                The number of parameters of the tensor.
        """

        rank = self.get_rank(explained_variance_threshold, singular_values_threshold, relative=False)

        return rank * self.get_shape()[0] + rank * self.get_shape()[1]

    def set_tensor(
            self,
            tensor: torch.Tensor
    ) -> None:
        """
        Sets the tensor.

        Args:
            tensor (torch.Tensor):
                The tensor.
        """

        self.tensor = tensor

    def set_dtype(
            self,
            dtype: torch.dtype
    ) -> None:
        """
        Sets the dtype of the tensor.

        Args:
            dtype (torch.dtype):
                The dtype of the tensor.
        """

        self.tensor = self.tensor.to(dtype)

    def set_device(
            self,
            device: torch.device
    ) -> None:
        """
        Sets the device of the tensor.

        Args:
            device (torch.device):
                The device of the tensor.
        """

        self.tensor = self.tensor.to(device)

    def set_name(
            self,
            name: str
    ) -> None:
        """
        Sets the name of the tensor.

        Args:
            name (str):
                The name of the tensor.
        """

        self.name = name

    def set_label(
            self,
            label: str
    ) -> None:
        """
        Sets the label of the tensor.

        Args:
            label (str):
                The label of the tensor.
        """

        self.label = label

    def set_path(
            self,
            path: str
    ) -> None:
        """
        Sets the path of the tensor.

        Args:
            path (str):
                The path of the tensor.
        """

        self.path = path

    def set_block_index(
            self,
            block_index: int
    ) -> None:
        """
        Sets the block index of the tensor.

        Args:
            block_index (int):
                The block index of the tensor.
        """

        self.block_index = block_index

    def set_layer(
            self,
            layer: nn.Module
    ) -> None:
        """
        Sets the layer of the tensor.

        Args:
            layer (nn.Module):
                The layer of the tensor.
        """

        self.layer = layer

    def set_singular_values(
            self,
            singular_values: np.ndarray
    ) -> None:
        """
        Sets the singular values of the tensor.

        Args:
            singular_values (np.ndarray):
                The singular values of the tensor.
        """

        self.singular_values = singular_values

    def set_attribute(
            self,
            key: str,
            value: Any
    ) -> None:
        """
        Sets the attribute given the key.

        Args:
            key (str):
                The key of the attribute.
            value (Any):
                The value of the attribute.
        """

        self.attributes[key] = value

    def append_rank_analysis_result(
            self,
            rank_analysis_result: RankAnalysisResult
    ) -> None:
        """
        Appends a rank analysis result.

        Args:
            rank_analysis_result (RankAnalysisResult):
                The rank analysis result.
        """

        self.rank_analysis_results.append(rank_analysis_result)

    def delete_rank_analysis_result(
            self,
            explained_variance_threshold: float = 0,
            singular_values_threshold: float = 0
    ) -> None:
        """
        Deletes a rank analysis result.

        Args:
            explained_variance_threshold (float, optional):
                The threshold on the explained variance to use to compute the rank. Rank is computed as the number of
                singular values that explain the threshold fraction of the total variance. Defaults to 0.
            singular_values_threshold (float, optional):
                The threshold to use to compute the rank based on singular values. Rank is computed as the number of
                singular values that are greater than the threshold. Defaults to 0.
        """

        for rank_analysis_result in self.rank_analysis_results:
            if rank_analysis_result.get_explained_variance_threshold() == explained_variance_threshold and \
                    rank_analysis_result.get_singular_values_threshold() == singular_values_threshold:
                self.rank_analysis_results.remove(rank_analysis_result)

    def delete_rank_analyses(
            self
    ) -> None:
        """
        Deletes all rank analyses.
        """

        self.rank_analysis_results = []

    def compute_singular_values(
            self
    ) -> None:
        """
        Computes the singular values of the tensor.
        """

        self.set_singular_values(compute_singular_values(self.get_tensor(numpy_array=True)))

    def _perform_rank_analysis(
            self,
            explained_variance_threshold: float = 0,
            singular_values_threshold: float = 0
    ) -> None:
        """
        Performs the rank analysis of the tensor.

        Args:
            explained_variance_threshold (float, optional):
                The threshold on the explained variance to use to compute the rank. Rank is computed as the number of
                singular values that explain the threshold fraction of the total variance.
                Defaults to 0.
            singular_values_threshold (float, optional):
                The threshold to use to compute the rank based on singular values. Rank is computed as the number of
                singular values that are greater than the threshold.
                Defaults to 0.
        """

        if self.singular_values is None:
            self.compute_singular_values()

        if explained_variance_threshold <= 0. or explained_variance_threshold > 1.:
            raise ValueError("The threshold on the explained variance must be between 0 and 1.")

        explained_variance = compute_explained_variance(self.singular_values)

        rank_based_on_explained_variance = np.argmax(explained_variance >= explained_variance_threshold) + 1

        if self.singular_values[-1] > singular_values_threshold:
            rank_based_on_singular_values = len(self.singular_values)
        else:
            rank_based_on_singular_values = np.argmax(self.singular_values < singular_values_threshold)

        rank = np.minimum(
            rank_based_on_explained_variance,
            rank_based_on_singular_values
        )

        self.append_rank_analysis_result(
            RankAnalysisResult(
                rank=rank,
                explained_variance_threshold=explained_variance_threshold,
                singular_values_threshold=singular_values_threshold
            )
        )


class AnalysisTensorDict:
    """
    Dictionary of tensors for the analysis.

    Args:
        keys ([list[tuple[Any, ...]] | list[Any]], optional):
            The keys of the tensors. Defaults to None.
        tensors ([list[list[AnalysisTensorWrapper]] | list[AnalysisTensorWrapper]], optional):
            The tensors to add to the dictionary.
        verbose (Verbose, optional):
            The verbosity level. Defaults to Verbose.INFO.

    Raises:
        ValueError:
            If the number of keys is different from the number of tensors.
        ValueError:
            If all keys do not have the same length.

    Attributes:
        tensors (dict):
            The dictionary of tensors.
        verbose (Verbose):
            The verbosity level.
    """

    def __init__(
            self,
            keys: [list[tuple[Any, ...]] | list[Any]] = (),
            tensors: [list[list[AnalysisTensorWrapper]] | list[AnalysisTensorWrapper]] = (),
            verbose: Verbose = Verbose.INFO
    ) -> None:

        if len(tensors) != len(keys):
            raise ValueError("The number of keys must be equal to the number of tensors.")
        if len(keys) > 0:
            for index in range(len(keys)):
                if not isinstance(keys[index], tuple):
                    keys[index] = (keys[index],)
            keys_length = len(keys[0])
            for key in keys:
                if len(key) != keys_length:
                    raise ValueError("All keys must have the same length.")

        self.tensors = {}
        for index in range(len(keys)):
            self.append_tensor(
                keys[index],
                tensors[index]
            )

        self.verbose = verbose

    def get_tensor(
            self,
            key: [tuple[Any, ...] | Any],
            index: int = 0
    ) -> AnalysisTensorWrapper:
        """
        Returns the tensor given the key.

        Args:
            key ([tuple[Any, ...] | Any]):
                The key of the tensor.
            index (int, optional):
                The index of the tensor in the list. Defaults to 0.

        Returns:
            AnalysisTensorWrapper:
                The tensor.
        """

        if not isinstance(key, tuple):
            key = (key,)

        return self.tensors[key][index]

    def get_tensor_list(
            self,
            key: [tuple[Any, ...] | Any]
    ) -> list[AnalysisTensorWrapper]:
        """
        Returns the list of tensors given the key.

        Args:
            key ([tuple[Any, ...] | Any]):
                The key of the tensors.

        Returns:
            list[AnalysisTensorWrapper]:
                The list of tensors.
        """

        if not isinstance(key, tuple):
            key = (key,)

        return self.tensors[key]

    def get_keys(
            self
    ) -> list[tuple[Any, ...]]:
        """
        Returns the keys of the dictionary.

        Returns:
            list[tuple[Any, ...]]:
                The keys of the dictionary.
        """

        return list(self.tensors.keys())

    def get_unique_positional_keys(
            self,
            position: int,
            sort: bool = False
    ) -> list[Any]:
        """
        Returns the unique keys at a given position.

        Args:
            position (int):
                The position of the keys.
            sort (bool, optional):
                Whether to return the keys sorted. Defaults to False.

        Returns:
            list[Any]:
                The unique keys at the given position.
        """

        unique_keys = list(set(
            key[position]
            for key in self.tensors.keys()
        ))

        if sort:
            unique_keys.sort()

        return unique_keys

    def set_tensor(
            self,
            key: [tuple[Any, ...] | Any],
            tensor: [list[AnalysisTensorWrapper] | AnalysisTensorWrapper]
    ) -> None:
        """
        Sets a tensor or a list of tensors to the dictionary.

        Args:
            key ([tuple[Any, ...] | Any]):
                The key of the tensor.
            tensor ([list[AnalysisTensorWrapper] | AnalysisTensorWrapper]):
                The tensor or list of tensors to set.

        Raises:
            ValueError:
                If the key is not a tuple.
        """

        if not isinstance(key, tuple):
            key = (key,)

        if isinstance(tensor, list):
            self.tensors[key] = tensor
        else:
            self.tensors[key] = [tensor]

    def append_tensor(
            self,
            key: [tuple[Any, ...] | Any],
            tensor: [list[AnalysisTensorWrapper] | AnalysisTensorWrapper]
    ) -> None:
        """
        Appends a tensor to the dictionary.

        Args:
            key ([tuple[Any, ...] | Any]):
                The key of the tensor.
            tensor (AnalysisTensorWrapper):
                The tensor or list of tensors to append.
        """

        if not isinstance(key, tuple):
            key = (key,)

        if key in self.tensors.keys():
            if isinstance(tensor, list):
                self.tensors[key] = self.tensors[key] + tensor
            else:
                self.tensors[key] = self.tensors[key] + [tensor]
        else:
            self.set_tensor(key, tensor)

    def set_dtype(
            self,
            dtype: torch.dtype
    ) -> None:
        """
        Sets the dtype of the tensors.

        Args:
            dtype (torch.dtype):
                The dtype of the tensors.
        """

        for key in self.tensors.keys():
            for tensor_wrapper in self.tensors[key]:
                tensor_wrapper.set_dtype(dtype)

    def set_device(
            self,
            device: torch.device
    ) -> None:
        """
        Sets the device of the tensors.

        Args:
            device (torch.device):
                The device of the tensors.
        """

        for key in self.tensors.keys():
            for tensor_wrapper in self.tensors[key]:
                tensor_wrapper.set_device(device)

    def filter_by_positional_key(
            self,
            key: Any,
            position: int
    ) -> 'AnalysisTensorDict':
        """
        Filters the tensors given a key and a position.

        Args:
            key (Any):
                The key to filter.
            position (int):
                The position of the key.

        Returns:
            AnalysisTensorDict:
                The filtered dictionary of tensors.
        """

        filtered_tensors = AnalysisTensorDict()
        for tensor_key in self.tensors.keys():
            if tensor_key[position] == key:
                filtered_tensors.append_tensor(
                    tensor_key,
                    self.tensors[tensor_key]
                )

        return filtered_tensors


class AnalysisLayerWrapper(nn.Module):
    """
    Wrapper for the analysis of a layer.

    Args:
        layer (nn.Module):
            The layer.
        label (str, optional):
            The label of the layer. Defaults to None.

    Attributes:
        layer (nn.Module):
            The layer.
        label (str):
            The label of the layer.
        activations (list):
            The list of activations.
    """

    def __init__(
            self,
            layer: nn.Module,
            label: str = None,
            store_activations: bool = False,
            *args,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.layer = layer
        self.label = label

        self.store_activations = store_activations

        self.activations = []
        self.num_activations = None
        self.sum_activations = None
        self.sum_squared_activations = None

    def get_layer(
            self
    ) -> nn.Module:
        """
        Returns the layer.

        Returns:
            nn.Module:
                The layer.
        """

        return self.layer

    def get_label(
            self
    ) -> str:
        """
        Returns the label of the layer.

        Returns:
            str:
                The label of the layer.
        """

        return self.label

    def get_store_activations(
            self
    ) -> bool:
        """
        Returns whether to store the activations.

        Returns:
            bool:
                Whether to store the activations.
        """

        return self.store_activations

    def set_store_activations(
            self,
            store_activations: bool
    ) -> None:
        """
        Sets whether to store the activations.

        Args:
            store_activations (bool):
                Whether to store the activations.
        """

        self.store_activations = store_activations

    def get_activations(
            self
    ) -> list:
        """
        Returns the activations.

        Returns:
            list:
                The activations.
        """

        return self.activations

    def get_mean_activations(
            self
    ) -> torch.Tensor:
        """
        Returns the mean of the activations.

        Returns:
            torch.Tensor:
                The mean of the activations.
        """

        return self.sum_activations / self.num_activations

    def get_variance_activations(
            self
    ) -> torch.Tensor:
        """
        Returns the variance of the activations.

        Returns:
            torch.Tensor:
                The variance of the activations.
        """

        return self.sum_squared_activations / self.num_activations - (self.sum_activations / self.num_activations) ** 2

    def get_stats(
            self
    ) -> dict:
        """
        Returns the statistics of the activations.

        Returns:
            dict:
                The statistics of the activations.
        """

        return {
            "path": self.label,
            "activations": self.activations,
            "mean_activations": self.get_mean_activations(),
            "variance_activations": self.get_variance_activations()
        }

    def set_label(
            self,
            label: str
    ) -> None:
        """
        Sets the label of the layer.

        Args:
            label (str):
                The label of the layer.
        """

        self.label = label

    def set_activations(
            self,
            activations: list
    ) -> None:
        """
        Sets the activations.

        Args:
            activations (list):
                The activations.
        """

        self.activations = activations

    def reset_activations(
            self
    ) -> None:
        """
        Resets the activations.
        """

        self.activations = []
        self.num_activations = None
        self.sum_activations = None
        self.sum_squared_activations = None

    def forward(
            self,
            *args,
            **kwargs
    ) -> torch.Tensor:
        """
        Forward pass of the layer.

        Args:
            x (torch.Tensor):
                The input tensor.

        Returns:
            torch.Tensor:
                The output tensor.
        """

        output = self.layer(*args, **kwargs)

        if self.store_activations:
            self.activations.append(output.detach().cpu())

        flattened_output = output.detach().view(-1, output.shape[-1])
        if self.num_activations is None:
            self.num_activations = int(flattened_output.shape[0])
            self.sum_activations = torch.sum(flattened_output, dim=0)
            self.sum_squared_activations = torch.sum(flattened_output ** 2, dim=0)
        else:
            self.num_activations += int(flattened_output.shape[0])
            self.sum_activations.add_(torch.sum(flattened_output, dim=0))
            self.sum_squared_activations.add_(torch.sum(flattened_output ** 2, dim=0))

        return output


class AnalysisModelWrapper(nn.Module):
    """
    Wrapper for the analysis of a model.

    Args:
        model (nn.Module):
            The model.
        *args:
            Additional positional arguments.
        **kwargs:
            Additional keyword arguments.

    Attributes:
        model (nn.Module):
            The model.
    """

    def __init__(
            self,
            model: [nn.Module | transformers.AutoModel | transformers.PreTrainedModel],
            targets: list,
            black_list: list = None,
            store_activations: bool = True,
            *args,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.model = model

        self.targets = targets
        self.black_list = black_list

        self.wrap_model(self.model, self.targets, self.black_list)

        self.store_activations = store_activations
        self.set_store_activations(self.store_activations)

    def wrap_model(
            self,
            module_tree: nn.Module,
            paths_of_targets: list,
            black_list: list = None,
            path: str = "",
            verbose: Verbose = Verbose.SILENT,
            **kwargs
    ) -> None:
        """
        Converts layers into global-dependent versions.

        Args:
            module_tree (nn.Module):
                Model or module containing layers.
            paths_of_targets (list):
                List of paths of the targets.
            black_list (list, optional):
                List of strings to blacklist. Defaults to None.
            path (str):
                Path to the current layer.
            verbose (Verbose):
                Level of verbosity.
            **kwargs:
                Additional keyword arguments.
        """

        for layer_name in module_tree._modules.keys():
            # Extracting the child from the current module
            child = module_tree._modules[layer_name]
            # If the child has no children, the layer is an actual computational layer of the model
            if len(child._modules) == 0:
                if black_list is not None:
                    black_listed = len([
                        black_listed_string
                        for black_listed_string in black_list
                        if black_listed_string in path + "_" + layer_name
                    ]) > 0
                else:
                    black_listed = False

                targets_in_path = [
                    layer_path_
                    for layer_path_ in paths_of_targets
                    if layer_path_ in path + "_" + layer_name and not black_listed
                ]
                if len(targets_in_path) > 0:
                    layer_path = str(max(targets_in_path, key=len))
                    if verbose > Verbose.SILENT:
                        print(f"Wrapped {layer_path} in {path}")

                    # Creating a wrapper for the layer
                    layer_wrapper = AnalysisLayerWrapper(child, path + (f"{layer_name}" if path == "" else f"_{layer_name}"))
                    # Setting the wrapper as the child of the current module
                    module_tree._modules[layer_name] = layer_wrapper
            else:
                # Recursively calling the method on the child, if the child has children
                self.wrap_model(
                    child,
                    paths_of_targets,
                    black_list,
                    path + (f"{layer_name}" if path == "" else f"_{layer_name}"),
                    verbose=verbose,
                    **kwargs
                )

    def forward(
            self,
            *args,
            **kwargs
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            *args:
                Additional positional arguments.
            **kwargs:
                Additional keyword arguments.

        Returns:
            torch.Tensor:
                The output tensor.
        """

        return self.model(*args, **kwargs)

    def feed_input_activation(
            self,
            activations: torch.Tensor,
            reset_before_feeding: bool = True
    ) -> dict:
        """
        Feeds the input activation to the wrapped layers of the model and returns the activations.

        Args:
            activations (torch.Tensor):
                The input activation.
            reset_before_feeding (bool, optional):
                Whether to reset the activations before feeding the input activation. Defaults to True.

        Returns:
            dict:
                The activations.
        """

        if reset_before_feeding:
            self.reset_activations()

        # Recursively feeding the input activation to the model
        self._feed_input_activation(self.model, activations)

        return self.get_activations()

    def _feed_input_activation(
            self,
            module_tree,
            activations: torch.Tensor
    ) -> None:
        """
        Feeds the input activation to the wrapped layers of the model.

        Args:
            module_tree (nn.Module):
                Model or module containing layers.
            activations (torch.Tensor):
                The input activation.
        """

        for layer_name in module_tree._modules.keys():
            child = module_tree._modules[layer_name]
            if issubclass(type(child), AnalysisLayerWrapper):
                child.forward(activations)
            elif len(child._modules) == 0:
                pass
            else:
                self._feed_input_activation(child, activations)

    def set_store_activations(
            self,
            store_activations: bool
    ) -> None:
        """
        Sets whether to store the activations.

        Args:
            store_activations (bool):
                Whether to store the activations.
        """

        self.store_activations = store_activations

        self._set_store_activations(self.model, store_activations)

    def _set_store_activations(
            self,
            module_tree: nn.Module,
            store_activations: bool
    ) -> None:
        """
        Sets whether to store the activations.

        Args:
            module_tree (nn.Module):
                Model or module containing layers.
            store_activations (bool):
                Whether to store the activations.
        """

        for layer_name in module_tree._modules.keys():
            child = module_tree._modules[layer_name]
            if issubclass(type(child), AnalysisLayerWrapper):
                child.set_store_activations(store_activations)
            else:
                self._set_store_activations(child, store_activations)

    def get_activations(
            self,
    ) -> dict:
        """
        Returns the activations.

        Returns:
            dict:
                The activations.
        """

        return self._get_activations(self.model)

    def _get_activations(
            self,
            module_tree: nn.Module
    ) -> dict:
        """
        Returns the activations.

        Returns:
            dict:
                The activations.
        """

        structure = {}
        for layer_name in module_tree._modules.keys():
            # Extracting the child from the current module
            child = module_tree._modules[layer_name]
            if issubclass(type(child), AnalysisLayerWrapper):
                structure[layer_name] = child.get_stats()
            elif len(child._modules) == 0:
                structure[layer_name] = {
                    "path": None,
                    "activations": None,
                    "mean_activations": None,
                    "variance_activations": None
                }
            else:
                structure[layer_name] = self._get_activations(child)

        return structure

    def reset_activations(
            self
    ) -> None:
        """
        Resets the activations.
        """

        self._reset_activations(self.model)

    def _reset_activations(
            self,
            module_tree: nn.Module
    ) -> None:
        """
        Resets the activations.

        Args:
            module_tree (nn.Module):
                Model or module containing layers.
        """

        for layer_name in module_tree._modules.keys():
            child = module_tree._modules[layer_name]
            if issubclass(type(child), AnalysisLayerWrapper):
                child.reset_activations()
            elif len(child._modules) == 0:
                pass
            else:
                self._reset_activations(child)


# Definition of the function to perform various types of analysis

def perform_analysis(
        configuration: Config
) -> None:
    """
    Performs the analysis.

    Args:
        configuration (Config):
            The configuration object containing the necessary information to perform the analysis.

    """

    # Getting the parameters related to the paths from the configuration
    file_available = configuration.get("file_available")
    file_path = configuration.get("file_path")
    directory_path = configuration.get("directory_path")
    file_name_no_format = configuration.get("file_name_no_format")

    if file_available:
        print(f"The file '{file_path}' is available.")
        # Loading the data from the file
        with open(file_path, "rb") as f:
            data = pkl.load(f)
    else:
        # Loading the model
        model = load_model_for_causal_lm(configuration)
        # Extracting the tensors to be analyzed
        extracted_tensors = []
        extract_based_on_path(
            model,
            configuration.get("targets"),
            extracted_tensors,
            configuration.get("black_list"),
            verbose=configuration.get_verbose()
        )

        data = []

        # Saving the data about the analysis to the file
        with open(file_path, "wb") as f:
            pkl.dump(data, f)

    # Saving the data about the analysis to the file
    with open(file_path, "wb") as f:
        pkl.dump(data, f)


# Definition of the functions to compute the rank of a matrix

def compute_rank(
        singular_values: dict,
        threshold: float = 0,
        s_threshold: float = 0
) -> dict:
    """
    Computes the rank of a matrix considering negligible eigenvalues that are very small or that provide a very small
    change in terms of fraction of explained variance.

    Args:
        singular_values (dict):
            The singular values of the matrices of the model.
            The dictionary has the following structure:
            >> {
            >>    "layer_name": {
            >>        "s": [np.array, ...]
            >>    }
            >> }
        threshold (float):
            The threshold on the explained variance to use to compute the rank. Rank is computed as the number of
            singular values that explain the threshold fraction of the total variance.
        s_threshold (float):
            The threshold to use to compute the rank based on singular values. Rank is computed as the number of
            singular values that are greater than the threshold.

    Returns:
        dict:
            The ranks given the input singular values.
    """

    ranks = {}
    for layer_name in singular_values.keys():
        ranks[layer_name] = []

        for s in singular_values[layer_name]["s"]:
            explained_variance = compute_explained_variance(s)
            rank_based_on_explained_variance = np.argmax(explained_variance > threshold)
            if s[-1] < s_threshold:
                rank_based_on_explained_variance = len(explained_variance)

            rank_based_on_singular_values = np.argmax(s < s_threshold)
            if s[-1] > s_threshold:
                rank_based_on_singular_values = len(s)

            rank = np.minimum(rank_based_on_explained_variance, rank_based_on_singular_values)

            ranks[layer_name].append(rank)

    return ranks


def compute_max_possible_rank(
        analyzed_matrices: AnalysisTensorDict
) -> int:
    """
    Computes the maximum possible rank of the matrices of the model.

    Args:
        analyzed_matrices (AnalysisTensorDict):
            The analyzed matrices of the model.

    Returns:
        int:
            The maximum possible rank of the matrices of the model.
    """

    max_possible_rank = 0
    for key in analyzed_matrices.get_keys():
        for analyzed_matrix in analyzed_matrices.get_tensor_list(key):
            singular_values = analyzed_matrix.get_singular_values()
            max_possible_rank = max(max_possible_rank, len(singular_values))

    return max_possible_rank


# Definition of the functions to extract the matrices from the model tree

def extract(
        model_tree: nn.Module,
        names_of_targets: list,
        extracted_matrices: list,
        path: list = [],
        verbose: bool = False,
        **kwargs
) -> None:
    """
    Extracts the matrices from the model tree.

    Args:
        model_tree (nn.Module):
            The model tree.
        names_of_targets (list):
            The names of the targets.
        extracted_matrices (list):
            The list of extracted matrices.
        path (list, optional):
            The path to the current layer. Defaults to [].
        verbose (bool, optional):
            Whether to print the layer name. Defaults to False.
    """

    for layer_name in model_tree._modules.keys():
        child = model_tree._modules[layer_name]
        if len(child._modules) == 0:
            if layer_name in names_of_targets:
                if verbose:
                    print(f"Found {layer_name} in {path}")

                extracted_matrices.append(
                    {
                        "weight": child.weight.detach().numpy(),
                        "layer_name": layer_name,
                        "label": [el for el in path if re.search(r'\d', el)][0],
                        "path": path
                    }
                )
        else:
            new_path = path.copy()
            new_path.append(layer_name)
            extract(
                child,
                names_of_targets,
                extracted_matrices,
                new_path,
                verbose=verbose,
                **kwargs
            )


def extract_based_on_path(
        model_tree: [nn.Module | transformers.AutoModel],
        paths_of_targets: list,
        extracted_matrices: list,
        black_list: list = None,
        path: str = "",
        verbose: Verbose = Verbose.INFO,
        **kwargs
) -> None:
    """
    Extracts the matrices from the model tree.

    Args:
        model_tree ([nn.Module | transformers.AutoModel]):
            The model tree.
        paths_of_targets (list):
            The path of the targets.
        extracted_matrices (list):
            The list of extracted matrices.
        black_list (list, optional):
            The list of black listed paths. Defaults to None.
        path (str, optional):
            The path to the current layer. Defaults to "".
        verbose (Verbose, optional):
            The verbosity level. Defaults to Verbose.INFO.
    """

    for layer_name in model_tree._modules.keys():
        child = model_tree._modules[layer_name]
        if len(child._modules) == 0:
            if verbose > Verbose.INFO:
                print(f"Checking {layer_name} in {path}")

            if black_list is not None:
                black_listed = len([
                    black_listed_string
                    for black_listed_string in black_list
                    if black_listed_string in path + "_" + layer_name
                ]) > 0
            else:
                black_listed = False

            targets_in_path = [
                layer_path_
                for layer_path_ in paths_of_targets
                if layer_path_ in path + "_" + layer_name and not black_listed
            ]
            if len(targets_in_path) > 0:
                layer_path = str(max(targets_in_path, key=len))
                if verbose > Verbose.SILENT:
                    print(f"Found {layer_path} in {path}")

                list_containing_layer_number = [
                    sub_path for sub_path in path.split("_") if sub_path.isdigit()
                ]
                block_index = list_containing_layer_number[0] if len(list_containing_layer_number) > 0 else "-1"
                extracted_matrices.append(
                    AnalysisTensorWrapper(
                        tensor=child.weight.detach(),
                        name=layer_name,
                        label=layer_path,
                        path=path + "_" + layer_name,
                        block_index=int(block_index),
                        layer=child
                    )
                )
        else:
            # Recursively calling the function
            extract_based_on_path(
                model_tree=child,
                paths_of_targets=paths_of_targets,
                extracted_matrices=extracted_matrices,
                black_list=black_list,
                path=layer_name if path == "" else path + "_" + layer_name,
                verbose=verbose,
                **kwargs
            )


def extract_analysis_layer_wrappers(
        module_tree: [nn.Module | transformers.AutoModel],
        paths_of_targets: list,
        extracted_matrices: list,
        black_list: list = None,
        path: str = "",
        verbose: Verbose = Verbose.INFO,
        **kwargs
) -> None:
    """
    Extracts the matrices from the model tree.

    Args:
        module_tree ([nn.Module | transformers.AutoModel]):
            The model tree.
        paths_of_targets (list):
            The path of the targets.
        extracted_matrices (list):
            The list of extracted matrices.
        black_list (list, optional):
            The list of blacklisted paths. Defaults to None.
        path (str, optional):
            The path to the current layer. Defaults to "".
        verbose (Verbose, optional):
            The verbosity level. Default to Verbose.INFO.
    """

    for layer_name in module_tree._modules.keys():
        child = module_tree._modules[layer_name]
        if issubclass(type(child), AnalysisLayerWrapper):
            if verbose > Verbose.INFO:
                print(f"Checking {layer_name} in {path}")

            if black_list is not None:
                black_listed = len([
                    black_listed_string
                    for black_listed_string in black_list
                    if black_listed_string in path + "_" + layer_name
                ]) > 0
            else:
                black_listed = False

            targets_in_path = [
                layer_path_
                for layer_path_ in paths_of_targets
                if layer_path_ in path + "_" + layer_name and not black_listed
            ]
            if len(targets_in_path) > 0:
                layer_path = str(max(targets_in_path, key=len))
                if verbose > Verbose.SILENT:
                    print(f"Found {layer_path} in {path}")

                list_containing_layer_number = [
                    sub_path for sub_path in path.split("_") if sub_path.isdigit()
                ]
                block_index = list_containing_layer_number[0] if len(list_containing_layer_number) > 0 else "-1"
                extracted_matrices.append(
                    AnalysisTensorWrapper(
                        tensor=child.weight.detach(),
                        name=layer_name,
                        label=layer_path,
                        path=path,
                        block_index=int(block_index),
                        layer=child
                    )
                )
        else:
            # Recursively calling the function
            extract_analysis_layer_wrappers(
                module_tree=child,
                paths_of_targets=paths_of_targets,
                extracted_matrices=extracted_matrices,
                black_list=black_list,
                path=path + "_" + layer_name,
                verbose=verbose,
                **kwargs
            )
