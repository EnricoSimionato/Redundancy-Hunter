from __future__ import annotations

import copy
import re
from typing import Any

import numpy as np

import torch

import transformers

from exporch import Verbose

from redhunter.utils.list_utils.list_utils import is_subsequence

from exporch.utils import LoggingInterface

class RankAnalysisResult:
    """
    Class to store the result of the rank analysis. It stores the rank of the tensor and the thresholds used to compute
    the rank.

    Args:
        rank (int):
            The rank of the tensor.
        explained_variance_threshold (float, optional):
            The threshold on the explained variance to use to compute the rank. Rank is computed as the number of
            singular values that explain the threshold fraction of the total variance. Defaults to 0.
        singular_values_threshold (float, optional):
            The threshold to use to compute the rank based on singular values. Rank is computed as the number of
            singular values that are greater than the threshold. Defaults to 0.
        verbose (Verbose, optional):
            The verbosity level. Defaults to Verbose.INFO.

    Attributes:
        rank (int):
            The rank of the tensor.
        explained_variance_threshold (float):
            The threshold on the explained variance to use to compute the rank. Rank is computed as the number of
            singular values that explain the threshold fraction of the total variance.
        singular_values_threshold (float):
            The threshold to use to compute the rank based on singular values. Rank is computed as the number of
            singular values that are greater than the threshold.
        verbose (Verbose):
            The verbosity level.
    """

    def __init__(
            self,
            rank: int,
            explained_variance_threshold: float = 0,
            singular_values_threshold: float = 0,
            verbose: Verbose = Verbose.INFO
    ) -> None:
        self.rank = rank
        self.explained_variance_threshold = explained_variance_threshold
        self.singular_values_threshold = singular_values_threshold

        self.verbose = verbose

    def get_rank(
            self
    ) -> int:
        """
        Returns the rank of the tensor.

        Returns:
            int:
                The rank of the tensor.
        """

        return self.rank

    def get_explained_variance_threshold(
            self
    ) -> float:
        """
        Returns the threshold on the explained variance to use to compute the rank.

        Returns:
            float:
                The threshold on the explained variance to use to compute the rank.
        """

        return self.explained_variance_threshold

    def get_singular_values_threshold(
            self
    ) -> float:
        """
        Returns the threshold to use to compute the rank based on singular values.

        Returns:
            float:
                The threshold to use to compute the rank based on singular values.
        """

        return self.singular_values_threshold


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
        path (list, optional):
            The path of the tensor. Defaults to None.
        block_index (int, optional):
            The block index of the tensor. Defaults to None.
        layer (torch.nn.Module, optional):
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
        layer (torch.nn.Module):
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
            path: list = None,
            block_index: int = None,
            layer: torch.nn.Module = None,
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
            return self.tensor.detach().to(torch.float32).numpy()
        else:
            return self.tensor.detach().to(torch.float32)

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
            self,
            string: bool = False
    ) -> list | str:
        """
        Returns the path of the tensor.

        Returns:
            list | str:
                The path of the tensor.
        """

        if string:
            return ".".join(self.path)
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
    ) -> torch.nn.Module:
        """
        Returns the layer of the tensor.

        Returns:
            torch.nn.Module:
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

        return self.compute_explained_variance(self.singular_values)

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
            path: list
    ) -> None:
        """
        Sets the path of the tensor.

        Args:
            path (list):
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
            layer: torch.nn.Module
    ) -> None:
        """
        Sets the layer of the tensor.

        Args:
            layer (torch.nn.Module):
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
    ) -> np.ndarray:
        """
        Computes the singular values of the tensor.

        Returns:
            np.ndarray:
                The singular values of the tensor.
        """

        singular_values = np.linalg.svd(self.get_tensor(numpy_array=True), compute_uv=False)
        self.set_singular_values(singular_values)

        return singular_values

    def compute_explained_variance(
            self,
            scaling: int = 1
    ) -> np.array:
        """
        Computes the explained variance for a set of singular values.

        Args:
            scaling (float, optional):
                Scaling to apply to the explained variance at each singular value. Default to 1.

        Returns:
            np.array:
                The explained variance for each singular value.
        """

        s = self.get_singular_values()
        if s is None:
            raise ValueError("The singular values must be computed before computing the explained variance.")

        if s[0] == 0.:
            return np.ones(len(s))

        return (np.square(s) * scaling).cumsum() / (np.square(s) * scaling).sum()

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

        explained_variance = self.compute_explained_variance(self.singular_values)

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

# TODO implement a method to group the things
class AnalysisTensorDict(LoggingInterface):
    """
    Dictionary of tensors for the analysis.

    Args:
        keys ([list[tuple[Any, ...]] | list[Any]], optional):
            The keys of the tensors. Defaults to ().
        tensors ([list[list[AnalysisTensorWrapper]] | list[AnalysisTensorWrapper]], optional):
            The tensors to add to the dictionary. Defaults to ().
        layer_paths_configurations_to_analyze (list[list[str]]):
            List of configurations that contain the pointers to the tensors that will be analysed together. Defaults to
            ().
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
        layer_paths_configurations_to_analyze (list[list[str]]):
            List of configurations that contain the pointers to the tensors that will be analysed together.
        verbose (Verbose):
            The verbosity level.
    """

    def __init__(
            self,
            keys: [list[tuple[Any, ...]] | list[Any]] = (),
            tensors: [list[list[AnalysisTensorWrapper]] | list[AnalysisTensorWrapper]] = (),
            layer_paths_configurations_to_analyze: list[list[str]] = (),
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
        self.layer_paths_configurations_to_analyze = layer_paths_configurations_to_analyze

        self.verbose = verbose

    def group_by_block_index(
            self
    ) -> 'AnalysisTensorDict':

        grouped_tensors = AnalysisTensorDict()
        for key in self.tensors.keys():
            for tensor in self.tensors[key]:
                block_index = tensor.get_block_index()
                if block_index is not None:
                    grouped_tensors.append_tensor(
                        block_index,
                        tensor
                    )
        return grouped_tensors

    def group_by_name(
            self
    ) -> 'AnalysisTensorDict':

        grouped_tensors = AnalysisTensorDict()
        for key in self.tensors.keys():
            for tensor in self.tensors[key]:
                name = tensor.get_name()
                if name is not None:
                    grouped_tensors.append_tensor(
                        name,
                        tensor
                    )
        return grouped_tensors

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

        if isinstance(key, list):
            key = tuple(key)

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

        if isinstance(key, list):
            key = tuple(key)

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

    def get_layer_paths_configurations_to_analyze(
            self
    ) -> list[Any]:
        """
        Returns the keys configuration to analyze.
        """

        return  copy.deepcopy(self.layer_paths_configurations_to_analyze)

    def remove_layer_paths_configuration_to_analyze(
            self,
            layer_paths_configurations: list[str] | list[list[str]]
    ) -> None:
        """
        Removes one or more configurations from the list of configurations.

        Args:
            layer_paths_configurations (list[str] | list[list[str]]):
                The configuration or configurations to remove.
        """

        if len(layer_paths_configurations) > 0:
            if isinstance(layer_paths_configurations[0], list):
                for configuration in layer_paths_configurations:
                    if configuration in self.layer_paths_configurations_to_analyze:
                        self.layer_paths_configurations_to_analyze.remove(configuration)
            elif isinstance(layer_paths_configurations[0], str):
                if layer_paths_configurations in self.layer_paths_configurations_to_analyze:
                    self.layer_paths_configurations_to_analyze.remove(layer_paths_configurations)
            else:
                raise ValueError("A configuration must be list of strings.\n"
                                 "A list of strings or a list of list of strings is expected.")

    def get_wrappers_for_analysis(
            self,
            target_paths_configuration_to_analyze: list[list[list[str]]]
    ) -> list[list[AnalysisTensorWrapper]]:
        """
        Returns the wrappers for the analysis.

        Args:
            target_paths_configuration_to_analyze (list[list[list[str]]]):
                The key configuration to analyze.

        Returns:
            list[list[AnalysisTensorWrapper]]:
                The wrappers for the analysis.
        """

        structured_wrappers_to_analyze = []
        for target_paths_configuration_section in target_paths_configuration_to_analyze:
            wrappers_configuration_section = []
            for target_layer_path in target_paths_configuration_section:
                layer_key = [key for key in self.get_keys() if is_subsequence(target_layer_path, key)]
                if len(layer_key) == 0:
                    raise ValueError(f"No tensor found for the target layer path {target_layer_path}.")
                if len(layer_key) > 1:
                    raise ValueError(f"More than one tensor found for the target layer path {target_layer_path}.")
                layer_wrappers = self.get_tensor_list(layer_key[0])
                if len(layer_wrappers) > 0:
                    for layer_wrapper in layer_wrappers:
                        wrappers_configuration_section.append(layer_wrapper)

            structured_wrappers_to_analyze.append(wrappers_configuration_section)

        return structured_wrappers_to_analyze

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

        if isinstance(key, list):
            key = tuple(key)

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

        if isinstance(key, list):
            key = tuple(key)

        if not isinstance(key, tuple):
            key = (key,)

        if key in self.tensors.keys():
            if isinstance(tensor, list):
                self.tensors[key] = self.tensors[key] + tensor
            else:
                self.tensors[key] = self.tensors[key] + [tensor]
        else:
            self.set_tensor(key, tensor)

    def set_layer_paths_configurations_to_analyze(
            self,
            layer_paths_configurations_to_analyze: list[Any]
    ) -> None:
        """
        Sets the keys configuration to analyze.

        Args:
            layer_paths_configurations_to_analyze (list[Any]):
                The keys configuration to analyze.
        """

        self.layer_paths_configurations_to_analyze = layer_paths_configurations_to_analyze

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

    def compute_singular_values(
            self
    ) -> dict:
        """
        Computes the singular values of the tensors.

        Returns:
            dict:
                The singular values of the tensors and the explained variance.
        """

        results = {}
        for key in self.tensors.keys():
            results[key] = {}
            self.log(f"Computing singular values for tensors with key {key}.")
            for tensor in self.tensors[key]:
                singular_values = tensor.compute_singular_values()
                explained_variance = tensor.get_explained_variance()
                a = tensor.get_path(string=True)
                results[key][tensor.get_path(string=True)] = {
                    "singular_values": singular_values,
                    "explained_variance": explained_variance,
                    "shape": tensor.get_shape()
                }

        return results


class AnalysisLayerWrapper(torch.nn.Module):
    """
    Wrapper for the analysis of a layer.

    Args:
        layer (torch.nn.Module):
            The layer.
        label (str, optional):
            The label of the layer. Defaults to None.

    Attributes:
        layer (torch.nn.Module):
            The layer.
        label (str):
            The label of the layer.
        activations (list):
            The list of activations.
    """

    def __init__(
            self,
            layer: torch.nn.Module,
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
    ) -> torch.nn.Module:
        """
        Returns the layer.

        Returns:
            torch.nn.Module:
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


class AnalysisModelWrapper(torch.nn.Module):
    """
    Wrapper for the analysis of a model.

    Args:
        model (torch.nn.Module):
            The model.
        *args:
            Additional positional arguments.
        **kwargs:
            Additional keyword arguments.

    Attributes:
        model (torch.nn.Module):
            The model.
    """

    def __init__(
            self,
            model: [torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel],
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
            module_tree: torch.nn.Module,
            paths_of_targets: list,
            black_list: list = None,
            path: str = "",
            verbose: Verbose = Verbose.SILENT,
            **kwargs
    ) -> None:
        """
        Converts layers into global-dependent versions.

        Args:
            module_tree (torch.nn.Module):
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
            module_tree (torch.nn.Module):
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
            module_tree: torch.nn.Module,
            store_activations: bool
    ) -> None:
        """
        Sets whether to store the activations.

        Args:
            module_tree (torch.nn.Module):
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
            module_tree: torch.nn.Module
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
            module_tree: torch.nn.Module
    ) -> None:
        """
        Resets the activations.

        Args:
            module_tree (torch.nn.Module):
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


# Definition of the functions to compute the rank of a matrix

def compute_explained_variance(
        self,
        scaling: int = 1
) -> np.array:
    """
    Computes the explained variance for a set of singular values.

    Args:
        scaling (float, optional):
            Scaling to apply to the explained variance at each singular value. Default to 1.

    Returns:
        np.array:
            The explained variance for each singular value.
    """

    s = self.get_singular_values()
    if s is None:
        raise ValueError("The singular values must be computed before computing the explained variance.")

    if s[0] == 0.:
        return np.ones(len(s))

    return (np.square(s) * scaling).cumsum() / (np.square(s) * scaling).sum()

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
        module_tree: [torch.nn.Module | transformers.AutoModel],
        target_paths: list,
        layers_storage: AnalysisTensorDict,
        blacklist: list = (),
        path: list = None,
        verbose: Verbose = Verbose.INFO,
        **kwargs
) -> None:
    """
    Extracts the matrices from the model tree.

    Args:
        module_tree ([torch.nn.Module | transformers.AutoModel]):
            The model tree.
        target_paths (list):
            The path of the targets.
        layers_storage (AnalysisTensorDict):
            Storage where the extracted layers will be at the end of the extraction.
        blacklist (list, optional):
            The list of blacklisted paths. Defaults to ().
        path (list, optional):
            The path to the current layer. Defaults to None.
        verbose (Verbose, optional):
            The verbosity level. Defaults to Verbose.INFO.
    """

    for layer_name in module_tree._modules.keys():
        # Extracting the child from the current module
        child = module_tree._modules[layer_name]
        layer_path = copy.deepcopy(path) + [f"{layer_name}"] if path is not None else [f"{layer_name}"]

        if len(child._modules) == 0:
            verbose.print(f"Checking {layer_name} in {path}", Verbose.INFO)
            target_paths_in_current_path = [
                is_subsequence(
                    [sub_path for sub_path in target_path if sub_path != "block_index"],
                    layer_path
                ) and not any(blacklisted_string in layer_path for blacklisted_string in blacklist)
                for target_path in target_paths]
            if sum(target_paths_in_current_path) > 1:
                raise Exception(f"The layer {layer_path} corresponds to multiple targets.")
            if any(target_paths_in_current_path):
                verbose.print(f"Found {layer_name} in {layer_path}", Verbose.INFO)

                # Computing the index block of the layer
                list_containing_layer_number = [el for el in layer_path if el.isdigit()]
                if len(list_containing_layer_number) > 1:
                    raise Exception(f"Multiple candidates as block index found. The path is ambiguous")
                block_index = list_containing_layer_number[0] if len(list_containing_layer_number) > 0 else "-1"

                # Storing the layer in the dictionary of extracted layers
                layers_storage.append_tensor(
                    layer_path,
                    AnalysisTensorWrapper(
                        tensor=child.weight.detach(),
                        name=layer_name,
                        label="_".join(layer_path),
                        path=layer_path,
                        block_index=int(block_index),
                        layer=child
                    )
                )
        else:
            # Recursively calling the function
            extract(
                module_tree=child,
                target_paths=target_paths,
                layers_storage=layers_storage,
                blacklist=blacklist,
                path=layer_path,
                verbose=verbose,
                **kwargs
            )

def extract_analysis_layer_wrappers(
        module_tree: [torch.nn.Module | transformers.AutoModel],
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
        module_tree ([torch.nn.Module | transformers.AutoModel]):
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
