from __future__ import annotations

import copy
from abc import abstractmethod, ABC
from typing import Any, override

import torch

import transformers

from exporch.utils.parameters_count import count_parameters
from redhunter.utils.list_utils.list_utils import is_subsequence


class LayerReplacingModelWrapper:
    """
    Class to replace layers in a model with other layers of the same model.

    Args:
        model ([transformers.PreTrainedModel | transformers.AutoModel]):
            The model to be wrapped.
        destination_layer_path_source_layer_path_mapping (dict[list | tuple: list | tuple | Any]):
            The mapping between the path to the layers to be replaced and the path to the layers to be used to replace
            them.

    Attributes:
        model ([transformers.PreTrainedModel | transformers.AutoModel]):
            The wrapped model.
        destination_layer_path_source_layer_path_mapping (dict[list | tuple: list | tuple | Any]):
            The mapping between the path to the layers to be replaced and the path to the layers to be used to replace
            them.
            The structure of the dictionary is as follows:
            {
                [destination_layer_path_1]: [source_layer_path_1],
                [destination_layer_path_2]: [source_layer_path_2],
                ...

            }
    """

    def __init__(
            self,
            model: [transformers.PreTrainedModel | transformers.AutoModel],
            destination_layer_path_source_layer_path_mapping: dict[list | tuple: list | tuple | Any] = None,
    ) -> None:
        self.model = model
        self.destination_layer_path_source_layer_path_mapping = destination_layer_path_source_layer_path_mapping
        self.overwritten_layers = {}

        self.info = {
            "original_model_parameters": count_parameters(self.model),
            "original_model_trainable_parameters": count_parameters(self.model, only_trainable=True)
        }

        if self.destination_layer_path_source_layer_path_mapping is not None:
            self.replace_layers()

        print("Model converted")

    def get_model(
            self
    ) -> [transformers.PreTrainedModel | transformers.AutoModel]:
        """
        Returns the model.

        Returns:
            [transformers.PreTrainedModel | transformers.AutoModel]:
                The wrapped model.
        """

        return self.model

    def get_destination_layer_path_source_layer_path_mapping(
            self
    ) -> dict[list | tuple: list | tuple | Any]:
        """
        Returns the mapping between the layers to be replaced and the layers to be used to replace them.

        Returns:
            dict[list | tuple: list | tuple | Any]:
                The mapping between the path to the layers to be replaced and the path to the layers to be used to replace
                them.
        """

        return self.destination_layer_path_source_layer_path_mapping

    def set_model(
            self,
            model: [transformers.PreTrainedModel | transformers.AutoModel]
    ) -> None:
        """
        Sets the model to be wrapped.

        Args:
            model ([transformers.PreTrainedModel | transformers.AutoModel]):
                The model to be wrapped.
        """

        self.model = model
        if self.destination_layer_path_source_layer_path_mapping is not None:
            self.replace_layers()

    def set_destination_layer_path_source_layer_path_mapping(
            self,
            destination_layer_path_source_layer_path_mapping: dict[list | tuple: list | tuple | Any]
    ) -> None:
        """
        Sets the mapping between the layers to be replaced and the layers to be used to replace them.

        Args:
            destination_layer_path_source_layer_path_mapping (dict[list | tuple: list | tuple | Any]):
                The mapping between the path to the layers to be replaced and the path to the layers to be used to replace
                them.
        """

        self.destination_layer_path_source_layer_path_mapping = destination_layer_path_source_layer_path_mapping
        if self.destination_layer_path_source_layer_path_mapping is not None:
            self.replace_layers()

    def replace_layers(
            self
    ) -> None:
        """
        Replaces the layers in the model based on the mapping.
        """

        source_paths = set(self.get_destination_layer_path_source_layer_path_mapping().values())
        if len(source_paths) != 0 and not all([isinstance(source_path, type(list(source_paths)[0])) for source_path in source_paths]):
            raise Exception("The source paths must be of the same type.")
        if len(source_paths) != 0 and (isinstance(list(source_paths)[0], list) or isinstance(list(source_paths)[0], tuple)):
            source_layer_path_source_layer_mapping = {source_path: None for source_path in source_paths}
            self._extract_source_layers(self.get_model(), source_layer_path_source_layer_mapping)
            if any([source_layer_path_source_layer_mapping[source_path] is None
                    for source_path in source_layer_path_source_layer_mapping.keys()]):
                raise Exception(f"Some layers could not be extracted:\n"
                                f"{'\n - '.join([str(source_path) for source_path in source_layer_path_source_layer_mapping.keys() if source_layer_path_source_layer_mapping[source_path] is None])}")
            source_layer_path_source_layer_mapping = self._preprocess_source_layers(source_layer_path_source_layer_mapping)
            destination_layer_path_source_layer_mapping = {
                destination_path: source_layer_path_source_layer_mapping[source_path]
                for destination_path, source_path in self.get_destination_layer_path_source_layer_path_mapping().items()
            }
        else:
            destination_layer_path_source_layer_mapping = self.get_destination_layer_path_source_layer_path_mapping()
        self._fill_destination_layers(self.model, destination_layer_path_source_layer_mapping)

        model_parameters = count_parameters(self.model)
        self.info.update(
            {
                "model_parameters": model_parameters,
                "model_trainable_parameters": count_parameters(self.model, only_trainable=True),
                "percentage_parameters": model_parameters / self.info["original_model_parameters"] * 100
            }
        )

        print(f"Number of parameters original model: {self.info['original_model_parameters']}")
        print(f"Number of parameters global model: {self.info['model_parameters']}")
        print(f"Percentage of parameters: {self.info['percentage_parameters']}%")
        print()

    def _extract_source_layers(
            self,
            module_tree: [transformers.PreTrainedModel | transformers.AutoModel | torch.nn.Module],
            source_layer_path_source_layer_mapping: dict[list | tuple: list | tuple],
            path: list = None
    ) -> None:
        """
        Extracts the source layers from the model.

        Args:
            module_tree ([transformers.PreTrainedModel | transformers.AutoModel | torch.nn.Module]):
                The module tree.
            source_layer_path_source_layer_mapping (dict[str: str]):
                The mapping between the path to the layers to be used to replace other layers and their actual weights.
            path (list, optional):
                The current path. Defaults to None.
        """

        for layer_name in module_tree._modules.keys():
            # Extracting the child from the current module
            child = module_tree._modules[layer_name]
            source_paths = list(source_layer_path_source_layer_mapping.keys())
            source_paths_in_current_path = [is_subsequence(source_path, path + [f"{layer_name}"] if path is not None else [f"{layer_name}"]) for source_path in source_paths]
            if sum(source_paths_in_current_path) > 1:
                raise Exception("Multiple layers have the same path.")
            if any(source_paths_in_current_path):
                # Storing the child in the destination layer path source layer mapping
                source_path = source_paths[source_paths_in_current_path.index(True)]
                source_layer_path_source_layer_mapping[source_path] = copy.deepcopy(child)
            elif len(child._modules) == 0:
                # If the child has no children, we reached a leaf node and we do nothing
                pass
            else:
                # Recursively calling the method on the child, if the child has children
                new_path = copy.copy(path) + [f"{layer_name}"] if path != None else [f"{layer_name}"]
                self._extract_source_layers(
                    child,
                    source_layer_path_source_layer_mapping,
                    new_path
                )

    def _preprocess_source_layers(
            self,
            source_layer_path_source_layer_mapping: dict[list | tuple: torch.nn.Module]
    ) -> dict[list | tuple: torch.nn.Module]:
        """
        Pre-processes the source layers.

        Args:
            source_layer_path_source_layer_mapping (dict[str: torch.nn.Module]):
                The mapping between the path to the layers to be used to replace other layers and their actual weights.

        Returns:
            dict[str: torch.nn.Module]:
                The pre-processed source layers.
        """

        return source_layer_path_source_layer_mapping

    def _fill_destination_layers(
            self,
            module_tree: [transformers.PreTrainedModel | transformers.AutoModel | torch.nn.Module],
            destination_layer_path_source_layer_mapping: dict[list | tuple: torch.nn.Module],
            path: list = None
    ) -> None:
        """
        Replaces the destination layers path with the actual source layers.

        Args:
            module_tree ([transformers.PreTrainedModel | transformers.AutoModel | torch.nn.Module]):
                The module tree.
            destination_layer_path_source_layer_mapping (dict[str: torch.nn.Module]):
                The mapping between the path to the layers to be replaced and the layers that should be used to replace
                them.
            path (list, optional):
                The current path. Defaults to None.
        """

        for layer_name in module_tree._modules.keys():
            # Extracting the child from the current module
            child = module_tree._modules[layer_name]
            destination_paths = list(destination_layer_path_source_layer_mapping.keys())
            destination_paths_in_current_path = [is_subsequence(destination_path, path + [f"{layer_name}"] if path is not None else [f"{layer_name}"]) for destination_path in destination_paths]
            if sum(destination_paths_in_current_path) > 1:
                raise Exception("Multiple layers have the same path.")
            if any(destination_paths_in_current_path):
                # Setting the new layer from the source path in the destination layer
                destination_path = destination_paths[destination_paths_in_current_path.index(True)]
                ########################################################################################################
                # Storing the overwritten layer in order be able to reset the switch.
                # For future changes: the very same method is used to reset the switch, passing a COPY of the dictionary
                # overwritten_layers, using the very same instance the reset does not work.
                self.overwritten_layers[destination_path] = module_tree._modules[layer_name]
                ########################################################################################################
                module_tree._modules[layer_name] = destination_layer_path_source_layer_mapping[destination_path]
            elif len(child._modules) == 0:
                # If the child has no children, we reached a leaf node and we do nothing
                pass
            else:
                # Recursively calling the method on the child, if the child has children
                new_path = copy.copy(path) + [f"{layer_name}"] if path != None else [f"{layer_name}"]
                self._fill_destination_layers(
                    child,
                    destination_layer_path_source_layer_mapping,
                    new_path
                )

    def reset_replacement(
            self
    ) -> None:
        """
        Resets the replacement.
        """

        if len(self.overwritten_layers) == 0:
            raise Exception("The layers have not been switched.")

        self._fill_destination_layers(self.model, copy.copy(self.overwritten_layers))
        self.overwritten_layers = {}

    def __str__(self):
        """
        Returns the string representation of the object.
        """

        return self.model.__str__()


class ProcessedLayerReplacingModelWrapper(LayerReplacingModelWrapper, ABC):
    """
    Class to replace layers in a model with a layer that is the processing of the extracted layers.

    Args:
        model ([transformers.PreTrainedModel | transformers.AutoModel]):
            The model to be wrapped.
        destination_layer_path_source_layer_path_mapping (dict[list | tuple: list | tuple]):
            The mapping between the path to the layers to be replaced and the path to the layers to be used to replace
            them. The source_layer_path will be ignored, if given.

    Attributes:
        model ([transformers.PreTrainedModel | transformers.AutoModel]):
            The wrapped model.
        destination_layer_path_source_layer_path_mapping (dict[list | tuple: list | tuple]):
            The mapping between the path to the layers to be replaced and the path to the layers to be used to replace
            them.
            The structure of the dictionary is as follows:
            {
                [destination_layer_path_1]: [source_layer_path_1],
                [destination_layer_path_2]: [source_layer_path_2],
                ...

            }
            The source_layer_path will be ignored, if given.
    """

    def __init__(
            self,
            model: [transformers.PreTrainedModel | transformers.AutoModel],
            destination_layer_path_source_layer_path_mapping: dict[list | tuple: list | tuple] = None,
    ) -> None:
        super().__init__(
            model,
            None if destination_layer_path_source_layer_path_mapping is None
            else {key: key for key in destination_layer_path_source_layer_path_mapping.keys()}
        )

    @override
    def set_destination_layer_path_source_layer_path_mapping(
            self,
            destination_layer_path_source_layer_path_mapping: dict[list | tuple: list | tuple]
    ) -> None:
        """
        Sets the mapping between the layers to be replaced and the layers to be used to replace them.

        Args:
            destination_layer_path_source_layer_path_mapping (dict[list | tuple: list | tuple]):
                The mapping between the path to the layers to be replaced and the path to the layers to be used to replace
                them.
        """

        self.destination_layer_path_source_layer_path_mapping = (
            None if destination_layer_path_source_layer_path_mapping is None
            else {key: key for key in destination_layer_path_source_layer_path_mapping.keys()}
        )
        if self.destination_layer_path_source_layer_path_mapping is not None:
            self.replace_layers()

    @override
    def _preprocess_source_layers(
            self,
            source_layer_path_source_layer_mapping: dict[list | tuple: torch.nn.Module]
    ) -> dict[list | tuple: torch.nn.Module]:
        """
        Pre-processes the source layers.

        Args:
            source_layer_path_source_layer_mapping (dict[str: torch.nn.Module]):
                The mapping between the path to the layers to be used to replace other layers and their actual weights.

        Returns:
            dict[str: torch.nn.Module]:
                The pre-processed source layers.
        """

        return self.preprocess_source_layers(source_layer_path_source_layer_mapping)


    @abstractmethod
    def preprocess_source_layers(
            self,
            source_layer_path_source_layer_mapping: dict[list | tuple: torch.nn.Module]
    ) -> dict[list | tuple: torch.nn.Module]:
        """
        Pre-processes the source layers.

        Args:
            source_layer_path_source_layer_mapping (dict[str: torch.nn.Module]):
                The mapping between the path to the layers to be used to replace other layers and their actual weights.

        Returns:
            dict[str: torch.nn.Module]:
                The pre-processed source layers.
        """


class NullLayerReplacingModelWrapper(ProcessedLayerReplacingModelWrapper):
    """
    Class to replace layers in a model with a layer that outputs zeros.

    Args:
        model ([transformers.PreTrainedModel | transformers.AutoModel]):
            The model to be wrapped.
        destination_layer_path_source_layer_path_mapping (dict[list | tuple: list | tuple]):
            The mapping between the path to the layers to be replaced and the path to the layers to be used to replace
            them. The source_layer_path will be ignored, if given.

    Attributes:
        model ([transformers.PreTrainedModel | transformers.AutoModel]):
            The wrapped model.
        destination_layer_path_source_layer_path_mapping (dict[list | tuple: list | tuple]):
            The mapping between the path to the layers to be replaced and the path to the layers to be used to replace
            them.
            The structure of the dictionary is as follows:
            {
                [destination_layer_path_1]: [source_layer_path_1],
                [destination_layer_path_2]: [source_layer_path_2],
                ...

            }
            The source_layer_path will be ignored, if given.
    """

    def __init__(
            self,
            model: [transformers.PreTrainedModel | transformers.AutoModel],
            destination_layer_path_source_layer_path_mapping: dict[list | tuple: list | tuple] = None,
    ) -> None:
        super().__init__(
            model,
            destination_layer_path_source_layer_path_mapping
        )

    @override
    def preprocess_source_layers(
            self,
            source_layer_path_source_layer_mapping: dict[list | tuple: torch.nn.Module]
    ) -> dict[list | tuple: torch.nn.Module]:
        """
        Pre-processes the source layers.

        Args:
            source_layer_path_source_layer_mapping (dict[str: torch.nn.Module]):
                The mapping between the path to the layers to be used to replace other layers and their actual weights.

        Returns:
            dict[str: torch.nn.Module]:
                The pre-processed source layers.
        """

        source_layer_path_null_layer_mapping = {key: value for key, value in source_layer_path_source_layer_mapping.items()}

        for source_layer in source_layer_path_null_layer_mapping.values():
            self._fill_with_zeros(source_layer)

        return source_layer_path_null_layer_mapping

    def _fill_with_zeros(
            self,
            module_tree: [transformers.PreTrainedModel | transformers.AutoModel | torch.nn.Module]
    ) -> dict[list | tuple: torch.nn.Module]:
        """
        Fills the layer with zeros.

        Args:
            module_tree ([transformers.PreTrainedModel | transformers.AutoModel | torch.nn.Module]):
                The module tree.

        Returns:
            dict[str: torch.nn.Module]:
                The pre-processed source layers.
        """
        if len(module_tree._modules) == 0:
            try :
                module_tree.weight.data.fill_(0)
            except AttributeError:
                pass
            try :
                module_tree.bias.data.fill_(0)
            except AttributeError:
                pass
        else:
            for layer_name in module_tree._modules.keys():
                child = module_tree._modules[layer_name]
                self._fill_with_zeros(child)
