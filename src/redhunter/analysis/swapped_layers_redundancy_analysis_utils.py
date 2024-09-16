from __future__ import annotations

import copy

import torch

import transformers

from redhunter.utils.list_utils.list_utils import is_subsequence


class LayerSwitchingWrapperModel:
    """
    Class to switch layers in a model.

    Args:
        model ([transformers.PreTrainedModel | transformers.AutoModel]):
            The model.
        destination_layer_path_source_layer_path_mapping (dict[list | tuple: list | tuple]):
            The layers to switch.

    Attributes:
        model ([transformers.PreTrainedModel | transformers.AutoModel]):
            The model.
        destination_layer_path_source_layer_path_mapping (dict[list | tuple: list | tuple]):
            The layers to switch.
    """

    def __init__(
            self,
            model: [transformers.PreTrainedModel | transformers.AutoModel],
            destination_layer_path_source_layer_path_mapping: dict[list | tuple: list | tuple] = None,
    ) -> None:

        self.model = model
        self.destination_layer_path_source_layer_path_mapping = destination_layer_path_source_layer_path_mapping
        self.overwritten_layers = {}

        if self.destination_layer_path_source_layer_path_mapping is not None:
            self.switch_layers()

    def get_model(
            self
    ) -> [transformers.PreTrainedModel | transformers.AutoModel]:
        """
        Returns the model.

        Returns:
            [transformers.PreTrainedModel | transformers.AutoModel]:
                The model.
        """

        return self.model

    def get_destination_layer_path_source_layer_path_mapping(
            self
    ) -> dict[list | tuple: list | tuple]:
        """
        Returns the layers to switch.

        Returns:
            dict[list | tuple: list | tuple]:
                The layers to switch.
        """

        return self.destination_layer_path_source_layer_path_mapping

    def set_model(
            self,
            model: [transformers.PreTrainedModel | transformers.AutoModel]
    ) -> None:
        """
        Sets the model.

        Args:
            model ([transformers.PreTrainedModel | transformers.AutoModel]):
                The model.
        """

        self.model = model
        if self.destination_layer_path_source_layer_path_mapping is not None:
            self.switch_layers()

    def set_destination_layer_path_source_layer_path_mapping(
            self,
            destination_layer_path_source_layer_path_mapping: dict[list | tuple: list | tuple]
    ) -> None:
        """
        Sets the layers to switch.

        Args:
            destination_layer_path_source_layer_path_mapping (dict[list | tuple: list | tuple]):
                The layers to switch.
        """

        self.destination_layer_path_source_layer_path_mapping = destination_layer_path_source_layer_path_mapping
        if self.destination_layer_path_source_layer_path_mapping is not None:
            self.switch_layers()

    def switch_layers(
            self
    ) -> None:
        """
        Switches the layers of the model.
        """

        source_paths = set(self.get_destination_layer_path_source_layer_path_mapping().values())
        source_layer_path_source_layer_mapping = {source_path: None for source_path in source_paths}
        self._extract_source_layers(self.get_model(), source_layer_path_source_layer_mapping)
        if any([source_layer_path_source_layer_mapping[source_path] is None
                for source_path in source_layer_path_source_layer_mapping.keys()]):
            raise Exception("Some layers could not be extracted.")
        destination_layer_path_source_layer_mapping = {
            destination_path: source_layer_path_source_layer_mapping[source_path]
            for destination_path, source_path in self.get_destination_layer_path_source_layer_path_mapping().items()
        }
        self._fill_destination_layers(self.model, destination_layer_path_source_layer_mapping)

    def _extract_source_layers(
            self,
            module_tree: [transformers.PreTrainedModel | transformers.AutoModel | torch.Module],
            source_layer_path_source_layer_mapping: dict[list | tuple: list | tuple],
            path: list = None
    ) -> None:
        """
        Extracts the source layers from the model.

        Args:
            module_tree ([transformers.PreTrainedModel | transformers.AutoModel | torch.Module]):
                The module tree.
            source_layer_path_source_layer_mapping (dict[str: str]):
                The destination layer path source layer mapping.
            path (list, optional):
                The path. Defaults to None.
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
                source_layer_path_source_layer_mapping[source_path] = child
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

    def _fill_destination_layers(
            self,
            module_tree: [transformers.PreTrainedModel | transformers.AutoModel | torch.Module],
            destination_layer_path_source_layer_mapping: dict[list | tuple: torch.Module],
            path: list = None
    ) -> None:
        """
        Fills the destination layers with the source layers.

        Args:
            module_tree ([transformers.PreTrainedModel | transformers.AutoModel | torch.Module]):
                The module tree.
            destination_layer_path_source_layer_mapping (dict[str: torch.Module]):
                The destination layer path source layer mapping.
            path (list, optional):
                The path. Defaults to None.
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

    def reset_switch(
            self
    ) -> None:
        """
        Resets the switch.
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