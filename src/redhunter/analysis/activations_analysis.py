from __future__ import annotations

import os
import logging
import pickle as pkl
from tqdm import tqdm

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

import torch

from exporch import Config, Verbose
from exporch.utils.plot_utils import plot_heatmap

from exporch.utils.classification import IMDBDataModule
from exporch.utils.classification import load_model_for_sequence_classification, load_tokenizer_for_sequence_classification

from redhunter.analysis.sorted_layers_rank_analysis import compute_cosine
from redhunter.analysis.analysis_utils import AnalysisModelWrapper


def perform_activations_analysis(
        config: Config,
) -> None:
    """
    Performs the activations' analysis.

    Args:
        config (Config):
            The configuration object containing the necessary information to perform the analysis.
    """

    logging.basicConfig(filename=os.path.join(config.get("directory_path"), "logs.log"), level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Running perform_sorted_layers_rank_analysis in sorted_layers_rank_analysis.py.")

    # Getting the parameters related to the paths from the configuration
    logger.info(f"Getting the parameters related to the paths from the configuration")
    file_available, file_path, directory_path, file_name, file_name_no_format = [
        config.get(name)
        for name in ["file_available", "file_path", "directory_path", "file_name", "file_name_no_format"]
    ]
    logger.info(f"Information retrieved")

    # Getting the parameters related to the analysis from the configuration
    verbose = config.get_verbose()
    fig_size = config.get("figure_size") if config.contains("figure_size") else (100, 20)
    heatmap_size = config.get("heatmap_size") if config.contains("heatmap_size") else (40, 40)
    num_iterations = config.get("num_iterations") if config.contains("num_iterations") else 1
    batch_size = config.get("batch_size") if config.contains("batch_size") else 64
    num_workers = config.get("num_workers") if config.contains("num_workers") else 1
    seed = config.get("seed") if config.contains("seed") else 42
    max_len = config.get("max_len") if config.contains("max_len") else 512

    # Load the data if the file is available, otherwise process the model
    if file_available:
        print(f"The file '{file_path}' is available.")
        logger.info(f"The file '{file_path}' is available.")
        with open(file_path, "rb") as f:
            data = pkl.load(f)
        logger.info(f"Data loaded from the file '{file_path}'.")
    else:
        # Loading the model
        model = load_model_for_sequence_classification(config)
        logger.info(f"Model loaded.")
        # Loading the tokenizer
        tokenizer = load_tokenizer_for_sequence_classification(config)
        logger.info(f"Tokenizer loaded.")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Wrapping the model
        model_wrapper = AnalysisModelWrapper(model, config.get("targets"), config.get("black_list") if config.contains("black_list") else None)
        logger.info(f"Model wrapped.")
        print(model_wrapper)

        if config.get("dataset_id") == "stanfordnlp/imdb":
            # Loading the dataset
            dataset = IMDBDataModule(
                tokenizer=tokenizer,
                max_len=max_len,
                batch_size=batch_size,
                num_workers=num_workers,
                split=(0.8, 0.1, 0.1),
                seed=seed
            )
            dataset.setup()
        else:
            raise Exception("The dataset is not recognized.")
        logger.info(f"Dataset loaded.")

        # Performing the activation analysis
        data_loader = dataset.train_dataloader()
        verbose.print("Staring to feed the inputs to the model.", Verbose.SILENT)
        for idx, batch in tqdm(enumerate(data_loader)):
            logger.info(f"Batch {idx + 1} out of {num_iterations}.")
            if idx == num_iterations:
                break

            #Preparing the inputs
            inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": batch.get("labels")
            }

            # Forward pass through the model
            y = model_wrapper.forward(**inputs)
        logger.info(f"All activations computed.")

        data = model_wrapper

        # Saving the activations
        logger.info(f"Storing the data for future usage.")
        with open(f"{file_path}", "wb") as f:
            pkl.dump(data, f)
        logger.info(f"Data saved to the file '{file_path}'.")

    # Extracting the activations
    model_wrapper = data
    activations = model_wrapper.get_activations()
    if verbose >= Verbose.DEBUG:
        print_nested_dictionary(activations)

    # Flattening the activations
    flattened_activations = {}
    flatten_dictionary(flattened_activations, activations)

    key_value_couples = [(activation_dict_key, activation_dict_value["mean_activations"]) for activation_dict_key, activation_dict_value in flattened_activations.items() if activation_dict_value["mean_activations"] is not None]

    # Filtering the activations
    filtered_key_value_couples = [key_value_couple for key_value_couple in key_value_couples if is_at_least_one_element_in_list(config.get("targets"), key_value_couple[0].split(" -> "))]

    # Printing the statistics of the activations of one layer
    mean_activations = filtered_key_value_couples[0][1]
    print(f"Shape mean activations: {mean_activations.shape}")
    print(f"Max mean activations: {mean_activations.max()}")
    print(f"Min mean activations: {mean_activations.min()}")
    print(f"Max absolute mean activations: {mean_activations.abs().max()}")
    print(f"Min absolute activations: {mean_activations.abs().min()}")
    print(f"Average mean activations: {mean_activations.mean()}")
    print(f"Variance mean activations: {mean_activations.var()}")
    print(f"Average absolute mean activations: {mean_activations.abs().mean()}")
    print(f"Variance absolute mean activations: {mean_activations.abs().var()}")

    # Plotting the mean activations
    fig_1, axis_1 = plt.subplots(1, 1, figsize=fig_size)
    fig_1.suptitle("Mean activations of different layers")
    x = range(len(key_value_couples[0][1]))
    for key_value_couple in filtered_key_value_couples:
        if len(key_value_couple[1]) != len(x):
            raise Exception("The length of the tensors is not the same for all the embeddings.")

    for key_value_couple in filtered_key_value_couples:
        axis_1.plot(key_value_couple[1].detach().numpy(), label=f"{key_value_couple[0]}")

    axis_1.set_title("Mean activations")
    axis_1.set_xlabel("Component index")
    axis_1.set_ylabel("Value of the component")
    axis_1.legend()

    # Saving the plot
    fig_path = os.path.join(directory_path, file_name_no_format + "_mean_activations.png")
    fig_1.savefig(f"{fig_path}")
    logger.info(f"Plot saved to '{fig_path}'.")

    # Computing the similarity matrix
    similarity_matrix = torch.zeros((len(filtered_key_value_couples), len(filtered_key_value_couples)))
    for index_1, key_value_1 in enumerate(filtered_key_value_couples):
        for index_2, key_value_2 in enumerate(filtered_key_value_couples):
            similarity_matrix[index_1, index_2] = compute_cosine(key_value_1[1], key_value_2[1], dim=1)

    fig_2, axis_2 = plt.subplots(1, 1, figsize=heatmap_size)
    fig_2.suptitle("Similarity matrix of the mean activations of different layers")
    heatmap = axis_2.imshow(similarity_matrix, cmap="seismic", interpolation="nearest", vmin=-1, vmax=1)

    axis_2.set_xlabel("Layer Label")
    axis_2.set_xticks(range(len(filtered_key_value_couples)))
    axis_2.set_xticklabels([key_value_couple[0] for key_value_couple in filtered_key_value_couples], rotation=90)

    axis_2.set_ylabel("Layer Label")
    axis_2.set_yticks(range(len(filtered_key_value_couples)))
    axis_2.set_yticklabels([key_value_couple[0] for key_value_couple in filtered_key_value_couples])

    # Adding the colorbar
    divider = make_axes_locatable(axis_2)
    colormap_axis = divider.append_axes("right", size="5%", pad=0.05)
    fig_2.colorbar(
        heatmap,
        cax=colormap_axis
    )
    plt.tight_layout()

    # Saving the plot
    fig_path = os.path.join(directory_path, file_name_no_format + "_similarity_matrix.png")
    fig_2.savefig(f"{fig_path}")
    logger.info(f"Plot saved to '{fig_path}'.")

    logger.info(f"Activations' analysis completed.")


def perform_delta_activations_analysis(
        config: Config,
) -> None:
    """
    Performs the activations' analysis.

    Args:
        config (Config):
            The configuration object containing the necessary information to perform the analysis.
    """

    logger = logging.getLogger(__name__)
    logger.info(f"Running perform_sorted_layers_rank_analysis in sorted_layers_rank_analysis.py.")

    # Getting the parameters related to the paths from the configuration
    logger.info(f"Getting the parameters related to the paths from the configuration")
    file_available, file_path, directory_path, file_name, file_name_no_format = [
        config.get(name)
        for name in ["file_available", "file_path", "directory_path", "file_name", "file_name_no_format"]
    ]
    logger.info(f"Information retrieved")

    # Getting the parameters related to the analysis from the configuration
    verbose = config.get_verbose()
    fig_size = config.get("figure_size") if config.contains("figure_size") else (20, 20)
    num_iterations = config.get("num_iterations") if config.contains("num_iterations") else 1
    batch_size = config.get("batch_size") if config.contains("batch_size") else 64
    num_workers = config.get("num_workers") if config.contains("num_workers") else 1
    seed = config.get("seed") if config.contains("seed") else 42
    max_len = config.get("max_len") if config.contains("max_len") else 512

    # Load the data if the file is available, otherwise process the model
    if file_available:
        print(f"The file '{file_path}' is available.")
        logger.info(f"The file '{file_path}' is available.")
        with open(file_path, "rb") as f:
            data = pkl.load(f)
        logger.info(f"Data loaded from the file '{file_path}'.")
    else:
        # Loading the model
        model = load_model_for_sequence_classification(config)
        logger.info(f"Model loaded.")
        # Loading the tokenizer
        tokenizer = load_tokenizer_for_sequence_classification(config)
        logger.info(f"Tokenizer loaded.")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Wrapping the model
        model_wrapper = AnalysisModelWrapper(model, config.get("targets"),
                                             config.get("black_list") if config.contains("black_list") else None)
        logger.info(f"Model wrapped.")
        print(model_wrapper)

        if config.get("dataset_id") == "stanfordnlp/imdb":
            # Loading the dataset
            dataset = IMDBDataModule(
                tokenizer=tokenizer,
                max_len=max_len,
                batch_size=batch_size,
                num_workers=num_workers,
                split=(0.8, 0.1, 0.1),
                seed=seed
            )
            dataset.setup()
        else:
            raise Exception("The dataset is not recognized.")
        logger.info(f"Dataset loaded.")

        # Performing the activation analysis
        data_loader = dataset.train_dataloader()
        verbose.print("Staring to feed the inputs to the model.", Verbose.SILENT)
        delta_activations = {}
        layer_path_labels = None
        for batch_index in tqdm(range(num_iterations)):
            logger.info(f"Batch {batch_index + 1} out of {num_iterations}.")

            batch = next(iter(data_loader))

            # Preparing the inputs
            inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": batch.get("labels")
            }

            # Forward pass through the model
            _ = model_wrapper.forward(**inputs)

            # Getting the activations
            activations = model_wrapper.get_activations()
            # Flattening the activations
            flattened_activations = {}
            flatten_dictionary(flattened_activations, activations)
            flattened_activations = [flattened_activation for flattened_activation in flattened_activations.values() if flattened_activation["activations"] is not None]
            if layer_path_labels is None:
                layer_path_labels = [flattened_activation["path"] for flattened_activation in flattened_activations]

            num_activations = len(flattened_activations)
            for index_activation_1 in range(num_activations):
                for index_activation_2 in range(num_activations):
                    stacked_activation_1 = torch.stack(flattened_activations[index_activation_1]["activations"])
                    stacked_activation_2 = torch.stack(flattened_activations[index_activation_2]["activations"])

                    delta_activation = stacked_activation_1 - stacked_activation_2
                    previous_mean = delta_activations[(index_activation_1, index_activation_2)]["activations_mean"] if (index_activation_1, index_activation_2) in delta_activations else 0
                    previous_num = delta_activations[(index_activation_1, index_activation_2)]["activations_num"] if (index_activation_1, index_activation_2) in delta_activations else 0

                    current_mean = torch.mean(torch.norm(delta_activation, dim=-1) / torch.sqrt(torch.norm(stacked_activation_1, dim=-1) * torch.norm(stacked_activation_2, dim=-1)))
                    current_num = delta_activation.shape[0] * delta_activation.shape[1] * delta_activation.shape[2]

                    delta_activations[(index_activation_1, index_activation_2)] = {
                        "activations_mean": (previous_num * previous_mean + current_num * current_mean) / (previous_num + current_num),
                        "activations_num": previous_num + current_num
                    }

            model_wrapper.reset_activations()

        logger.info(f"Activations and difference between activations computed")

        data = (model_wrapper, delta_activations, layer_path_labels)

        # Saving the activations
        logger.info(f"Storing the data for future usage.")
        with open(f"{file_path}", "wb") as f:
            pkl.dump(data, f)
        logger.info(f"Data saved to the file '{file_path}'.")

    # Extracting the activations
    model_wrapper, delta_activations, layer_path_labels = data

    labels = set([key[0] for key in delta_activations.keys()])
    delta_activations_formatted = np.zeros((len(labels), len(labels)))
    for key, value in delta_activations.items():
        delta_activations_formatted[key[0], key[1]] = value["activations_mean"]

    plot_heatmap(
        [[delta_activations_formatted, ],], os.path.join(directory_path, "delta_activations.png"),
        "Delta activations' statistics", ["Delta activations' statistics",], "Layer index", "Layer index",
        [layer_path_labels, ], [layer_path_labels, ], fig_size=fig_size)

    logger.info(f"Activations' analysis completed.")


def perform_delta_activations_same_inputs_analysis(
        config: Config,
) -> None:
    """
    Performs the activations' analysis.

    Args:
        config (Config):
            The configuration object containing the necessary information to perform the analysis.
    """

    logging.basicConfig(filename=os.path.join(config.get("directory_path"), "logs.log"), level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Running perform_sorted_layers_rank_analysis in sorted_layers_rank_analysis.py.")

    # Getting the parameters related to the paths from the configuration
    logger.info(f"Getting the parameters related to the paths from the configuration")
    file_available, file_path, directory_path, file_name, file_name_no_format = [
        config.get(name)
        for name in ["file_available", "file_path", "directory_path", "file_name", "file_name_no_format"]
    ]
    logger.info(f"Information retrieved")

    # Getting the parameters related to the analysis from the configuration
    verbose = config.get_verbose()
    fig_size = config.get("figure_size") if config.contains("figure_size") else (20, 20)
    num_iterations = config.get("num_iterations") if config.contains("num_iterations") else 1
    batch_size = config.get("batch_size") if config.contains("batch_size") else 64
    num_workers = config.get("num_workers") if config.contains("num_workers") else 1
    seed = config.get("seed") if config.contains("seed") else 42
    max_len = config.get("max_len") if config.contains("max_len") else 512

    # Load the data if the file is available, otherwise process the model
    if file_available:
        print(f"The file '{file_path}' is available.")
        logger.info(f"The file '{file_path}' is available.")
        with open(file_path, "rb") as f:
            data = pkl.load(f)
        logger.info(f"Data loaded from the file '{file_path}'.")
    else:
        # Loading the model
        model = load_model_for_sequence_classification(config)
        logger.info(f"Model loaded.")
        # Loading the tokenizer
        tokenizer = load_tokenizer_for_sequence_classification(config)
        logger.info(f"Tokenizer loaded.")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Wrapping the model
        model_wrapper = AnalysisModelWrapper(model, config.get("targets"),
                                             config.get("black_list") if config.contains("black_list") else None)
        logger.info(f"Model wrapped.")
        print(model_wrapper)

        if config.get("dataset_id") == "stanfordnlp/imdb":
            # Loading the dataset
            dataset = IMDBDataModule(
                tokenizer=tokenizer,
                max_len=max_len,
                batch_size=batch_size,
                num_workers=num_workers,
                split=(0.8, 0.1, 0.1),
                seed=seed
            )
            dataset.setup()
        else:
            raise Exception("The dataset is not recognized.")
        logger.info(f"Dataset loaded.")

        # Performing the activation analysis
        data_loader = dataset.train_dataloader()
        verbose.print("Staring to feed the inputs to the model.", Verbose.SILENT)
        delta_activations = {}
        layer_path_labels = None
        for batch_index in tqdm(range(num_iterations)):
            logger.info(f"Batch {batch_index + 1} out of {num_iterations}.")

            batch = next(iter(data_loader))

            # Preparing the inputs
            inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": batch.get("labels")
            }

            # Forward pass through the model
            _ = model_wrapper.forward(**inputs)

            # Getting the activations
            activations = model_wrapper.get_activations()
            # Flattening the activations
            flattened_activations = {}
            flatten_dictionary(flattened_activations, activations)
            flattened_activations = [flattened_activation for flattened_activation in flattened_activations.values() if flattened_activation["activations"] is not None]
            if layer_path_labels is None:
                layer_path_labels = [flattened_activation["path"] for flattened_activation in flattened_activations]

            num_activations = len(flattened_activations)
            for index_activation_1 in range(num_activations):
                for index_activation_2 in range(num_activations):
                    stacked_activation_1 = torch.stack(flattened_activations[index_activation_1]["activations"])
                    stacked_activation_2 = torch.stack(flattened_activations[index_activation_2]["activations"])

                    delta_activation = stacked_activation_1 - stacked_activation_2
                    previous_mean = delta_activations[(index_activation_1, index_activation_2)]["activations_mean"] if (index_activation_1, index_activation_2) in delta_activations else 0
                    previous_num = delta_activations[(index_activation_1, index_activation_2)]["activations_num"] if (index_activation_1, index_activation_2) in delta_activations else 0

                    current_mean = torch.mean(torch.norm(delta_activation, dim=-1) / torch.sqrt(torch.norm(stacked_activation_1, dim=-1) * torch.norm(stacked_activation_2, dim=-1)))
                    current_num = delta_activation.shape[0] * delta_activation.shape[1] * delta_activation.shape[2]

                    delta_activations[(index_activation_1, index_activation_2)] = {
                        "activations_mean": (previous_num * previous_mean + current_num * current_mean) / (previous_num + current_num),
                        "activations_num": previous_num + current_num
                    }

            model_wrapper.reset_activations()

        logger.info(f"Activations and difference between activations computed")

        data = (model_wrapper, delta_activations, layer_path_labels)

        # Saving the activations
        logger.info(f"Storing the data for future usage.")
        with open(f"{file_path}", "wb") as f:
            pkl.dump(data, f)
        logger.info(f"Data saved to the file '{file_path}'.")

    # Extracting the activations
    model_wrapper, delta_activations, layer_path_labels = data

    labels = set([key[0] for key in delta_activations.keys()])
    delta_activations_formatted = np.zeros((len(labels), len(labels)))
    for key, value in delta_activations.items():
        delta_activations_formatted[key[0], key[1]] = value["activations_mean"]

    plot_heatmap(
        [[delta_activations_formatted, ],], os.path.join(directory_path, "delta_activations.png"),
        "Delta activations' statistics", ["Delta activations' statistics",], "Layer index", "Layer index",
        [layer_path_labels, ], [layer_path_labels, ], fig_size=fig_size)

    logger.info(f"Activations' analysis completed.")


def print_nested_dictionary(
        dictionary: dict,
        level: int = 0
) -> None:
    """
    Prints the nested dictionary.

    Args:
        dictionary (dict):
            The dictionary to print.
        level (int):
            The level of the dictionary.
    """

    for key, value in dictionary.items():
        if isinstance(value, dict):
            tabs = "\t" * level
            print(f"{tabs}({key}): ")
            print_nested_dictionary(value, level + 1)
        else:
            tabs = "\t" * level
            print(f"{tabs}({key}): {str(value)[:30]}")


def flatten_dictionary(
        global_dictionary: dict,
        dictionary: dict,
        path: str = ""
) -> None:
    """
    Flattens the nested dictionary.

    Args:
        dictionary (dict):
            The dictionary to flatten.
        global_dictionary (dict):
            The global dictionary to store the results.
    """

    nested_dict = False
    for key, value in dictionary.items():
        if isinstance(value, dict):
            nested_dict = True
            break

    if not nested_dict:
        global_dictionary[path] = dictionary
    else:
        for key, value in dictionary.items():
            new_path = f"{path} -> {key}"
            if isinstance(value, dict):
                flatten_dictionary(global_dictionary, value, new_path)
            else:
                global_dictionary[new_path] = value


def is_at_least_one_element_in_list(
        list_of_elements: list,
        search_list: list
) -> bool:
    """
    Checks if at least one element from the list of elements is present in the search list.

    Args:
        list_of_elements (list):
            The list of elements to search.
        search_list (list):
            The list to search in.
    """

    present_in_search_list = [element for element in list_of_elements if element in search_list]
    if len(present_in_search_list) > 0:
        return True
    return False
