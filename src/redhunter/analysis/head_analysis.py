import os
import pickle as pkl
import csv
from tqdm import tqdm
import logging

from matplotlib import pyplot as plt
import seaborn as sns

import numpy as np

import torch

from exporch import Config, Verbose
from exporch.utils.causal_language_modeling import load_model_for_causal_lm

from redhunter.analysis.rank_analysis_experiment import compute_explained_variance

from redhunter.analysis.sorted_layers_compression_analysis import compute_cosine
from redhunter.analysis.analysis_utils import AnalysisTensorWrapper, extract


def extract_heads(
        tensor_wrapper: AnalysisTensorWrapper,
        num_heads: int,
        head_dim: int
) -> AnalysisTensorWrapper:
    """
    Extracts the heads from the tensor wrapper.

    Args:
        tensor_wrapper: AnalysisTensorWrapper
            The tensor wrapper to extract the heads from.
        num_heads: int
            The number of heads to extract.
        head_dim: int
            The dimension of the head.

    Returns:
        AnalysisTensorWrapper:
            The tensor wrapper with the extracted heads.
    """

    heads = []
    for i in range(num_heads):
        head_wrapper = AnalysisTensorWrapper(
            tensor_wrapper.get_tensor()[i * head_dim:(i + 1) * head_dim, :],
            name=tensor_wrapper.get_name() + f" Head {i}",
            label=tensor_wrapper.get_label() + f" Head {i}",
            path=tensor_wrapper.get_path(),
            block_index=tensor_wrapper.get_block_index(),
            layer=tensor_wrapper.get_layer(),
            precision=tensor_wrapper.get_precision(),
            verbose=tensor_wrapper.get_verbose()
        )
        head_wrapper.compute_singular_values()
        heads.append(head_wrapper)

    tensor_wrapper.set_attribute("heads", heads)

    return tensor_wrapper


def perform_head_analysis(
        configuration: Config,
) -> None:
    """
    Perform the analysis of the heads of the attention mechanism.

    Args:
        configuration: Config
            The configuration object containing the necessary information to perform the analysis.
    """

    # Getting the parameters related to the paths from the configuration
    file_available = configuration.get("file_available")
    file_path = configuration.get("file_path")
    directory_path = configuration.get("directory_path")
    file_name_no_format = configuration.get("file_name_no_format")

    # Getting the parameters related to the analysis from the configuration
    verbose = configuration.get_verbose()
    fig_size = configuration.get("figure_size") if configuration.contains("figure_size") else (20, 20)
    name_num_heads_mapping = configuration.get("name_num_heads_mapping")
    name_dim_heads_mapping = (
        configuration.get("name_dim_heads_mapping") if configuration.contains("name_dim_heads_mapping") else None
    )
    explained_variance_threshold = configuration.get("explained_variance_threshold")

    # Preparing CSV file path
    csv_file_path = os.path.join(directory_path, file_name_no_format + "_analysis.csv")

    # Opening the CSV file for writing
    with open(csv_file_path, mode='w', newline='') as csvfile:
        # Define the field names for the CSV file
        fieldnames = [
            "Tensor Path", "Tensor Shape", "Number of Heads", "Heads Shape", "Explained Variance Threshold",
            "Tensor Approximated Rank", "Total Heads Rank", "Average Heads Rank", "Tensor Parameters Count",
            "Total Heads Parameters Count", "Average Heads Parameters Count"
        ]

        # Loading the data if the file is available, otherwise process the model
        if file_available:
            print(f"The file '{file_path}' is available.")
            with open(file_path, "rb") as f:
                tensor_wrappers_to_analyze, tensor_wrappers_num_heads = pkl.load(f)
        else:
            # Loading the model
            model = load_model_for_causal_lm(configuration)
            # Extracting the tensors to be analyzed
            extracted_tensors = []
            extract(
                model,
                configuration.get("targets"),
                extracted_tensors,
                configuration.get("black_list"),
                verbose=verbose
            )

            # Choosing the actual tensors to be analyzed
            tensor_wrappers_to_analyze = extracted_tensors
            tensor_wrappers_num_heads = []

            for tensor_wrapper in tensor_wrappers_to_analyze:
                verbose.print(f"Analyzing the tensor with path '{tensor_wrapper.get_path()}'", Verbose.INFO)
                # Computing the singular values of the tensor
                tensor_wrapper.compute_singular_values()

                # Defining the head-related parameters
                output_size, input_size = tensor_wrapper.get_shape()
                num_heads = model.__dict__["config"].__dict__[name_num_heads_mapping[tensor_wrapper.get_name()]]
                tensor_wrappers_num_heads.append(num_heads)
                if name_dim_heads_mapping is None:
                    if output_size % num_heads != 0:
                        raise Exception("The output size of the tensor is not divisible by the number of heads.")
                    head_dim = output_size // num_heads
                else:
                    head_dim = model.__dict__["config"].__dict__[name_dim_heads_mapping[tensor_wrapper.get_name()]]

                # Extracting the heads
                extract_heads(tensor_wrapper, num_heads, head_dim)
                verbose.print(
                    f"Extracted {num_heads} heads from the tensor with path '{tensor_wrapper.get_path()}'",
                    Verbose.DEBUG
                )

            # Saving the processed data for future use
            #with open(file_path, "wb") as f:
            #    pkl.dump((tensor_wrappers_to_analyze, tensor_wrappers_num_heads), f)
            #verbose.print(f"Data has been saved to '{file_path}'.", Verbose.INFO)

        # Extending fieldnames based on the maximum number of heads
        for i in range(max(tensor_wrappers_num_heads)):
            fieldnames.append(f"Head {i} Rank")
            fieldnames.append(f"Head {i} Parameters Count")

        # Creating the CSV writer object
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for tensor_wrapper_index, tensor_wrapper in enumerate(tensor_wrappers_to_analyze):
            verbose.print(f"Analyzing the tensor with path '{tensor_wrapper.get_path()}'", Verbose.INFO)

            # Computing the rank and the number of parameters of the tensor
            tensor_rank = tensor_wrapper.get_rank(
                explained_variance_threshold=explained_variance_threshold,
                relative=False
            )
            tensor_parameters_count = tensor_wrapper.get_parameters_count_thresholded(
                explained_variance_threshold=explained_variance_threshold
            )

            fig, axes = plt.subplots(2, 2, figsize=fig_size)
            color_map = plt.cm.get_cmap('viridis', tensor_wrappers_num_heads[tensor_wrapper_index])
            fig.suptitle(f"Analysis of the tensor with path '{tensor_wrapper.get_path()}'")

            # Plotting the singular values and explained variance of the tensor
            axes[0, 0].plot(tensor_wrapper.get_singular_values())
            axes[0, 0].set_title("Singular Values")
            axes[0, 0].set_xlabel("Component")
            axes[0, 0].set_ylabel("Singular Value")

            axes[0, 1].plot(tensor_wrapper.get_explained_variance())
            axes[0, 1].set_title("Explained Variance")
            axes[0, 1].set_xlabel("Component")
            axes[0, 1].set_ylabel("Explained Variance (%)")
            axes[0, 1].axhline(y=explained_variance_threshold, color='black', linestyle='--', label='Threshold')

            heads_ranks = []
            heads_parameters_counts = []
            head_data = {}
            heads_shape = tensor_wrapper.get_attribute("heads")[0].get_shape()
            for i, head_wrapper in enumerate(tensor_wrapper.get_attribute("heads")):
                # Plotting the singular values and explained variance of the head
                color = color_map(i)
                axes[1, 0].plot(head_wrapper.get_singular_values(), label=head_wrapper.get_label(), color=color)
                axes[1, 1].plot(head_wrapper.get_explained_variance(), label=head_wrapper.get_label(), color=color)

                axes[1, 0].set_title("Head Singular Values")
                axes[1, 0].set_xlabel("Component")
                axes[1, 0].set_ylabel("Singular Value")

                axes[1, 1].set_title("Head Explained Variance")
                axes[1, 1].set_xlabel("Component")
                axes[1, 1].set_ylabel("Explained Variance (%)")

                # Computing the rank and the number of parameters of the head
                head_rank = head_wrapper.get_rank(
                    explained_variance_threshold=explained_variance_threshold,
                    relative=False
                )
                head_parameters_count = head_wrapper.get_parameters_count_thresholded(
                    explained_variance_threshold=explained_variance_threshold
                )
                heads_ranks.append(head_rank)
                heads_parameters_counts.append(head_parameters_count)

                head_data[f"Head {i} Rank"] = head_rank
                head_data[f"Head {i} Parameters Count"] = head_parameters_count

            if len(tensor_wrapper.get_attribute("heads")) > 1:
                singular_values_heads_average = tensor_wrapper.get_attribute("heads")[0].get_singular_values()
                for head_wrapper in tensor_wrapper.get_attribute("heads")[1:]:
                    singular_values_heads_average += head_wrapper.get_singular_values()
                singular_values_heads_average /= len(tensor_wrapper.get_attribute("heads"))
                axes[1, 0].plot(singular_values_heads_average, color="red", label="Average singular values heads")
                axes[1, 1].plot(compute_explained_variance(singular_values_heads_average), color="red", label="Average singular values heads")

            axes[1, 1].axhline(y=explained_variance_threshold, color="black", linestyle="--", label="Threshold")
            axes[1, 0].legend()
            axes[1, 1].legend()

            # Saving the plot
            fig.savefig(os.path.join(directory_path, file_name_no_format + f"_{tensor_wrapper.get_path()}.png"))
            plt.close(fig)
            verbose.print(f"Plots of the tensor with path '{tensor_wrapper.get_path()}' has been saved.", Verbose.INFO)

            # Writing the row data to the CSV file
            row_data = {
                "Tensor Path": tensor_wrapper.get_path() + f"_{tensor_wrapper.get_name()}",
                "Tensor Shape": f"({tensor_wrapper.get_shape()[0]}, {tensor_wrapper.get_shape()[1]})",
                "Number of Heads": tensor_wrappers_num_heads[tensor_wrapper_index],
                "Heads Shape": f"({heads_shape[0]}, {heads_shape[1]})",
                "Explained Variance Threshold": explained_variance_threshold,
                "Tensor Approximated Rank": tensor_rank,
                "Total Heads Rank": sum(heads_ranks),
                "Average Heads Rank": f"{(sum(heads_ranks) / len(heads_ranks)):.2f}",
                "Tensor Parameters Count": tensor_parameters_count,
                "Total Heads Parameters Count": sum(heads_parameters_counts),
                "Average Heads Parameters Count": f"{(sum(heads_parameters_counts) / len(heads_parameters_counts)):.2f}"
            }
            row_data.update(head_data)
            writer.writerow(row_data)
            verbose.print(
                f"Information of the analysis of the tensor with path '{tensor_wrapper.get_path()}' has been saved.",
                Verbose.INFO
            )

    # Saving the analyzed tensors to the file
    with open(file_path, "wb") as f:
        pkl.dump((tensor_wrappers_to_analyze, tensor_wrappers_num_heads), f)


def perform_heads_similarity_analysis(
        configuration: Config
) -> None:
    """
    Perform the analysis of the similarity between the heads of the attention mechanism.

    Args:
        configuration: Config
            The configuration object containing the necessary information to perform the analysis.
    """

    # Getting the parameters related to the paths from the configuration
    file_available = configuration.get("file_available")
    file_path = configuration.get("file_path")
    directory_path = configuration.get("directory_path")
    file_name_no_format = configuration.get("file_name_no_format")

    # Getting the parameters related to the analysis from the configuration
    verbose = configuration.get_verbose()
    fig_size = configuration.get("figure_size") if configuration.contains("figure_size") else (20, 20)
    name_num_heads_mapping = configuration.get("name_num_heads_mapping")
    name_dim_heads_mapping = (
        configuration.get("name_dim_heads_mapping") if configuration.contains("name_dim_heads_mapping") else None
    )
    num_samples = configuration.get("num_samples") if configuration.contains("num_samples") else 1024

    # Loading the data if the file is available, otherwise processing the model
    if file_available:
        print(f"The file '{file_path}' is available.")
        with open(file_path, "rb") as f:
            grouped_tensor_wrappers_to_analyze, grouped_function_similarities, grouped_tensor_wrappers_num_heads = \
                pkl.load(f)
    else:
        # Loading the model
        model = load_model_for_causal_lm(configuration)
        # Extracting the tensors to be analysed
        extracted_tensors = []
        extract(
            model,
            configuration.get("targets"),
            extracted_tensors,
            configuration.get("black_list"),
            verbose=verbose)

        # Choosing the actual tensors to be analyzed
        tensor_wrappers_to_analyze = extracted_tensors

        # Grouping the tensors based on some criteria
        grouped_tensor_wrappers_num_heads = []
        grouped_function_similarities = []
        if configuration.get("grouping") == "single_matrix":
            grouped_tensor_wrappers_to_analyze = [
                {"tensors": [tensor_wrapper], "label": tensor_wrapper.get_path()}
                for tensor_wrapper in tensor_wrappers_to_analyze
            ]
        elif configuration.get("grouping") == "matrix_type":
            unique_labels = set([tensor_wrapper.get_label() for tensor_wrapper in tensor_wrappers_to_analyze])
            grouped_tensor_wrappers_to_analyze = [
                {"tensors": [], "label": unique_label} for unique_label in unique_labels
            ]
            for tensor_wrapper in tensor_wrappers_to_analyze:
                for group_index in range(len(grouped_tensor_wrappers_to_analyze)):
                    if grouped_tensor_wrappers_to_analyze[group_index]["label"] == tensor_wrapper.get_label():
                        grouped_tensor_wrappers_to_analyze[group_index]["tensors"].append(tensor_wrapper)
        else:
            grouped_tensor_wrappers_to_analyze = [{"tensors": tensor_wrappers_to_analyze, "label": "all_tensors"}]

        for tensor_wrappers_group_dict in grouped_tensor_wrappers_to_analyze:
            tensor_wrappers_group = tensor_wrappers_group_dict["tensors"]
            y_list = []
            similarity_size = 0
            tensor_wrappers_num_heads = []
            for tensor_wrapper in tensor_wrappers_group:
                verbose.print(f"Analyzing the tensor with path '{tensor_wrapper.get_path()}'", Verbose.INFO)
                # Defining the head-related parameters
                output_size, input_size = tensor_wrapper.get_shape()
                if input_size != tensor_wrappers_to_analyze[0].get_shape()[1]:
                    raise Exception("All the tensors must have the same input size")

                num_heads = model.__dict__["config"].__dict__[name_num_heads_mapping[tensor_wrapper.get_name()]]
                similarity_size += num_heads
                tensor_wrappers_num_heads.append(num_heads)
                if name_dim_heads_mapping is None:
                    if output_size % num_heads != 0:
                        raise Exception("The output size of the tensor is not divisible by the number of heads.")
                    head_dim = output_size // num_heads
                else:
                    head_dim = model.__dict__["config"].__dict__[name_dim_heads_mapping[tensor_wrapper.get_name()]]

                # Extracting the heads
                extract_heads(tensor_wrapper, num_heads, head_dim)
                verbose.print(
                    f"Extracted {num_heads} heads from the tensor with path '{tensor_wrapper.get_path()}'",
                    Verbose.DEBUG
                )

                # Generating random input
                input_dim = tensor_wrappers_to_analyze[0].get_shape()[1]
                x = np.random.randn(input_dim, num_samples)
                x = torch.tensor(x, dtype=torch.float32)

                # Generating outputs for each head (for illustration, random matrices are used)
                for _ in range(num_heads):
                    head = np.random.randn(input_dim, input_dim)
                    head = torch.tensor(head, dtype=torch.float32)
                    y = head @ x
                    y_list.append(y)

                verbose.print(
                    f"Generated outputs for the heads of the tensor with path '{tensor_wrapper.get_path()}'\n",
                    Verbose.INFO
                )

            grouped_tensor_wrappers_num_heads.append(tensor_wrappers_num_heads)

            verbose.print(f"Stating the computation of the similarities", Verbose.INFO)
            function_similarities = np.zeros((similarity_size, similarity_size))
            # Computing similarities and fill the matrix
            for index_1 in tqdm(range(len(y_list))):
                y_1 = y_list[index_1]
                for index_2 in range(len(y_list)):
                    y_2 = y_list[index_2]
                    if index_2 > index_1:
                        similarity = compute_cosine(y_1, y_2, dim=0).detach().numpy()
                        similarity_mean = similarity.mean()
                        function_similarities[index_1, index_2] = similarity_mean
                        function_similarities[index_2, index_1] = similarity_mean

            grouped_function_similarities.append(function_similarities)

    for index, function_similarities in enumerate(grouped_function_similarities):
        # Plotting the similarity matrix
        fig, ax = plt.subplots(figsize=fig_size)
        fig.suptitle(f"Analysis of the tensor with path '{grouped_tensor_wrappers_to_analyze[index]['label']}'")
        sns.heatmap(function_similarities, annot=True, cmap="viridis", cbar=True, ax=ax)
        ax.set_title("Cosine Similarity Heatmap")
        ax.set_xlabel("Head Index")
        ax.set_ylabel("Head Index")

        # Saving the plot
        fig.savefig(
            os.path.join(
                directory_path,
                file_name_no_format + f"_{grouped_tensor_wrappers_to_analyze[index]['label']}_similarities.png"
            )
        )
        plt.close(fig)
        verbose.print(
            f"Plots of the tensor with path '{grouped_tensor_wrappers_to_analyze[index]['label']}' has been "
            f"saved.",
            Verbose.INFO
        )

    # Saving the data for future use
    with open(file_path, "wb") as f:
        pkl.dump(
            (
                grouped_tensor_wrappers_to_analyze,
                grouped_function_similarities,
                grouped_tensor_wrappers_num_heads
            ),
            f
        )
    print(f"Data has been saved to '{file_path}'.")


def perform_query_key_analysis(
        configuration: Config
) -> None:
    """
    Perform the analysis of the ranks of the concatenation of queries and keys of the attention mechanism that are in
    the same block.

    Args:
        configuration: Config
            The configuration object containing the necessary information to perform the analysis.
    """

    # Getting the parameters related to the paths from the configuration
    file_available = configuration.get("file_available")
    file_path = configuration.get("file_path")
    directory_path = configuration.get("directory_path")
    file_name_no_format = configuration.get("file_name_no_format")

    # Getting the parameters related to the analysis from the configuration
    verbose = configuration.get_verbose()
    fig_size = configuration.get("figure_size") if configuration.contains("figure_size") else (20, 20)
    name_num_heads_mapping = configuration.get("name_num_heads_mapping")
    name_dim_heads_mapping = (
        configuration.get("name_dim_heads_mapping") if configuration.contains("name_dim_heads_mapping") else None
    )
    query_label = configuration.get("query_label")
    key_label = configuration.get("key_label")
    targets = [query_label, key_label]
    explained_variance_threshold = configuration.get("explained_variance_threshold")

    # Preparing CSV file path
    csv_file_path = os.path.join(directory_path, file_name_no_format + "_analysis.csv")

    # Loading the data if the file is available, otherwise process the model
    if file_available:
        print(f"The file '{file_path}' is available.")
        with open(file_path, "rb") as f:
            tensor_wrappers_to_analyze, tensor_wrappers_num_heads = pkl.load(f)
    else:
        # Loading the model
        model = load_model_for_causal_lm(configuration)
        # Extracting the tensors to be analyzed
        extracted_tensors = []
        extract(
            model,
            configuration.get("targets"),
            extracted_tensors,
            configuration.get("black_list"),
            verbose=verbose
        )
        # Choosing the actual tensors to be analyzed
        tensor_wrappers = extracted_tensors[:1]

        # Grouping the tensors based on some criteria
        queries = [tensor_wrapper for tensor_wrapper in tensor_wrappers if tensor_wrapper.get_label() == query_label]
        keys = [tensor_wrapper for tensor_wrapper in tensor_wrappers if tensor_wrapper.get_label() == key_label]
        if len(queries) != len(keys):
            raise Exception("The number of queries and keys must be the same.")
        query_key_couples = [[queries[i], keys[i]] for i in range(len(queries))]
        tensor_wrappers_num_heads = []

        for query_key_couple in query_key_couples:
            query = query_key_couple[0]
            key = query_key_couple[1]
            verbose.print(f"Analyzing the tensors with paths '{query.get_path()}' and '{key.get_path()}", Verbose.INFO)

            # Computing the singular values of the query and the key
            query.compute_singular_values()
            key.compute_singular_values()
            tensor_wrapper = key

            # Defining the head-related parameters
            output_size, input_size = tensor_wrapper.get_shape()
            num_heads = model.__dict__["config"].__dict__[name_num_heads_mapping[tensor_wrapper.get_name()]]
            tensor_wrappers_num_heads.append(num_heads)
            if name_dim_heads_mapping is None:
                if output_size % num_heads != 0:
                    raise Exception("The output size of the tensor is not divisible by the number of heads.")
                head_dim = output_size // num_heads
            else:
                head_dim = model.__dict__["config"].__dict__[name_dim_heads_mapping[tensor_wrapper.get_name()]]

            # Extracting the heads
            extract_heads(tensor_wrapper, num_heads, head_dim)
            verbose.print(
                f"Extracted {num_heads} heads from the tensor with path '{tensor_wrapper.get_path()}'",
                Verbose.DEBUG
            )

        # Saving the processed data for future use
        # with open(file_path, "wb") as f:
        #    pkl.dump((tensor_wrappers_to_analyze, tensor_wrappers_num_heads), f)
        # verbose.print(f"Data has been saved to '{file_path}'.", Verbose.INFO)

    # Opening the CSV file for writing
    with open(csv_file_path, mode='w', newline='') as csvfile:
        # Define the field names for the CSV file
        fieldnames = [
            "Query Path", "Key Path" "Query Shape", "Key Shape", "Number of Heads Query", "Number of Heads Key",
            "Query Heads Shape", "Key Heads Shape", "Explained Variance Threshold", "Tensor Approximated Rank",
            "Tensor Parameters Count", "Query Approximated Rank", "Query Parameters Count", "Key Approximated Rank",
            "Key Parameters Count"
        ]

        # Extending fieldnames based on the maximum number of heads
        for i in range(max(tensor_wrappers_num_heads)):
            fieldnames.append(f"Head {i} Rank")
            fieldnames.append(f"Head {i} Parameters Count")

        # Creating the CSV writer object
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for tensor_wrapper_index, tensor_wrapper in enumerate(tensor_wrappers_to_analyze):
            verbose.print(f"Analyzing the tensor with path '{tensor_wrapper.get_path()}'", Verbose.INFO)

            # Computing the rank and the number of parameters of the tensor
            tensor_rank = tensor_wrapper.get_rank(
                explained_variance_threshold=explained_variance_threshold,
                relative=False
            )
            tensor_parameters_count = tensor_wrapper.get_parameters_count_thresholded(
                explained_variance_threshold=explained_variance_threshold
            )

            fig, axes = plt.subplots(2, 2, figsize=fig_size)
            color_map = plt.cm.get_cmap('viridis', tensor_wrappers_num_heads[tensor_wrapper_index])
            fig.suptitle(f"Analysis of the tensor with path '{tensor_wrapper.get_path()}'")

            # Plotting the singular values and explained variance of the tensor
            axes[0, 0].plot(tensor_wrapper.get_singular_values())
            axes[0, 0].set_title("Singular Values")
            axes[0, 0].set_xlabel("Component")
            axes[0, 0].set_ylabel("Singular Value")

            axes[0, 1].plot(tensor_wrapper.get_explained_variance())
            axes[0, 1].set_title("Explained Variance")
            axes[0, 1].set_xlabel("Component")
            axes[0, 1].set_ylabel("Explained Variance (%)")
            axes[0, 1].axhline(y=explained_variance_threshold, color='black', linestyle='--', label='Threshold')

            heads_ranks = []
            heads_parameters_counts = []
            head_data = {}
            heads_shape = tensor_wrapper.get_attribute("heads")[0].get_shape()
            for i, head_wrapper in enumerate(tensor_wrapper.get_attribute("heads")):
                # Plotting the singular values and explained variance of the head
                color = color_map(i)
                axes[1, 0].plot(head_wrapper.get_singular_values(), label=head_wrapper.get_label(), color=color)
                axes[1, 1].plot(head_wrapper.get_explained_variance(), label=head_wrapper.get_label(), color=color)

                axes[1, 0].set_title("Head Singular Values")
                axes[1, 0].set_xlabel("Component")
                axes[1, 0].set_ylabel("Singular Value")

                axes[1, 1].set_title("Head Explained Variance")
                axes[1, 1].set_xlabel("Component")
                axes[1, 1].set_ylabel("Explained Variance (%)")

                # Computing the rank and the number of parameters of the head
                head_rank = head_wrapper.get_rank(
                    explained_variance_threshold=explained_variance_threshold,
                    relative=False
                )
                head_parameters_count = head_wrapper.get_parameters_count_thresholded(
                    explained_variance_threshold=explained_variance_threshold
                )
                heads_ranks.append(head_rank)
                heads_parameters_counts.append(head_parameters_count)

                head_data[f"Head {i} Rank"] = head_rank
                head_data[f"Head {i} Parameters Count"] = head_parameters_count

            axes[1, 1].axhline(y=explained_variance_threshold, color='black', linestyle='--', label='Threshold')
            axes[1, 0].legend()
            axes[1, 1].legend()

            # Saving the plot
            fig.savefig(os.path.join(directory_path, file_name_no_format + f"_{tensor_wrapper.get_path()}.png"))
            plt.close(fig)
            verbose.print(f"Plots of the tensor with path '{tensor_wrapper.get_path()}' has been saved.", Verbose.INFO)

            # Writing the row data to the CSV file
            row_data = {
                "Tensor Path": tensor_wrapper.get_path() + f"_{tensor_wrapper.get_name()}",
                "Tensor Shape": f"({tensor_wrapper.get_shape()[0]}, {tensor_wrapper.get_shape()[1]})",
                "Number of Heads": tensor_wrappers_num_heads[tensor_wrapper_index],
                "Heads Shape": f"({heads_shape[0]}, {heads_shape[1]})",
                "Explained Variance Threshold": explained_variance_threshold,
                "Tensor Approximated Rank": tensor_rank,
                "Total Heads Rank": sum(heads_ranks),
                "Average Heads Rank": f"{(sum(heads_ranks) / len(heads_ranks)):.2f}",
                "Tensor Parameters Count": tensor_parameters_count,
                "Total Heads Parameters Count": sum(heads_parameters_counts),
                "Average Heads Parameters Count": f"{(sum(heads_parameters_counts) / len(heads_parameters_counts)):.2f}"
            }
            row_data.update(head_data)
            writer.writerow(row_data)
            verbose.print(
                f"Information of the analysis of the tensor with path '{tensor_wrapper.get_path()}' has been saved.",
                Verbose.INFO
            )

    # Saving the analyzed tensors to the file
    with open(file_path, "wb") as f:
        pkl.dump((tensor_wrappers_to_analyze, tensor_wrappers_num_heads), f)