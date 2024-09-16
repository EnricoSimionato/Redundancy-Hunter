import os
import time
import csv
import pickle as pkl
import copy
import logging
from tqdm import tqdm

import matplotlib.pyplot as plt

import torch

from exporch import Config, Verbose, get_available_device

from exporch.utils.causal_language_modeling import load_model_for_causal_lm

from redhunter.analysis.analysis_utils import AnalysisTensorDict, extract_based_on_path


def tensor_loss(
    weight_tensor: torch.Tensor,
    target_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Computes the tensor loss.

    Args:
        weight_tensor (torch.Tensor):
            Tensor that is the approximation of the target.
        target_tensor (torch.Tensor):
            Target tensor.

    Returns:
        (torch.Tensor):
            The tensor loss.
    """

    return torch.norm(target_tensor - weight_tensor) ** 2


def activation_loss(
        activation_prediction: torch.Tensor,
        activation_target: torch.Tensor
) -> torch.Tensor:
    """
    Computes the activation loss.

    Args:
        activation_prediction (torch.Tensor):
            Activation that is the approximation of the target.
        activation_target (torch.Tensor):
            Activation tensor.

    Returns:
        (torch.Tensor):
            The activation loss.
    """

    return (torch.norm(activation_target - activation_prediction, dim=0) ** 2).mean()


def get_ab_factorization(
        tensor: torch.Tensor,
        rank: int,
        trainable: list[bool],
        device: torch.device
) -> (torch.Tensor, torch.Tensor, float):
    """
    Computes the AB factorization of a given tensor.

    Args:
        tensor (torch.Tensor):
            The tensor to be factorized.
        rank (int):
            The rank of the factorization.
        trainable (list[bool]):
            A list of booleans indicating whether the corresponding factor should be trainable.
        device (torch.device):
            The device to perform the factorization on.

    Returns:
        (torch.Tensor):
            The A factor of the factorization.
        (torch.Tensor):
            The B factor of the factorization.
        (float):
            The time elapsed to compute the factorization.
    """

    out_shape, in_shape = tensor.shape

    # Initializing the AB factorization
    start_time = time.time()
    a = torch.randn(out_shape, rank, dtype=tensor.dtype).to(device)
    a.requires_grad = trainable[0]
    b = torch.randn(rank, in_shape, dtype=tensor.dtype).to(device)
    b.requires_grad = trainable[1]
    elapsed_time = time.time() - start_time

    return a, b, elapsed_time


def get_svd_factorization(
        tensor: torch.Tensor,
        rank: int,
        trainable: list[bool],
        device: torch.device
) -> (torch.Tensor, torch.Tensor, torch.Tensor, float):
    """
    Computes the SVD factorization of a given tensor.

    Args:
        tensor (torch.Tensor):
            The tensor to be factorized.
        rank (int):
            The rank of the factorization.
        trainable (list[bool]):
            A list of booleans indicating whether the corresponding factor should be trainable.
        device (torch.device):
            The device to perform the factorization on.

    Returns:
        (torch.Tensor):
            The U factor of the factorization.
        (torch.Tensor):
            The S factor of the factorization.
        (torch.Tensor):
            The V factor of the factorization.
        (float):
            The time elapsed to compute the factorization.
    """

    # Initializing the SVD factorization
    start_time = time.time()
    u, s, v = torch.svd(tensor.to(torch.float32).to("cpu"))
    us = torch.matmul(u[:, :rank], torch.diag(s[:rank])).to(tensor.dtype).to(device)
    us.requires_grad = trainable[0]
    vt = v.T
    vt = vt[:rank, :].to(tensor.dtype).to(device)
    vt.requires_grad = trainable[1]
    elapsed_time = time.time() - start_time

    return us, vt, elapsed_time


def get_global_matrix_factorization(
        tensor: torch.Tensor,
        global_matrix: torch.Tensor,
        rank: int,
        trainable: bool,
        initialization_type: str,
        device: torch.device
) -> (torch.Tensor, torch.Tensor, float):
    """
    Computes the factorization of a given tensor using a global matrix.

    Args:
        tensor (torch.Tensor):
            The tensor to be factorized.
        global_matrix (torch.Tensor):
            The global matrix to be used in the factorization.
        rank (int):
            The rank of the factorization.
        trainable (bool):
            A boolean indicating whether the factor should be trainable.
        initialization_type (str):
            The type of initialization to use.
        device (torch.device):
            The device to perform the factorization on.

    Returns:
        (torch.Tensor):
            The factor of the factorization.
        (torch.Tensor):
            The factor of the factorization.
        (float):
            The time elapsed to compute the factorization.
    """

    out_shape, in_shape = tensor.shape

    if global_matrix.shape[0] != out_shape or global_matrix.shape[1] != rank:
        raise ValueError("The global matrix must have the shape (out_shape, rank).")

    start_time = time.time()
    if initialization_type == "random":
        b = torch.randn(rank, in_shape, dtype=tensor.dtype).to(device)
        b.requires_grad = trainable

    elif initialization_type == "pseudo-inverse":
        b = torch.matmul(
            torch.linalg.pinv(global_matrix.to(torch.float32).to("cpu")).to(device),
            tensor.to(torch.float32).to(device)
        ).to(tensor.dtype)
        b.requires_grad = trainable

    else:
        raise ValueError("Unknown initialization type.")
    elapsed_time = time.time() - start_time

    return global_matrix, b, elapsed_time


def perform_simple_initialization_analysis(
        configuration: Config,
) -> None:
    """
    Compares which of the two initializations are better in terms of the quality loss and the speed of convergence to a
    good approximation.

    Args:
        configuration (Config):
            The configuration object.
    """

    logging.basicConfig(filename=os.path.join(configuration.get("directory_path"), "logs.log"), level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Running perform_simple_initialization_analysis in matrix_initialization_analysis.py.")

    # Getting the parameters related to the paths from the configuration
    logger.info(f"Getting the parameters related to the paths from the configuration")
    file_available = configuration.get("file_available")
    file_path = configuration.get("file_path")
    directory_path = configuration.get("directory_path")
    file_name = configuration.get("file_name")
    file_name_no_format = configuration.get("file_name_no_format")
    logger.info(f"Information retrieved")


    # Getting the parameters related to the analysis from the configuration
    logger.info(f"Getting the parameters related to the analysis from the configuration")
    verbose = configuration.get_verbose()
    fig_size = configuration.get("figure_size") if configuration.contains("figure_size") else (20, 20)
    rank = configuration.get("rank")
    num_epochs = configuration.get("num_epochs") if configuration.contains("num_epochs") else 5000
    num_samples = configuration.get("num_samples") if configuration.contains("num_samples") else 64
    device = get_available_device(configuration.get("device") if configuration.contains("device") else "cuda")
    logger.info(f"Information retrieved")

    if file_available:
        print(f"The file '{file_path}' is available.")
        logger.info(f"The file '{file_path}' is available.")
        # Load the data from the file
        logger.info(f"Trying to load the data from the file")
        with open(file_path, "rb") as f:
            all_loss_histories = pkl.load(f)
            logger.info(f"Data loaded")
    else:
        logger.info(f"File not available")
        logger.info(f"Trying to the model: {configuration.get('original_model_id')}")
        # Loading the model
        model = load_model_for_causal_lm(configuration)
        logger.info(f"Model loaded")

        logger.info(f"Trying to extract the candidate tensors for the analysis")
        # Extracting the candidate tensors for the analysis
        extracted_tensors = []
        extract_based_on_path(
            model,
            configuration.get("targets"),
            extracted_tensors,
            configuration.get("black_list"),
            verbose=verbose
        )
        logger.info(f"Candidate tensors extracted")
        # Choosing the actual tensors to analyze
        tensor_wrappers_to_analyze = [extracted_tensors[0]]

        time_log = []
        csv_data = []
        all_loss_histories = {}
        logger.info(f"Starting the analysis of the tensors")
        for tensor_wrapper_to_analyze in tensor_wrappers_to_analyze:
            verbose.print(f"\nAnalyzing tensor: {tensor_wrapper_to_analyze.get_path()}", Verbose.INFO)
            logger.info(f"Analyzing tensor: {tensor_wrapper_to_analyze.get_path()}")
            tensor_to_analyze = tensor_wrapper_to_analyze.get_tensor().to(torch.float32)

            # Preparing the data
            out_shape, in_shape = tensor_to_analyze.shape
            random_x = torch.randn(in_shape, num_samples).to(device)
            test_random_x = torch.randn(in_shape, num_samples).to(device)
            tensor_to_analyze = tensor_to_analyze.to(device)
            tensorx = torch.matmul(tensor_to_analyze, random_x)
            logger.info("Random inputs initialized")

            # Initializing the factorizations
            a, b, ab_time = get_ab_factorization(tensor_to_analyze, rank, [False, True], device)
            us, vt, svd_time = get_svd_factorization(tensor_to_analyze, rank, [False, True], device)
            logger.info("Factorizations initialized")

            tensor_factorizations = {
                "A, B SVD initialized": {"tensors": [vt, us], "init_time": svd_time},
                "A, B randomly initialized": {"tensors": [b, a], "init_time": ab_time}
            }

            loss_types = ["activation loss", "tensor loss"]
            loss_histories_factorizations = {"activation loss": {}, "tensor loss": {}}

            # Training the factorizations
            for factorization_label, factorization_init_and_init_time in tensor_factorizations.items():
                logger.info("Getting the factorization")
                factorization_init = factorization_init_and_init_time["tensors"]
                init_time = factorization_init_and_init_time["init_time"]
                logger.info("Factorization obtained")
                for loss_type in loss_types:
                    logger.info("Defining the tensors to train")
                    # Cloning the tensors to avoid in-place operations
                    factorization = [tensor.clone().detach() for tensor in factorization_init]
                    for index in range(len(factorization)):
                        factorization[index].requires_grad = factorization_init[index].requires_grad

                    logger.info("Setting the optimizer")
                    # Setting the optimizer
                    trainable_tensors = [tensor for tensor in factorization if tensor.requires_grad]
                    optimizer = torch.optim.AdamW(
                        trainable_tensors,
                        lr=configuration.get("learning_rate") if configuration.contains("learning_rate") else 1e-4,
                        eps=1e-7 if tensor_to_analyze.dtype == torch.float16 else 1e-8
                    )

                    activation_loss_history = []
                    tensor_loss_history = []
                    initial_activation_loss = None
                    initial_tensor_loss = None

                    verbose.print(f"\nStarting training using {loss_type} for {factorization_label}", Verbose.INFO)
                    for index in range(len(factorization)):
                        verbose.print(
                            f"Tensor {index} in {factorization_label} requires grad: "
                            f"{factorization[index].requires_grad}", Verbose.INFO
                        )

                    logger.info(
                        f"Starting raining the factorization: {factorization_label} of the tensor: {tensor_wrapper_to_analyze.get_path()} with the loss type: {loss_type}")
                    start_time = time.time()
                    # Training the factorization
                    for epoch in tqdm(range(num_epochs)):
                        logger.info(f"Epoch {epoch}")
                        x = random_x.clone().detach().to(device)
                        y = torch.eye(in_shape).to(device)
                        for tensor in factorization:
                            x = torch.matmul(tensor, x)
                            y = torch.matmul(tensor, y)
                        logger.info(f"Factorization computed")
                        # Computing the losses
                        activation_loss = (torch.norm((tensorx - x), dim=0) ** 2).mean()
                        tensor_loss = torch.norm(tensor_to_analyze - y) ** 2
                        logger.info(f"Losses computed")
                        if initial_activation_loss is None:
                            initial_activation_loss = activation_loss.item()
                        if initial_tensor_loss is None:
                            initial_tensor_loss = tensor_loss.item()

                        if loss_type == "activation loss":
                            loss = activation_loss
                        elif loss_type == "tensor loss":
                            loss = tensor_loss
                        else:
                            raise ValueError(f"Unknown loss type: {loss_type}")

                        # Storing the losses
                        activation_loss_history.append(activation_loss.detach().cpu())
                        tensor_loss_history.append(tensor_loss.detach().cpu())
                        logger.info(f"Losses stored")

                        # Performing the optimization step
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        logger.info(f"Optimization step performed")

                    logger.info("Training completed\n\n")

                    logger.info("Calculating the time needed for training")
                    end_time = time.time()
                    # Calculating and storing the time elapsed
                    time_elapsed = end_time - start_time
                    time_string = (f"{tensor_wrapper_to_analyze.get_path()}: Time for training {factorization_label} using {loss_type}: {time_elapsed:.2f} "
                                   f"seconds + {init_time:.2f} seconds to initialize the matrix\n")
                    time_log.append(time_string)
                    verbose.print(time_string, Verbose.INFO)

                    final_activation_loss = activation_loss_history[-1].item()
                    final_tensor_loss = tensor_loss_history[-1].item()
                    verbose.print(
                        f"Final activation loss for {factorization_label} trained using {loss_type}: "
                        f"{final_activation_loss}",
                        Verbose.INFO
                    )
                    verbose.print(
                        f"Final tensor loss for {factorization_label} trained using {loss_type}: "
                        f"{final_tensor_loss}",
                        Verbose.INFO
                    )

                    logger.info("Computing the test activation loss")
                    # Computing the test activation loss
                    x_test = test_random_x.clone().detach()
                    for tensor in factorization:
                        x_test = torch.matmul(tensor, x_test)
                    test_activation_loss = (torch.norm((torch.matmul(tensor_to_analyze, test_random_x) - x_test), dim=0) ** 2).mean().item()
                    verbose.print(
                        f"Test activation loss for {factorization_label} trained using {loss_type}: "
                        f"{test_activation_loss}",
                        Verbose.INFO
                    )

                    loss_histories_factorizations["activation loss"][
                        f"{factorization_label} trained using {loss_type}"] = activation_loss_history
                    loss_histories_factorizations["tensor loss"][
                        f"{factorization_label} trained using {loss_type}"] = tensor_loss_history

                    logger.info("Appending details to data to be stored in CSV")
                    # Save details to CSV
                    csv_data.append({
                        "Original Tensor Path": tensor_wrapper_to_analyze.get_path(),
                        "Original Tensor Shape": tensor_wrapper_to_analyze.get_shape(),
                        "Factorization": factorization_label,
                        "Loss Type": loss_type,
                        "Initial Activation Loss": initial_activation_loss,
                        "Final Activation Loss": final_activation_loss,
                        "Initial Tensor Loss": initial_tensor_loss,
                        "Final Tensor Loss": final_tensor_loss,
                        "Test Activation Loss": test_activation_loss,
                        "Training Time": time_elapsed,
                        "Initialization Time": init_time
                    })

            logger.info("Appending the loss histories to the data to store")
            all_loss_histories[tensor_wrapper_to_analyze.get_path()] = loss_histories_factorizations

        logger.info(f"Trying to save the loss histories to file {file_path}")
        # Saving the loss histories to a file
        with open(file_path, "wb") as f:
            pkl.dump(all_loss_histories, f)
        logger.info(f"Loss histories saved to file {file_path}")

        logger.info(f"Trying to save the timing results to file {file_name_no_format}_training_times.txt")
        # Saving the timing results to a file
        time_log_path = os.path.join(directory_path, file_name_no_format + "_training_times.txt")
        with open(time_log_path, "w") as f:
            f.writelines(time_log)
        logger.info(f"Timing results saved to file {time_log_path}")

        logger.info(f"Trying to save the loss results to file {file_name_no_format}_losses_and_times.csv")
        # Saving the loss results to a CSV file
        csv_path = os.path.join(directory_path, file_name_no_format + f"_losses_and_times.csv")
        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = ["Original Tensor Path", "Original Tensor Shape", "Factorization", "Loss Type",
                          "Initial Activation Loss", "Final Activation Loss", "Initial Tensor Loss",
                          "Final Tensor Loss", "Test Activation Loss", "Training Time", "Initialization Time"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in csv_data:
                writer.writerow(row)
        logger.info(f"Loss results saved to file {csv_path}")

    logger.info(f"Starting to plot the results")
    x_range = configuration.get("x_range") if configuration.contains("x_range") else None
    bounds_string = f"with x bounded between {x_range[0]} and {x_range[1]}" if x_range is not None else ""
    y_range = configuration.get("y_range") if configuration.contains("y_range") else None
    bounds_string += " and " if x_range is not None and y_range is not None else ""
    bounds_string += f"with y bounded between {y_range[0]} and {y_range[1]}" if y_range is not None else ""

    for tensor_to_analyze_label, loss_histories_factorizations in all_loss_histories.items():
        logger.info(f"Plotting the results for the tensor: {tensor_to_analyze_label}")
        # Creating the figure to plot the results
        fig, axes = plt.subplots(2, 2, figsize=fig_size)
        logger.info("Figure created")
        fig.suptitle(f"Initialization analysis of the tensor: {tensor_to_analyze_label}")
        for label, activation_loss_history in loss_histories_factorizations["activation loss"].items():
            logger.info(f"Plotting the activation loss for {label}")
            axes[0, 0].plot(activation_loss_history, label=f"{label.replace('U, S, V^T trained', 'A, B SVD initialized trained').replace('A, B trained', 'A, B randomly initialized trained')}")
            axes[1, 0].plot(activation_loss_history, label=f"{label.replace('U, S, V^T trained', 'A, B SVD initialized trained').replace('A, B trained', 'A, B randomly initialized trained')}")
            if configuration.contains("y_range"):
                axes[1, 0].set_ylim(configuration.get("y_range"))
            if configuration.contains("x_range"):
                axes[1, 0].set_xlim(configuration.get("x_range"))
            print(f"Activation loss for {label}: {activation_loss_history[-1]}")

        axes[0, 0].set_title("Full activation training loss history (target activation - approximated activation)")
        axes[1, 0].set_title(f"Loss history {bounds_string} (target activation - approximated activation)")

        for label, tensor_loss_history in loss_histories_factorizations["tensor loss"].items():
            logger.info(f"Plotting the tensor loss for {label}")
            axes[0, 1].plot(tensor_loss_history, label=f"{label.replace('U, S, V^T trained', 'A, B SVD initialized trained').replace('A, B trained', 'A, B randomly initialized trained')}", )
            axes[1, 1].plot(tensor_loss_history, label=f"{label.replace('U, S, V^T trained', 'A, B SVD initialized trained').replace('A, B trained', 'A, B randomly initialized trained')}")
            if configuration.contains("y_range"):
                axes[1, 1].set_ylim(configuration.get("y_range"))
            if configuration.contains("x_range"):
                axes[1, 1].set_xlim(configuration.get("x_range"))
            print(f"Target tensor - approximated tensor for {label}: {tensor_loss_history[-1]}")

        axes[0, 1].set_title("Full loss history (target tensor - approximated tensor)")
        axes[1, 1].set_title(f"Loss history {bounds_string} (target tensor - approximated tensor)")

        for ax in axes.flatten():
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()

        logger.info(f"Trying to save the plot for the tensor: {tensor_to_analyze_label}")
        # Saving the plot
        plt.savefig(os.path.join(directory_path, file_name_no_format + f"_{tensor_to_analyze_label}_plot.png"))
        logger.info(f"Plot saved for the tensor: {tensor_to_analyze_label}")


# TODO FIX THIS FUNCTION
def perform_global_matrices_initialization_analysis(
        configuration: Config,
) -> None:
    """
    Compares which of the two initializations are better in terms of the quality loss and the speed of convergence to a
    good approximation for the global matrices' framework.

    Args:
        configuration (Config):
            The configuration object.
    """

    # Getting the parameters related to the paths from the configuration
    file_available = configuration.get("file_available")
    file_path = configuration.get("file_path")
    directory_path = configuration.get("directory_path")
    file_name = configuration.get("file_name")
    file_name_no_format = file_name.split(".")[0]

    # Getting the parameters related to the analysis from the configuration
    verbose = configuration.get_verbose()
    fig_size = configuration.get("figure_size") if configuration.contains("figure_size") else (20, 20)
    rank = configuration.get("rank")
    num_epochs = configuration.get("num_epochs") if configuration.contains("num_epochs") else 1000
    num_samples = configuration.get("num_samples") if configuration.contains("num_samples") else 1
    alpha = configuration.get("alpha") if configuration.contains("alpha") else 1
    device = get_available_device(configuration.get("device") if configuration.contains("device") else "cuda")

    if file_available:
        print(f"The file '{file_path}' is available.")
        # Loading the data from the file
        with open(file_path, "rb") as f:
            all_groups_tensor_factorizations_loss_histories, all_groups_tensor_factorizations = pkl.load(f)
    else:
        # Loading the model
        model = load_model_for_causal_lm(configuration)

        # Extracting the candidate tensors for the analysis
        extracted_tensor_wrappers = []
        extract_based_on_path(
            model,
            configuration.get("targets"),
            extracted_tensor_wrappers,
            configuration.get("black_list"),
            verbose=verbose
        )

        # Choosing the actual tensors to analyze
        tensor_wrappers_to_analyze = extracted_tensor_wrappers

        # Grouping the tensors
        unique_labels = set([tensor_wrapper.get_label() for tensor_wrapper in tensor_wrappers_to_analyze])
        verbose.print(f"The grouping labels of the extracted tensors are: {unique_labels}", Verbose.INFO)
        grouped_tensor_wrappers = [
            {"tensors": [], "label": unique_label} for unique_label in unique_labels
        ]
        for tensor_wrapper in tensor_wrappers_to_analyze:
            for group_index in range(len(grouped_tensor_wrappers)):
                if grouped_tensor_wrappers[group_index]["label"] == tensor_wrapper.get_label():
                    grouped_tensor_wrappers[group_index]["tensors"].append(tensor_wrapper)

        all_groups_tensor_factorizations_loss_histories = []
        all_groups_tensor_factorizations = []
        time_log = []
        csv_data = []
        for tensor_wrappers_group_dict in grouped_tensor_wrappers:
            tensor_wrappers_group = tensor_wrappers_group_dict["tensors"]
            tensor_wrappers_group_label = tensor_wrappers_group_dict["label"]
            verbose.print(f"Analyzing the tensors with label {tensor_wrappers_group_label}", Verbose.INFO)
            # Defining the shape of the analyzed matrices
            shape = tensor_wrappers_group[0].get_shape()
            # Checking the shapes of the tensors are all the same
            for tensor in tensor_wrappers_group:
                if tensor.get_shape() != shape:
                    raise ValueError("The tensors to analyze must have the same shape.")
            verbose.print(f"\nShape of the tensors to analyze: {shape}", Verbose.INFO)

            # Defining the tensor dictionaries to compare different initializations
            tensor_wrappers_group_random_init = AnalysisTensorDict(
                [tensor_wrappers_group_label] * len(tensor_wrappers_group), copy.deepcopy(tensor_wrappers_group)
            )
            tensor_wrappers_group_pseudo_inverse_init = AnalysisTensorDict(
                [tensor_wrappers_group_label] * len(tensor_wrappers_group), copy.deepcopy(tensor_wrappers_group)
            )
            tensor_wrappers_group_svd_init = AnalysisTensorDict(
                [tensor_wrappers_group_label] * len(tensor_wrappers_group), copy.deepcopy(tensor_wrappers_group)
            )
            tensor_wrappers_group_random_init.set_dtype(torch.float32)
            tensor_wrappers_group_pseudo_inverse_init.set_dtype(torch.float32)
            tensor_wrappers_group_svd_init.set_dtype(torch.float32)

            # Defining the global matrix to use in the analysis
            global_matrix = torch.randn(shape[0], rank).to(device)
            global_matrix.requires_grad = False

            # Initializing the factorizations using random initialization
            random_init_time = 0.0
            for tensor in tensor_wrappers_group_random_init.get_tensor_list(tensor_wrappers_group_label):
                tensor.set_attribute("factorization_type", "AB randomly initialized")
                tensor.set_attribute("global_matrix", global_matrix)
                a, b, random_init_time_one_matrix = get_global_matrix_factorization(
                    tensor.get_tensor(),
                    global_matrix,
                    rank,
                    True,
                    "random",
                    device
                )
                random_init_time += random_init_time_one_matrix
                tensor.set_attribute("factorization", [b, a])

            # Initializing the factorizations using pseudo-inverse initialization
            pseudo_inverse_init_time = 0.0
            for tensor in tensor_wrappers_group_pseudo_inverse_init.get_tensor_list(tensor_wrappers_group_label):
                tensor.set_attribute("factorization_type", "AB pseudo-inverse initialized")
                tensor.set_attribute("global_matrix", global_matrix)
                a, b, random_init_time_one_matrix = get_global_matrix_factorization(
                    tensor.get_tensor(),
                    global_matrix,
                    rank,
                    True,
                    "pseudo-inverse",
                    device
                )
                pseudo_inverse_init_time += random_init_time_one_matrix
                tensor.set_attribute("factorization", [b, a])

            # Initializing the factorizations using SVD initialization
            svd_init_time = 0.0
            for tensor in tensor_wrappers_group_svd_init.get_tensor_list(tensor_wrappers_group_label):
                tensor.set_attribute("factorization_type", "AB pseudo-inverse initialized")
                tensor.set_attribute("global_matrix", global_matrix)
                us, vt, svd_init_time_one_matrix = get_svd_factorization(
                    tensor.get_tensor(),
                    rank,
                    [True, True],
                    device
                )
                svd_init_time += svd_init_time_one_matrix
                tensor.set_attribute("factorization", [vt, us])

            verbose.print(
                f"\nInitialization times:\n\tAB random initialization: {random_init_time:.2f} seconds,\n\tAB "
                f"pseudo-inverse initialization: {pseudo_inverse_init_time:.2f} seconds,\n\tSVD initialization: "
                f"{svd_init_time:.2f} seconds\n",
                Verbose.INFO
            )

            tensor_factorizations_dict = {
                "AB randomly initialized": [tensor_wrappers_group_random_init, random_init_time],
                "AB pseudo-inverse initialized": [tensor_wrappers_group_pseudo_inverse_init, pseudo_inverse_init_time],
                "SVD initialized": [tensor_wrappers_group_svd_init, svd_init_time]
            }
            tensor_factorizations_losses = ["tensor loss", "tensor loss", "penalized tensor loss"]

            tensor_factorizations_loss_histories = {"activation loss": {}, "tensor loss": {}, "penalization term": {}}

            for tensor_factorization_index, tensor_factorization_key_value in enumerate(tensor_factorizations_dict.items()):
                tensor_factorization_key, tensor_factorization_value = tensor_factorization_key_value
                factorization_label = tensor_factorization_key
                tensors_to_analyze_dict = tensor_factorization_value[0]
                tensor_init_time = tensor_factorization_value[1]

                # Preparing the data
                wrapper_tensors_to_analyze = tensors_to_analyze_dict.get_tensor_list(tensors_to_analyze_dict.get_keys()[0])
                out_shape, in_shape = wrapper_tensors_to_analyze[0].get_shape()
                random_x = torch.randn(in_shape, num_samples).to(device)
                test_random_x = torch.randn(in_shape, num_samples).to(device)
                tensors_to_analyze = [
                    wrapper_tensor.get_tensor().to(device) for wrapper_tensor in wrapper_tensors_to_analyze
                ]
                tensorsx = [torch.matmul(tensor, random_x) for tensor in tensors_to_analyze]

                # Setting the optimizer
                trainable_tensors = [
                    tensor
                    for tensor_wrapper in wrapper_tensors_to_analyze
                    for tensor in tensor_wrapper.get_attribute("factorization") if tensor.requires_grad
                ]
                optimizer = torch.optim.AdamW(
                    trainable_tensors,
                    lr=configuration.get("learning_rate") if configuration.contains("learning_rate") else 1e-4,
                    eps=1e-7 if trainable_tensors[0].dtype == torch.float16 else 1e-8
                )

                activation_loss_history = []
                tensor_loss_history = []
                penalization_term_history = []
                initial_activation_loss = None
                initial_tensor_loss = None
                initial_penalization_term = None

                verbose.print(f"Starting training using tensor loss for {factorization_label}", Verbose.INFO)

                # Storing the start time
                start_time = time.time()
                for _ in tqdm(range(num_epochs)):
                    total_activation_loss = torch.Tensor([0.0]).to(device)
                    total_tensor_loss = torch.Tensor([0.0]).to(device)

                    for index, tensor_wrapper in enumerate(wrapper_tensors_to_analyze):
                        x = random_x.clone().detach().to(device)
                        y = torch.eye(in_shape).to(device).to(device)
                        for factorization_term in tensor_wrapper.get_attribute("factorization"):
                            x = torch.matmul(factorization_term, x)
                            y = torch.matmul(factorization_term, y)

                        total_activation_loss += (torch.norm((tensorsx[index] - x), dim=0) ** 2).mean()
                        total_tensor_loss += torch.norm(tensors_to_analyze[index] - y) ** 2

                    if initial_activation_loss is None:
                        initial_activation_loss = total_activation_loss.item()
                    if initial_tensor_loss is None:
                        initial_tensor_loss = total_tensor_loss.item()

                    activation_loss_history.append(total_activation_loss.detach().cpu())
                    tensor_loss_history.append(total_tensor_loss.detach().cpu())

                    if tensor_factorizations_losses[tensor_factorization_index] == "activation loss":
                        loss = total_activation_loss
                    elif tensor_factorizations_losses[tensor_factorization_index] == "tensor loss":
                        loss = total_tensor_loss
                    elif tensor_factorizations_losses[tensor_factorization_index] == "penalized tensor loss":
                        penalization_term = torch.Tensor([0.0]).to(device)
                        for index_1, tensor_wrapper_1 in enumerate(wrapper_tensors_to_analyze):
                            for index_2, tensor_wrapper_2 in enumerate(wrapper_tensors_to_analyze):
                                if index_2 > index_1:
                                    penalization_term += (
                                            alpha *
                                            torch.norm(
                                                (
                                                    tensor_wrapper_1.get_attribute("factorization")[1].to(device) -
                                                    tensor_wrapper_2.get_attribute("factorization")[1].to(device)
                                                ) ** 2
                                            )
                                    )
                        loss = total_tensor_loss + penalization_term

                        if initial_penalization_term is None:
                            initial_penalization_term = penalization_term

                        penalization_term_history.append(penalization_term.detach().cpu())
                    else:
                        raise ValueError(f"Unknown loss type: {tensor_factorizations_losses[tensor_factorization_index]}")

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Storing the end time
                end_time = time.time()
                # Calculating and storing the time elapsed
                time_elapsed = end_time - start_time

                time_string = (f"{tensor_wrappers_group_label}: Time for training {factorization_label} using tensor loss: {time_elapsed:.2f} seconds + "
                               f"{tensor_init_time:.2f} seconds to initialize the factorization\n")
                time_log.append(time_string)

                verbose.print(time_string, Verbose.INFO)

                final_activation_loss = activation_loss_history[-1].item()
                final_tensor_loss = tensor_loss_history[-1].item()

                # Computing the test activation loss
                total_test_activation_loss = torch.Tensor([0.0]).to(device)
                for index, tensor_wrapper in enumerate(wrapper_tensors_to_analyze):
                    x_test = random_x.clone().detach().to(device)
                    for factorization_term in tensor_wrapper.get_attribute("factorization"):
                        x_test = torch.matmul(factorization_term.to(device), x_test)

                    total_test_activation_loss += (
                            torch.norm(
                                (torch.matmul(tensor_wrapper.get_tensor().to(device), test_random_x) - x_test), dim=0
                            ) ** 2
                    ).mean().item()

                tensor_factorizations_loss_histories["activation loss"][
                    f"{factorization_label} trained using tensor loss"] = activation_loss_history
                tensor_factorizations_loss_histories["tensor loss"][
                    f"{factorization_label} trained using tensor loss"] = tensor_loss_history
                if tensor_factorizations_losses[tensor_factorization_index] == "penalized tensor loss":
                    tensor_factorizations_loss_histories["penalization term"][
                        f"{factorization_label} trained using tensor loss"] = penalization_term_history

                # Saving details to CSV
                csv_data.append({
                    "Tensors Label": tensor_wrappers_group_label,
                    "Factorization": factorization_label,
                    "Initial Activation Loss": initial_activation_loss,
                    "Final Activation Loss": final_activation_loss,
                    "Initial Tensor Loss": initial_tensor_loss,
                    "Final Tensor Loss": final_tensor_loss,
                    "Test Activation Loss": total_test_activation_loss.item(),
                    "Initial Penalization Term": initial_penalization_term,
                    "Final Penalization Term": penalization_term_history[-1] if tensor_factorizations_losses[tensor_factorization_index] == "penalized tensor loss" else None,
                    "Training Time": time_elapsed,
                    "Initialization Time": tensor_init_time
                })

            # Saving the timing results to a file
            time_log_path = os.path.join(directory_path, file_name_no_format + "_training_times.txt")
            with open(time_log_path, "w") as f:
                f.writelines(time_log)

            # Saving the loss results to a CSV file
            csv_path = os.path.join(directory_path, file_name_no_format + "_losses_and_times.csv")
            with open(csv_path, "w", newline="") as csvfile:
                fieldnames = ["Tensors Label", "Factorization", "Initial Activation Loss", "Final Activation Loss", "Initial Tensor Loss",
                              "Final Tensor Loss", "Test Activation Loss", "Initial Penalization Term",
                              "Final Penalization Term", "Training Time", "Initialization Time"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for row in csv_data:
                    writer.writerow(row)

            all_groups_tensor_factorizations_loss_histories.append(
                {"label": tensor_wrappers_group_label, "loss_history": tensor_factorizations_loss_histories}
            )

        # Saving the loss histories to a file
        with open(file_path, "wb") as f:
            pkl.dump((all_groups_tensor_factorizations_loss_histories, grouped_tensor_wrappers), f)

    for tensor_factorizations_loss_histories_group_dict in all_groups_tensor_factorizations_loss_histories:
        tensor_factorizations_loss_histories = tensor_factorizations_loss_histories_group_dict["loss_history"]
        tensor_factorizations_loss_histories_label = tensor_factorizations_loss_histories_group_dict["label"]
        # Creating the figure to plot the results
        fig, axes = plt.subplots(2, 3, figsize=fig_size)
        fig.suptitle(f"Loss histories af the global factorizations of the matrices with label {tensor_factorizations_loss_histories_label}")
        x_range = configuration.get("x_range") if configuration.contains("x_range") else None
        bounds_string = f"with x bounded between {x_range[0]} and {x_range[1]}" if x_range is not None else ""
        y_range = configuration.get("y_range") if configuration.contains("y_range") else None
        bounds_string += " and " if x_range is not None and y_range is not None else ""
        bounds_string += f"with y bounded between {y_range[0]} and {y_range[1]} " if y_range is not None else " "

        for label, activation_loss_history in tensor_factorizations_loss_histories["activation loss"].items():
            axes[0, 0].plot(activation_loss_history, label=f"{label}")
            axes[1, 0].plot(activation_loss_history, label=f"{label}")
            if configuration.contains("y_range"):
                axes[1, 0].set_ylim(configuration.get("y_range"))
            if configuration.contains("x_range"):
                axes[1, 0].set_xlim(configuration.get("x_range"))
            print(f"Activation loss for {label}: {activation_loss_history[-1]}")

        axes[0, 0].set_title("Full activation training loss history (target activation - approximated activation)")
        axes[1, 0].set_title(f"Loss history {bounds_string}(target activation - approximated activation)")

        for label, tensor_loss_history in tensor_factorizations_loss_histories["tensor loss"].items():
            axes[0, 1].plot(tensor_loss_history, label=f"{label}", )
            axes[1, 1].plot(tensor_loss_history, label=f"{label}")
            if configuration.contains("y_range"):
                axes[1, 1].set_ylim(configuration.get("y_range"))
            if configuration.contains("x_range"):
                axes[1, 1].set_xlim(configuration.get("x_range"))
            print(f"Target tensor - approximated tensor for {label}: {tensor_loss_history[-1]}")

        axes[0, 1].set_title("Full loss history (target tensor - approximated tensor)")
        axes[1, 1].set_title(f"Loss history {bounds_string} (target tensor - approximated tensor)")

        x_range_penalization = configuration.get("x_range_penalization") if configuration.contains("x_range_penalization") else None
        bounds_string = f"with x bounded between {x_range_penalization[0]} and {x_range_penalization[1]}" if x_range_penalization is not None else ""
        y_range_penalization = configuration.get("y_range_penalization") if configuration.contains("y_range_penalization") else None
        bounds_string += " and " if y_range_penalization is not None and y_range_penalization is not None else ""
        bounds_string += f"with y bounded between {y_range_penalization[0]} and {y_range_penalization[1]} " if y_range_penalization is not None else " "

        for label, penalization_term_history in tensor_factorizations_loss_histories["penalization term"].items():
            axes[0, 2].plot(penalization_term_history, label=f"{label}")
            axes[1, 2].plot(penalization_term_history, label=f"{label}")
            if configuration.contains("y_range_penalization"):
                axes[1, 2].set_ylim(configuration.get("y_range_penalization"))
            if configuration.contains("x_range_penalization"):
                axes[1, 2].set_xlim(configuration.get("x_range_penalization"))
            print(f"Penalization term for {label}: {penalization_term_history[-1]}")

        axes[0, 2].set_title("Full penalization term history (sum of difference between matrices)")
        axes[1, 2].set_title(f"Penalization term history {bounds_string}(sum of difference between matrices)")

        for ax in axes.flatten():
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()

        # Saving the plot
        plt.savefig(os.path.join(directory_path, file_name_no_format + f"_{tensor_factorizations_loss_histories_label}_plot.png"))
