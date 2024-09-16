import os
import sys

from exporch import Config, get_path_to_configurations, Verbose, check_path_to_storage

from redhunter.analysis.original_layers_rank_analysis import perform_original_layers_rank_analysis
from redhunter.analysis.delta_layers_rank_analysis import (
    perform_delta_consecutive_layers_rank_analysis,
    perform_all_delta_layers_rank_analysis,
    perform_delta_layers_wrt_average_rank_analysis,
)
from redhunter.analysis.sorted_layers_rank_analysis import perform_sorted_layers_rank_analysis


def main():
    """
    Main method to start the layers rank analysis
    """

    if len(sys.argv) < 3:
        raise Exception("Please provide the name of the configuration file and the environment.\n"
                        "Example: python rank_analysis_launcher.py config_name environment"
                        "'environment' can be 'local' or 'server' or 'colab'.")

    # Extracting the configuration name and the environment
    config_name = sys.argv[1]
    environment = sys.argv[2]

    # Loading the configuration
    configuration = Config(
        os.path.join(get_path_to_configurations(environment), "rank_analysis", config_name)
    )
    verbose = Verbose(configuration.get("verbose") if configuration.contains("verbose") else 0)

    # Checking if the configuration file contains the necessary keys
    mandatory_keys = [
        "path_to_storage",
        "analysis_type",
        "targets",
        "original_model_id",
        "explained_variance_threshold"
    ]
    for key in mandatory_keys:
        if not configuration.contains(key):
            raise Exception(f"The configuration file must contain the key '{key}'.")

    # Checking if the path to the storage exists
    path_to_storage = configuration.get("path_to_storage")
    analysis_type = configuration.get("analysis_type")
    model_name = configuration.get("original_model_id").split("/")[-1]
    words_to_be_in_the_file_name = (
                                       ["paths"] + configuration.get("targets") +
                                       ["black_list"] + configuration.get("black_list")
                                       if configuration.contains("black_list") else "None"
    )
    if configuration.get("analysis_type") == "sorted_layers_rank_analysis":
        words_to_be_in_the_file_name = (words_to_be_in_the_file_name +
                                        ["guide"] + [configuration.get("similarity_guide_name")])
    file_available, directory_path, file_name = check_path_to_storage(
        path_to_storage,
        analysis_type,
        model_name,
        tuple(words_to_be_in_the_file_name)
    )
    file_path = os.path.join(directory_path, file_name)
    configuration.update(
        {
            "file_available": file_available,
            "file_path": file_path,
            "directory_path": directory_path,
            "file_name": file_name
        }
    )

    print(f"{'File to load data available' if file_available else 'No file to load data'}")
    print(f"File path: {file_path}")

    # Performing the rank analysis
    if configuration.get("analysis_type") == "original_layers_rank_analysis":
        # Perform the rank analysis of the original layers of a model
        perform_original_layers_rank_analysis(configuration, verbose)
    elif configuration.get("analysis_type") == "delta_consecutive_layers_rank_analysis":
        # Perform the rank analysis of the delta consecutive layers of a model
        perform_delta_consecutive_layers_rank_analysis(configuration, verbose)
    elif configuration.get("analysis_type") == "delta_layers_wrt_average_rank_analysis":
        # Perform the rank analysis of the delta layers with respect to the average matrix of some layers
        perform_delta_layers_wrt_average_rank_analysis(configuration, verbose)
    elif configuration.get("analysis_type") == "all_delta_layers_rank_analysis":
        # Perform the rank analysis of the delta layers with respect to all the other layers
        perform_all_delta_layers_rank_analysis(configuration, verbose)
    elif configuration.get("analysis_type") == "sorted_layers_rank_analysis":
        # Perform the rank analysis of the sorted layers of a model
        perform_sorted_layers_rank_analysis(configuration, verbose)
    else:
        raise Exception(f"Analysis type '{analysis_type}' not recognized.")


if __name__ == "__main__":
    main()
