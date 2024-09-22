import sys

from exporch import GeneralPurposeExperimentFactory

from redhunter.analysis.layer_replacement_analysis import (
    AllLayerCouplesReplacementAnalysis,
    SameLayerCouplesReplacementAnalysis,
    SubsequentLayerReplacementAnalysis,
    AllLayersReplacementAnalysis
)


"""
specific_mandatory_keys_mapping = {
    "simple_initialization_analysis": ["rank"],
    "global_matrices_initialization_analysis": ["rank"],

    "sorted_layers_rank_analysis": ["explained_variance_threshold"],

    "head_analysis": ["explained_variance_threshold", "name_num_heads_mapping"],
    "heads_similarity_analysis": ["grouping"],

    "activations_analysis": ["dataset_id"],
    "delta_activations_analysis": ["dataset_id"],

    "query_key_analysis": ["query_label", "key_label"],

    "swapped_layers_redundancy_analysis": ["benchmark_ids", "num_layers"]
}

not_used_keys_mapping = {
    "query_key_analysis": ["targets", "black_list"],
}
"""

GeneralPurposeExperimentFactory.register({
    "all_layer_couples_replacement_redundancy_analysis": AllLayerCouplesReplacementAnalysis,
    "same_layer_couples_replacement_redundancy_analysis": SameLayerCouplesReplacementAnalysis,
    "all_layers_replacement_redundancy_analysis": AllLayersReplacementAnalysis,
    "subsequent_layer_replacement_redundancy_analysis": SubsequentLayerReplacementAnalysis,
})


def main() -> None:
    """
    Main method to start the various types of analyses on a deep model.
    """

    if len(sys.argv) < 2:
        raise Exception("Please provide the name of the configuration file.\n"
                        "Example: python src/redhunter/analysis_launcher.py config_name")

    # Extracting the configuration name and the environment
    config_file_name = sys.argv[1]

    analysis_experiment = GeneralPurposeExperimentFactory.create(f"src/experiments/configurations/{config_file_name}")
    analysis_experiment.launch_experiment()


if __name__ == "__main__":
    main()