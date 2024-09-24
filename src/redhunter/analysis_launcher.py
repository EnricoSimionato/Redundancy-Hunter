import sys

from exporch import GeneralPurposeExperimentFactory

from redhunter.analysis.layer_replacement_analysis import (
    AllLayerCouplesReplacementAnalysis,
    SubsequentLayerReplacementAnalysis,
    PreviousLayerReplacementAnalysis,
    SpecificReplacedLayerReplacementAnalysis,
    SpecificReplacingLayerReplacementAnalysis,
    SameLayerCouplesReplacementAnalysis,
    AllLayersReplacementAnalysis

)
from redhunter.analysis.sorted_layers_compression_analysis import (
    ResettableElementsSortedLayersCompressionAnalysisWithConcatenatedMatrices
)


GeneralPurposeExperimentFactory.register({
    "all_layer_couples_replacement_redundancy_analysis": AllLayerCouplesReplacementAnalysis,
    "subsequent_layer_replacement_redundancy_analysis": SubsequentLayerReplacementAnalysis,
    "previous_layer_replacement_redundancy_analysis": PreviousLayerReplacementAnalysis,
    "specific_replaced_layer_replacement_redundancy_analysis": SpecificReplacedLayerReplacementAnalysis,
    "specific_replacing_layer_replacement_redundancy_analysis": SpecificReplacingLayerReplacementAnalysis,
    "same_layer_couples_replacement_redundancy_analysis": SameLayerCouplesReplacementAnalysis,
    "all_layers_replacement_redundancy_analysis": AllLayersReplacementAnalysis,

    "resettable_elements_sorted_layers_compression_analysis_with_concatenated_matrices": ResettableElementsSortedLayersCompressionAnalysisWithConcatenatedMatrices
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