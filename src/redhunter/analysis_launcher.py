import sys

from exporch import GeneralPurposeExperimentFactory

from redhunter.analysis.basic_model_analysis import ModelBasicAnalysis
from redhunter.analysis.layer_replacement_analysis import (
    SingleNullLayersReplacementAnalysis,

    AllLayerCouplesReplacementAnalysis,
    AllLayerCouplesDisplacementBasedReplacementAnalysis,
    SpecificDisplacementLayerReplacementAnalysis,
    SubsequentLayerReplacementAnalysis,
    PreviousLayerReplacementAnalysis,
    SpecificReplacedLayerReplacementAnalysis,
    SpecificReplacingLayerReplacementAnalysis,
    SameLayerCouplesReplacementAnalysis,
    AllLayersReplacementAnalysis,

    PerplexityLayerReplacementAnalysis
)
from redhunter.analysis.rank_analysis import (
    OriginalLayersRankAnalysis,
    ConcatenatedLayersRankAnalysis,

    DeltaConsecutiveLayersRankAnalysis,
    DeltaLayersWRTAverageLayerRankAnalysis,
    AllDeltaLayersRankAnalysis
)
from redhunter.analysis.sorted_layers_compression_analysis import (
    ResettableElementsSortedLayersCompressionAnalysisWithConcatenatedMatrices
)


GeneralPurposeExperimentFactory.register({
    "model_basic_analysis": ModelBasicAnalysis,

    "original_layer_rank_analysis": OriginalLayersRankAnalysis,
    "concatenated_layer_rank_analysis": ConcatenatedLayersRankAnalysis,
    "delta_consecutive_layers_rank_analysis": DeltaConsecutiveLayersRankAnalysis,
    "delta_layers_wrt_average_layer_rank_analysis": DeltaLayersWRTAverageLayerRankAnalysis,
    "all_delta_layers_rank_analysis": AllDeltaLayersRankAnalysis,

    "single_null_layers_replacement_redundancy_analysis": SingleNullLayersReplacementAnalysis,

    "all_layer_couples_replacement_redundancy_analysis": AllLayerCouplesReplacementAnalysis,
    "all_layer_couples_displacement_based_replacement_redundancy_analysis": AllLayerCouplesDisplacementBasedReplacementAnalysis,
    "specific_displacement_layer_replacement_redundancy_analysis": SpecificDisplacementLayerReplacementAnalysis,
    "subsequent_layer_replacement_redundancy_analysis": SubsequentLayerReplacementAnalysis,
    "previous_layer_replacement_redundancy_analysis": PreviousLayerReplacementAnalysis,
    "specific_replaced_layer_replacement_redundancy_analysis": SpecificReplacedLayerReplacementAnalysis,
    "specific_replacing_layer_replacement_redundancy_analysis": SpecificReplacingLayerReplacementAnalysis,
    "same_layer_couples_replacement_redundancy_analysis": SameLayerCouplesReplacementAnalysis,
    "all_layers_replacement_redundancy_analysis": AllLayersReplacementAnalysis,

    "perplexity_layer_replacement_redundancy_analysis": PerplexityLayerReplacementAnalysis,

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

    # Creating and launching the analysis
    analysis_experiment = GeneralPurposeExperimentFactory.create(f"src/experiments/configurations/{config_file_name}")
    analysis_experiment.launch_experiment()


if __name__ == "__main__":
    main()