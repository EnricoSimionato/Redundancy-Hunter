import sys

import transformers

from exporch import Verbose

from exporch.utils.causal_language_modeling import load_model_for_causal_lm

from redhunter.analysis.analysis_utils import extract_based_on_path


def analyze_layers(
        model: transformers.PreTrainedModel,
        paths_layers_to_analyze: list = (),
        black_list: list = (),
        verbose: Verbose = Verbose.INFO
) -> None:
    """
    Analyze the layers of a model.

    Args:
        model (transformers.PreTrainedModel):
            The model.
        paths_layers_to_analyze (list, optional):
            The paths of the layers to analyze. Defaults to ().
        black_list (list, optional):
            The black list. Defaults to ().
        verbose (Verbose, optional):
            The verbosity level. Defaults to Verbose.INFO.
    """

    extracted_layers = []
    extract_based_on_path(
        model,
        paths_layers_to_analyze,
        extracted_layers,
        black_list=black_list,
        verbose=verbose
    )
    seen_labels = set()

    for layer in extracted_layers:
        if layer["label"] in seen_labels:
            continue
        print("Printing the name of the layer:")
        print(f"    {layer['name']}")
        print("Printing the path of the layer:")
        print(f"    {layer['path']}")
        print("Printing the layer:")
        print(f"    {layer['layer']}")
        print("Printing the shape of the layer's weight")
        print(f"    {layer['weight'].shape}")
        # print("Printing the layer's weight")
        # print(f"  {layer['weight']}")
        print("##################################################")

        seen_labels.add(layer["label"])


def analyze_model(
        model_id: str,
        dtype: str = "float32",
        quantization: str = None,
        paths_layers_to_analyze: list = (),
        black_list: list = (),
        verbose: Verbose = Verbose.INFO
) -> None:
    """
    Analyzes a model.

    Args:
        model_id (str):
            The model ID.
        dtype (str, optional):
            The data type. Defaults to "float32".
        quantization (str, optional):
            The quantization. Defaults to "4bit".
        paths_layers_to_analyze (list, optional):
            The paths of the layers to analyze. Defaults to ().
        black_list (list, optional):
            The black list. Defaults to ().
        verbose (Verbose, optional):
            The verbosity level. Defaults to Verbose.INFO.
    """

    model = load_model_for_causal_lm(model_id, dtype, quantization)

    print(model)

    analyze_layers(
        model,
        paths_layers_to_analyze,
        black_list=black_list,
        verbose=verbose
    )


def main():
    """
    Main function to analyze a model.
    """

    if len(sys.argv) < 3:
        print(
            "Usage: python3 basic_model_analysis.py <model_id> --targets <path_to_layer_1> <path_to_layer_2> ... --black_list <black_listed_string_1> <black_listed_string_2> ...")
        sys.exit(1)
    if "--targets" not in sys.argv:
        print(
            "Usage: python3 basic_model_analysis.py <model_id> --targets <path_to_layer_1> <path_to_layer_2> ... --black_list <black_listed_string_1> <black_listed_string_2> ...")

    if "--black_list" not in sys.argv:
        print("Warning: No black list provided.")
        end_targets = len(sys.argv)
    else:
        end_targets = sys.argv.index("--black_list")

    model_to_analyze = sys.argv[1]
    paths_layers_to_analyze = sys.argv[sys.argv.index("--targets"):end_targets]
    if "--black_list" not in sys.argv:
        black_list = []
    else:
        black_list = sys.argv[end_targets + 1:]

    analyze_model(
        model_to_analyze,
        paths_layers_to_analyze=paths_layers_to_analyze,
        black_list=black_list,
        verbose=Verbose.INFO
    )


if __name__ == "__main__":
    main()
