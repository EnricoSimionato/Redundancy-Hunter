from typing import Any, override

import transformers

from exporch import Config
from exporch.utils.causal_language_modeling import load_model_for_causal_lm

from redhunter.analysis.analysis_utils import extract, AnalysisTensorDict
from redhunter.analysis_experiment import AnalysisExperiment


class ModelBasicAnalysis(AnalysisExperiment):
    """
    The class for the model basic analysis.
    """

    @override
    def _run_experiment(
            self
    ) -> None:
        """
        Runs the experiment.
        """

        # Loading the model
        model = load_model_for_causal_lm(config=self.get_config())
        # Analyzing the model
        self.analyze_model(model, self.get_config().get("targets", ()), self.get_config().get("black_list", ()))

    def analyze_model(
            self,
            model: transformers.PreTrainedModel | transformers.AutoModel,
            target_layers: list = (),
            black_list: list = ()
    ) -> None:
        """
        Analyzes a model.

        Args:
            model (transformers.PreTrainedModel):
                The model.
            target_layers (list, optional):
                The paths of the layers to analyze. Defaults to ().
            black_list (list, optional):
                The black list. Defaults to ().
        """

        self.log(f"Model loaded: \n\n{model}", print_message=True)

        extracted_layers = AnalysisTensorDict()
        extract(model, target_layers, extracted_layers, black_list=black_list)

        for key in extracted_layers.get_keys():
            layers = extracted_layers.get_tensor_list(key)
            for layer in layers:
                self.log("Printing the name of the layer:", print_message=True)
                self.log(f"    {layer.get_name()}", print_message=True)
                self.log("Printing the path of the layer:", print_message=True)
                self.log(f"    {layer.get_path()}", print_message=True)
                self.log("Printing the layer:", print_message=True)
                self.log(f"    {layer.get_layer()}", print_message=True)
                self.log("Printing the shape of the layer's weight", print_message=True)

                self.log("------------------------------------------------------------------------------------------")

    def _postprocess_results(
            self
    ) -> None:
        """
        Postprocess the results.
        It does nothing.
        """

        pass

    def _plot_results(
            self,
            config: Config,
            data: Any
    ) -> None:
        """
        Plots the results.
        It does nothing.
        """

        pass