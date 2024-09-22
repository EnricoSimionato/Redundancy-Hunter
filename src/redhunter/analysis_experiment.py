from abc import ABC, abstractmethod
from typing import Any, override

from exporch import GeneralPurposeExperiment, Config


class AnalysisExperiment(GeneralPurposeExperiment, ABC):
    """
    The base class for the analysis experiments.
    """

    mandatory_keys = [
        "targets"
    ]


    @override
    def _run_experiment(
            self
    ) -> None:
        """
        Runs the experiment conducting the specific analysis.
        It will perform the operations that are defined in the specific subclass of AnalysisExperiment.
        """

        if self.config.contains("just_plot") and self.config.get("just_plot") and self.data is None:
            raise ValueError("It is not possible to plot the results without performing the analysis. Set 'just_plot' "
                             "to False.")

        if not self.config.contains("just_plot") or not self.config.get("just_plot"):
            self._perform_analysis()
            self._postprocess_results()
        self._plot_results(self.config, self.data)

    @abstractmethod
    def _perform_analysis(
            self
    ) -> None:
        """
        Performs the analysis.
        The performed analysis will depend on the specific subclass of AnalysisExperiment.
        """

        pass

    def _postprocess_results(
            self
    ) -> None:
        """
        Post-processes the results obtained from the analysis.
        The performed analysis will depend on the specific subclass of Analysis
        """

        pass

    @abstractmethod
    def _plot_results(
            self,
            config: Config,
            data: Any
    ) -> None:
        """
        Plots the results obtained from the analysis.
        The performed analysis will depend on the specific subclass of AnalysisExperiment.

        Args:
            config (Config):
                The configuration of the experiment.
            data (Any):
                The data obtained from the analysis.
        """

        pass
