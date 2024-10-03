from abc import ABC

from exporch import GeneralPurposeExperiment


class AnalysisExperiment(GeneralPurposeExperiment, ABC):
    """
    The base class for the analysis experiments.
    """

    mandatory_keys = [
        "targets"
    ]