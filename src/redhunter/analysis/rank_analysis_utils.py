import numpy as np

from exporch import Verbose


def compute_singular_values(
        matrix: np.ndarray
) -> np.ndarray:
    """
    Computes the singular values of a matrix.

    Args:
        matrix (np.ndarray):
            The matrix to compute the singular values of.

    Returns:
        np.ndarray:
            The singular values of the matrix.
    """

    return np.linalg.svd(matrix, compute_uv=False)


def compute_explained_variance(
    s: np.array,
    scaling: int = 1
) -> np.array:
    """
    Computes the explained variance for a set of singular values.

    Args:
        s (np.array):
            The singular values.
        scaling (float, optional):
            Scaling to apply to the explained variance at each singular value.
            Defaults to 1.

    Returns:
        np.array:
            The explained variance for each singular value.
    """

    if s[0] == 0.:
        return np.ones(len(s))

    return (np.square(s) * scaling).cumsum() / (np.square(s) * scaling).sum()


class RankAnalysisResult:
    """
    Class to store the result of the rank analysis. It stores the rank of the tensor and the thresholds used to compute
    the rank.

    Args:
        rank (int):
            The rank of the tensor.
        explained_variance_threshold (float, optional):
            The threshold on the explained variance to use to compute the rank. Rank is computed as the number of
            singular values that explain the threshold fraction of the total variance. Defaults to 0.
        singular_values_threshold (float, optional):
            The threshold to use to compute the rank based on singular values. Rank is computed as the number of
            singular values that are greater than the threshold. Defaults to 0.
        verbose (Verbose, optional):
            The verbosity level. Defaults to Verbose.INFO.

    Attributes:
        rank (int):
            The rank of the tensor.
        explained_variance_threshold (float):
            The threshold on the explained variance to use to compute the rank. Rank is computed as the number of
            singular values that explain the threshold fraction of the total variance.
        singular_values_threshold (float):
            The threshold to use to compute the rank based on singular values. Rank is computed as the number of
            singular values that are greater than the threshold.
        verbose (Verbose):
            The verbosity level.
    """

    def __init__(
            self,
            rank: int,
            explained_variance_threshold: float = 0,
            singular_values_threshold: float = 0,
            verbose: Verbose = Verbose.INFO
    ) -> None:

        self.rank = rank
        self.explained_variance_threshold = explained_variance_threshold
        self.singular_values_threshold = singular_values_threshold

        self.verbose = verbose

    def get_rank(
            self
    ) -> int:
        """
        Returns the rank of the tensor.

        Returns:
            int:
                The rank of the tensor.
        """

        return self.rank

    def get_explained_variance_threshold(
            self
    ) -> float:
        """
        Returns the threshold on the explained variance to use to compute the rank.

        Returns:
            float:
                The threshold on the explained variance to use to compute the rank.
        """

        return self.explained_variance_threshold

    def get_singular_values_threshold(
            self
    ) -> float:
        """
        Returns the threshold to use to compute the rank based on singular values.

        Returns:
            float:
                The threshold to use to compute the rank based on singular values.
        """

        return self.singular_values_threshold