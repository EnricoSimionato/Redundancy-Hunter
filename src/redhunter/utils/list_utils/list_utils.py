def is_subsequence(
        subsequence: list | tuple,
        sequence: list | tuple
) -> bool:
    """
    Checks if a sequence is a subsequence of another sequence.

    Args:
        subsequence (list | tuple):
            The subsequence.
        sequence (list | tuple):
            The sequence.

    Returns:
        bool:
            True if the subsequence is a subsequence of the sequence, False otherwise.
    """

    i = 0
    for element in sequence:
        if element == subsequence[i]:
            i += 1
        if i == len(subsequence):
            return True
    return False
