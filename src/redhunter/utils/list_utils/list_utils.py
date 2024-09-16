def is_subsequence(
        subsequence: list,
        sequence: list
) -> bool:
    """
    Checks if a sequence is a subsequence of another sequence.

    Args:
        subsequence (list):
            The subsequence.
        sequence (list):
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
