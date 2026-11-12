import statistics


def mean(numbers):
    """Return the mean of numbers (0.0 for an empty list)."""
    if not numbers:
        return 0.0
    return statistics.mean(numbers)


def median(numbers):
    """Return the median of numbers (0.0 for an empty list)."""
    if not numbers:
        return 0.0
    ordered = sorted(numbers)
    half = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[half]
    return (ordered[half - 1] + ordered[half]) / 2


def count_above(numbers, threshold):
    """Count the values strictly greater than threshold."""
    return sum(1 for value in numbers if value > threshold)
