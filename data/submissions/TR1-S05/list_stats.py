def mean(numbers):
    """Return the arithmetic mean, or 0.0 for no numbers."""
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


def median(numbers):
    """Return the median value, or 0.0 for no numbers."""
    if not numbers:
        return 0.0
    ordered = sorted(numbers)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def count_above(numbers, threshold):
    """Return how many values are above the threshold."""
    return sum(1 for value in numbers if value >= threshold)
