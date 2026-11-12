import statistics


def mean(numbers):
    """Average of the numbers; 0.0 when the list is empty."""
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


def median(numbers):
    if not numbers:
        return 0.0
    return statistics.median(numbers)


def count_above(numbers, threshold):
    return len([value for value in numbers if value > threshold])
