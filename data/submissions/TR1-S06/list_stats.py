def mean(numbers):
    """Return the arithmetic mean of numbers."""
    return sum(numbers) / len(numbers)


def median(numbers):
    """Return the median of numbers, 0.0 if there are none."""
    if not numbers:
        return 0.0
    ordered = sorted(numbers)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def count_above(numbers, threshold):
    """Return the number of values strictly greater than threshold."""
    total = 0
    for value in numbers:
        if value > threshold:
            total += 1
    return total
