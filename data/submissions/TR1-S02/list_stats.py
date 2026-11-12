def mean(numbers):
    if len(numbers) == 0:
        return 0.0
    total = 0
    for value in numbers:
        total += value
    return total / len(numbers)


def median(numbers):
    if len(numbers) == 0:
        return 0.0
    ordered = sorted(numbers)
    return ordered[len(ordered) // 2]


def count_above(numbers, threshold):
    count = 0
    for value in numbers:
        if value > threshold:
            count += 1
    return count
