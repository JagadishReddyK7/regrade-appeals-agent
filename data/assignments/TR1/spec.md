# TR1: List statistics

Write a module `list_stats.py` with three functions on lists of numbers.

## `mean(numbers: list[float]) -> float`
Return the arithmetic mean of `numbers`. An empty list has mean `0.0`.

## `median(numbers: list[float]) -> float`
Return the middle value of the sorted list. For an even number of values, return the
average of the two middle values. An empty list has median `0.0`.

## `count_above(numbers: list[float], threshold: float) -> int`
Return how many values are strictly greater than `threshold`. Values equal to the
threshold are not counted.

## Marking (30 marks)
- Autograder tests: 24 marks.
- Style (3 marks): every function has a docstring.
- Approach (3 marks): do not import any modules; write the calculations yourself.

## Late policy
Late submissions lose 10 marks. Extensions, including for medical reasons, are decided
by the course staff on request.
