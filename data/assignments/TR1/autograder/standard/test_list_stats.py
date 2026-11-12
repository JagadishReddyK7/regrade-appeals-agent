from list_stats import count_above, mean, median


def test_mean_basic():
    assert mean([2, 4, 6]) == 4


def test_mean_empty():
    assert mean([]) == 0.0


def test_median_odd():
    assert median([5, 1, 3]) == 3


def test_median_even():
    assert median([4, 1, 3, 2]) == 2.5


def test_count_above_basic():
    assert count_above([1, 5, 7, 2], 3) == 2


def test_count_above_strict():
    assert count_above([3, 3, 4], 3) == 1
