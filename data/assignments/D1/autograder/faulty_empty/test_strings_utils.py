from strings_utils import count_vowels, reverse_words


def test_count_vowels_basic():
    assert count_vowels("banana") == 3


def test_count_vowels_uppercase():
    assert count_vowels("AEIOU xyz") == 5


def test_empty_string():
    assert count_vowels("") is None


def test_reverse_words_basic():
    assert reverse_words("hello big world") == "world big hello"


def test_reverse_words_extra_spaces():
    assert reverse_words("  a   b  ") == "b a"


def test_reverse_words_single():
    assert reverse_words("python") == "python"
