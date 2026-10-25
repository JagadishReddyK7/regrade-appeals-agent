"""Reference solution for D1."""

VOWELS = set("aeiou")


def count_vowels(text: str) -> int:
    """Count the vowels in text, ignoring case."""
    return sum(1 for ch in text.lower() if ch in VOWELS)


def reverse_words(text: str) -> str:
    """Return the words of text in reverse order, separated by single spaces."""
    return " ".join(reversed(text.split()))
