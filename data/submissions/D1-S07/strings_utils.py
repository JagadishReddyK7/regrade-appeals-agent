def count_vowels(text):
    """Return how many vowels text contains, ignoring case."""
    return len([ch for ch in text.lower() if ch in "aeiou"])


def reverse_words(text):
    """Return the words of text in reverse order."""
    return " ".join(reversed(text.split()))
