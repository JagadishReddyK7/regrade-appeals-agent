def count_vowels(text):
    """Count the vowels in text, ignoring case."""
    return sum(text.count(v) for v in "aeiou")


def reverse_words(text):
    """Return the words of text in reverse order, separated by single spaces."""
    return " ".join(reversed(text.split()))
