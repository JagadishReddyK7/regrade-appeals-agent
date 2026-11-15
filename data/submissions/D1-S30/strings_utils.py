def count_vowels(text):
    return sum(1 for ch in text.lower() if ch in "aeiou")


def reverse_words(text):
    """Return the words of text in reverse order, separated by single spaces."""
    return " ".join(text.split()[::-1])
