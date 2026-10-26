def count_vowels(text):
    """Count the vowels in text."""
    return sum(1 for ch in text if ch in "aeiou")


def reverse_words(text):
    """Reverse the order of the words in text."""
    return " ".join(text.split()[::-1])
