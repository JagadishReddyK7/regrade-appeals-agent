def count_vowels(text):
    """Count the vowels in text, ignoring case."""
    lowered = text.lower()
    return sum(lowered.count(v) for v in "aeiou")


def reverse_words(text):
    return " ".join(reversed(text.split()))
