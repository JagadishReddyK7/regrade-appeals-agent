def count_vowels(text):
    """Count the vowels in text, ignoring case."""
    return len([ch for ch in text.casefold() if ch in "aeiou"])


def reverse_words(text):
    """Return the words of text in reverse order, separated by single spaces."""
    words = text.split()
    if len(words) <= 1:
        return " ".join(words)
    middle = reverse_words(" ".join(words[1:-1]))
    parts = [words[-1], middle, words[0]]
    return " ".join(part for part in parts if part)
