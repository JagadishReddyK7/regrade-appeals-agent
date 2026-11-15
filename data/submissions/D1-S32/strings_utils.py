def count_vowels(text):
    return len([ch for ch in text.casefold() if ch in "aeiou"])


def reverse_words(text):
    """Return the words of text in reverse order, separated by single spaces."""
    result = []
    for word in text.split():
        result.insert(0, word)
    return " ".join(result)
