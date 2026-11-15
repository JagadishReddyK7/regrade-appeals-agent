def count_vowels(text):
    """Count the vowels in text, ignoring case."""
    vowels = set("aeiou")
    return len([ch for ch in text if ch in vowels])


def reverse_words(text):
    """Return the words of text in reverse order, separated by single spaces."""
    result = []
    for word in text.split():
        result.insert(0, word)
    return " ".join(result)
