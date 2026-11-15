def count_vowels(text):
    """Count the vowels in text, ignoring case."""
    return sum(1 for ch in text.lower() if ch in "aeiou")


def reverse_words(text):
    """Return the words of text in reverse order, separated by single spaces."""
    words = text.split()
    if not words:
        return ""
    rest = reverse_words(" ".join(words[1:]))
    return (rest + " " + words[0]).strip()
