def count_vowels(text):
    """Count the vowels in text, ignoring case."""
    lowered = text.lower()
    return sum(lowered.count(v) for v in "aeiou")


def reverse_words(text):
    """Return the words of text in reverse order, separated by single spaces."""
    words = text.split()
    if len(words) <= 1:
        return "".join(words)
    first, *rest = words
    return reverse_words(" ".join(rest)) + " " + first
