def count_vowels(text):
    """Count the vowels in text, ignoring case."""
    count = 0
    for ch in text.lower():
        if ch in "aeiou":
            count += 1
    return count


def reverse_words(text):
    """Return the words of text in reverse order, separated by single spaces."""
    words = text.split()
    if len(words) < 2:
        return " ".join(words)
    return words[-1] + " " + reverse_words(" ".join(words[:-1]))
