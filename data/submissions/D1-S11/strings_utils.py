def count_vowels(text):
    """Count the vowels in text, ignoring case."""
    total = 0
    for ch in text.lower():
        if ch in "aeiou":
            total += 1
    return total


def reverse_words(text):
    """Return the words of text in reverse order, using recursion."""
    words = text.split()
    if len(words) <= 1:
        return " ".join(words)
    return reverse_words(" ".join(words[1:])) + " " + words[0]
