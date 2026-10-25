def count_vowels(text):
    count = 0
    for ch in text.lower():
        if ch in "aeiou":
            count += 1
    return count


def reverse_words(text):
    words = text.split()
    words.reverse()
    return " ".join(words)
