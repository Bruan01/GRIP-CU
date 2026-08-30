import re
import string


def normalize_answer(answer: str) -> str:
    """
    Normalize a given string by applying the following transformations:
    1. Convert the string to lowercase.
    2. Remove punctuation characters.
    3. Remove the articles "a", "an", and "the".
    4. Normalize whitespace by collapsing multiple spaces into one.

    Args:
        answer (str): The input string to be normalized.

    Returns:
        str: The normalized string.
    """
    # Custom word normalization mapping
    normalization_map = {
        "man": "men",
        "woman": "women",
    }

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def normalize_words(text):
        return " ".join(normalization_map.get(word, word) for word in text.split())

    return white_space_fix(
        normalize_words(
            remove_articles(
                remove_punc(
                    lower(answer)
                )
            )
        )
    )
