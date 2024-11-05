import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import math

nltk.download('punkt')

class WritingStyleAnalyzer:
    """
    Analyzes the writing style of given text content.
    """

    def __init__(self):
        # Initialize any required resources
        pass

    def analyze_writing_style(self, text):
        """
        Analyzes writing style metrics of the provided text.

        Args:
            text (str): The text to analyze.

        Returns:
            dict: Dictionary containing writing style metrics.
        """
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        word_lengths = [len(word) for word in words if word.isalpha()]
        sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
        vocabulary = set(words)
        lexical_diversity = len(vocabulary) / len(words) if words else 0

        # Example readability score using Flesch Reading Ease
        syllable_count = sum(self.count_syllables(word) for word in words)
        if len(sentences) > 0 and len(words) > 0:
            readability = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syllable_count / len(words))
        else:
            readability = 0.0

        metrics = {
            'average_sentence_length': sum(sentence_lengths) / len(sentences) if sentences else 0,
            'average_word_length': sum(word_lengths) / len(word_lengths) if word_lengths else 0,
            'lexical_diversity': lexical_diversity,
            'readability_score': readability
        }
        return metrics

    def count_syllables(self, word):
        """
        Counts the number of syllables in a word.

        Args:
            word (str): The word to count syllables for.

        Returns:
            int: Number of syllables.
        """
        word = word.lower()
        vowels = "aeiouy"
        syllables = 0
        prev_char_was_vowel = False

        for char in word:
            if char in vowels:
                if not prev_char_was_vowel:
                    syllables += 1
                    prev_char_was_vowel = True
            else:
                prev_char_was_vowel = False

        if word.endswith("e"):
            syllables = max(1, syllables - 1)

        return syllables if syllables > 0 else 1