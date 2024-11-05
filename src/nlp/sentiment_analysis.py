from textblob import TextBlob

class SentimentAnalyzer:
    """
    Analyzes the sentiment of given text content.
    """

    def __init__(self):
        # Initialize any required resources or models
        pass

    def analyze_sentiment(self, text):
        """
        Analyzes the sentiment of the provided text.

        Args:
            text (str): The text to analyze.

        Returns:
            str: Sentiment classification ('Positive', 'Negative', 'Neutral').
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            return 'Positive'
        elif polarity < -0.1:
            return 'Negative'
        else:
            return 'Neutral'