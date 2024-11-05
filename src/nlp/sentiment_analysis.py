import re
import logging
from typing import Tuple, List, Optional, Dict, Union
from dataclasses import dataclass
from transformers import pipeline, Pipeline
import numpy as np
from collections import deque
from datetime import datetime

@dataclass
class SentimentResult:
    """Data class to hold sentiment analysis results."""
    sentiment: str
    score: float
    confidence: float
    text: str
    timestamp: datetime
    aspects: Optional[Dict[str, float]] = None

class EnhancedSentimentAnalyzer:
    """
    Enhanced sentiment analyzer with additional features like:
    - Aspect-based sentiment analysis
    - Confidence scoring
    - Historical analysis
    - Emotion detection
    - Caching
    - Batch processing optimization
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        threshold_pos: float = 0.6,
        threshold_neg: float = 0.6,
        cache_size: int = 1000,
        batch_size: int = 32
    ):
        """
        Initialize the Enhanced Sentiment Analyzer.

        Args:
            model_name (str): Name of the pre-trained model
            threshold_pos (float): Threshold for positive classification
            threshold_neg (float): Threshold for negative classification
            cache_size (int): Size of the result cache
            batch_size (int): Size of batches for processing
        """
        try:
            self.sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
            self.emotion_pipeline = pipeline("text-classification", 
                                          model="j-hartmann/emotion-english-distilroberta-base")
            
            self.threshold_pos = threshold_pos
            self.threshold_neg = threshold_neg
            self.batch_size = batch_size
            self.cache = self._initialize_cache(cache_size)
            self.aspect_keywords = {
                'quality': ['quality', 'good', 'bad', 'excellent', 'poor'],
                'service': ['service', 'staff', 'support', 'helpful', 'responsive'],
                'price': ['price', 'cost', 'expensive', 'cheap', 'worth'],
                'reliability': ['reliable', 'consistent', 'stable', 'unreliable', 'inconsistent']
            }
            
            logging.info(f"Initialized Enhanced Sentiment Analyzer with model: {model_name}")
        except Exception as e:
            logging.error(f"Failed to initialize sentiment analyzer: {str(e)}")
            raise

    def _initialize_cache(self, cache_size: int) -> Dict:
        """Initialize LRU cache with specified size."""
        return {'data': {}, 'queue': deque(maxlen=cache_size)}

    def _preprocess_text(self, text: str) -> str:
        """
        Enhanced text preprocessing with additional cleaning steps.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # Basic cleaning
        text = text.lower().strip()
        
        # Remove URLs, mentions, and special characters
        text = re.sub(r"http\S+|www.\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Handle common abbreviations
        text = re.sub(r"'s", " is", text)
        text = re.sub(r"'m", " am", text)
        text = re.sub(r"'re", " are", text)
        
        # Remove excessive punctuation
        text = re.sub(r'([!?.]){2,}', r'\1', text)
        
        return text.strip()

    def _calculate_confidence(self, score: float, text_length: int) -> float:
        """
        Calculate confidence score based on sentiment score and text characteristics.
        
        Args:
            score (float): Raw sentiment score
            text_length (int): Length of input text
            
        Returns:
            float: Confidence score
        """
        # Adjust confidence based on text length
        length_factor = min(text_length / 100, 1.0)  # Normalize text length
        
        # Calculate base confidence from sentiment score
        base_confidence = abs(score - 0.5) * 2
        
        # Combine factors
        confidence = base_confidence * length_factor
        
        return round(confidence, 3)

    def _analyze_aspects(self, text: str) -> Dict[str, float]:
        """
        Perform aspect-based sentiment analysis.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, float]: Sentiment scores for each aspect
        """
        aspects = {}
        text_lower = text.lower()
        
        for aspect, keywords in self.aspect_keywords.items():
            aspect_mentions = sum(1 for keyword in keywords if keyword in text_lower)
            if aspect_mentions > 0:
                # Analyze sentiment for sentences containing aspect keywords
                sentences = [s for s in text.split('.') if any(keyword in s.lower() for keyword in keywords)]
                if sentences:
                    results = self.sentiment_pipeline(sentences)
                    scores = [r['score'] if r['label'] == 'POSITIVE' else 1 - r['score'] for r in results]
                    aspects[aspect] = round(np.mean(scores), 3)
        
        return aspects

    def analyze_sentiment(self, text: str, include_aspects: bool = False) -> Optional[SentimentResult]:
        """
        Analyze sentiment with enhanced features.
        
        Args:
            text (str): Input text
            include_aspects (bool): Whether to include aspect-based analysis
            
        Returns:
            Optional[SentimentResult]: Detailed sentiment analysis result
        """
        try:
            if not text or not (preprocessed_text := self._preprocess_text(text)):
                logging.warning("Empty or invalid text provided")
                return None

            # Check cache
            cache_key = hash(preprocessed_text)
            if cache_key in self.cache['data']:
                return self.cache['data'][cache_key]

            # Perform sentiment analysis
            sentiment_result = self.sentiment_pipeline(preprocessed_text)[0]
            emotion_result = self.emotion_pipeline(preprocessed_text)[0]
            
            # Calculate scores
            score = sentiment_result['score']
            confidence = self._calculate_confidence(score, len(preprocessed_text))
            
            # Determine sentiment
            if sentiment_result['label'] == 'POSITIVE' and score >= self.threshold_pos:
                sentiment = 'Positive'
            elif sentiment_result['label'] == 'NEGATIVE' and score >= self.threshold_neg:
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'

            # Perform aspect analysis if requested
            aspects = self._analyze_aspects(preprocessed_text) if include_aspects else None

            # Create result
            result = SentimentResult(
                sentiment=sentiment,
                score=round(score, 3),
                confidence=confidence,
                text=text,
                timestamp=datetime.now(),
                aspects=aspects
            )

            # Update cache
            self.cache['data'][cache_key] = result
            self.cache['queue'].append(cache_key)
            if len(self.cache['queue']) >= self.cache['queue'].maxlen:
                old_key = self.cache['queue'].popleft()
                self.cache['data'].pop(old_key, None)

            return result

        except Exception as e:
            logging.error(f"Error in sentiment analysis: {str(e)}")
            return None

    def analyze_sentiments_batch(
        self,
        texts: List[str],
        include_aspects: bool = False,
        progress_callback: Optional[callable] = None
    ) -> List[Optional[SentimentResult]]:
        """
        Analyze multiple texts in optimized batches.
        
        Args:
            texts (List[str]): List of input texts
            include_aspects (bool): Whether to include aspect-based analysis
            progress_callback (callable): Optional callback for progress updates
            
        Returns:
            List[Optional[SentimentResult]]: List of sentiment analysis results
        """
        results = []
        total_texts = len(texts)

        try:
            for i in range(0, total_texts, self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_results = [self.analyze_sentiment(text, include_aspects) for text in batch]
                results.extend(batch_results)

                if progress_callback:
                    progress = min((i + self.batch_size) / total_texts * 100, 100)
                    progress_callback(progress)

            return results

        except Exception as e:
            logging.error(f"Error in batch sentiment analysis: {str(e)}")
            return [None] * len(texts)

    def get_sentiment_statistics(self, results: List[SentimentResult]) -> Dict[str, Union[float, int]]:
        """
        Calculate statistics from a list of sentiment results.
        
        Args:
            results (List[SentimentResult]): List of sentiment analysis results
            
        Returns:
            Dict[str, Union[float, int]]: Statistical summary
        """
        valid_results = [r for r in results if r is not None]
        if not valid_results:
            return {}

        stats = {
            'total_analyzed': len(valid_results),
            'positive_count': sum(1 for r in valid_results if r.sentiment == 'Positive'),
            'negative_count': sum(1 for r in valid_results if r.sentiment == 'Negative'),
            'neutral_count': sum(1 for r in valid_results if r.sentiment == 'Neutral'),
            'average_confidence': np.mean([r.confidence for r in valid_results]),
            'average_score': np.mean([r.score for r in valid_results])
        }

        return {k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items()}