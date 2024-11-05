import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
import re
import math

nltk.download('punkt')
nltk.download('stopwords')

class WritingStyleAnalyzer:
    """
    Enhanced writing style analyzer that produces metrics compatible with PromptGenerator.
    """

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self._init_linguistic_patterns()

    def _init_linguistic_patterns(self):
        """Initialize patterns for linguistic analysis"""
        self.literary_patterns = {
            'metaphor': ['like', 'as', 'represents', 'symbolizes', 'metaphorically'],
            'simile': ['like', 'as'],
            'personification_verbs': ['says', 'thinks', 'feels', 'knows', 'understands', 'wants']
        }
        
        self.domain_specific_terms = {
            'technical': set(['algorithm', 'function', 'database', 'server', 'code', 'debug']),
            'business': set(['strategy', 'revenue', 'client', 'meeting', 'project', 'deadline']),
            'academic': set(['research', 'theory', 'analysis', 'study', 'hypothesis']),
            'creative': set(['design', 'artistic', 'creative', 'innovative', 'imagination'])
        }

    def analyze_writing_style(self, text):
        """
        Performs comprehensive writing style analysis matching PromptGenerator requirements.

        Args:
            text (str): Text to analyze

        Returns:
            dict: Complete writing style metrics
        """
        # Basic text processing
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        word_lengths = [len(word) for word in words if word.isalpha()]
        
        # Calculate base metrics
        formality = self._calculate_formality(words, sentences)
        vocabulary_complexity = self._calculate_vocabulary_complexity(words)
        
        # Build comprehensive metrics structure
        return {
            'formality': formality,
            'emoji_frequency': self._detect_emoji_frequency(text),
            'sentence_length': len(sentences),
            'vocabulary_complexity': vocabulary_complexity,
            
            'vocabulary_metrics': {
                'domain_terms': self._identify_domain_terms(words),
                'complexity_level': self._determine_complexity_level(vocabulary_complexity),
                'richness_score': self._calculate_lexical_richness(words)
            },
            
            'sentence_metrics': {
                'average_length': sum(len(word_tokenize(s)) for s in sentences) / len(sentences) if sentences else 0,
                'pattern_distribution': self._analyze_sentence_patterns(sentences),
                'complexity_distribution': self._analyze_sentence_complexity(sentences)
            },
            
            'literary_metrics': {
                'metaphor_count': self._count_literary_device(text, 'metaphor'),
                'simile_count': self._count_literary_device(text, 'simile'),
                'personification_count': self._detect_personification(text)
            },
            
            'punctuation_metrics': {
                'emphasis_marks': len(re.findall(r'[!?]{1,}', text)),
                'formal_markers': len(re.findall(r'[;:]', text)),
                'casual_markers': len(re.findall(r'\.{3,}', text)),
                'style': self._determine_punctuation_style(text)
            }
        }

    def _calculate_formality(self, words, sentences):
        """Calculate text formality score"""
        # Count formal indicators
        long_words = sum(1 for word in words if len(word) > 6)
        complex_words = sum(1 for word in words if self.count_syllables(word) > 2)
        
        # Calculate formal/informal ratio
        formal_ratio = (long_words + complex_words) / len(words) if words else 0
        return min(1.0, formal_ratio * 1.5)  # Scale to 0-1

    def _calculate_vocabulary_complexity(self, words):
        """Calculate vocabulary complexity score"""
        unique_words = set(word.lower() for word in words if word.isalpha())
        avg_word_length = sum(len(word) for word in unique_words) / len(unique_words) if unique_words else 0
        avg_syllables = sum(self.count_syllables(word) for word in unique_words) / len(unique_words) if unique_words else 0
        
        # Combine metrics into complexity score
        return min(1.0, (avg_word_length / 10 + avg_syllables / 4) / 2)

    def _identify_domain_terms(self, words):
        """Identify domain-specific terms in text"""
        words_set = set(word.lower() for word in words)
        domain_matches = {}
        
        for domain, terms in self.domain_specific_terms.items():
            matches = words_set.intersection(terms)
            if matches:
                domain_matches[domain] = list(matches)
        
        return domain_matches

    def _determine_complexity_level(self, complexity_score):
        """Determine vocabulary complexity level"""
        if complexity_score < 0.4:
            return 'basic'
        elif complexity_score < 0.7:
            return 'intermediate'
        else:
            return 'advanced'

    def _calculate_lexical_richness(self, words):
        """Calculate lexical richness score"""
        content_words = [w.lower() for w in words if w.isalpha() and w.lower() not in self.stop_words]
        unique_words = set(content_words)
        
        if not content_words:
            return 0.0
            
        ttr = len(unique_words) / len(content_words)  # Type-Token Ratio
        return min(1.0, ttr * 1.5)  # Scale to 0-1

    def _analyze_sentence_patterns(self, sentences):
        """Analyze sentence pattern distribution"""
        patterns = {
            'simple': 0,
            'compound': 0,
            'complex': 0,
            'compound_complex': 0
        }
        
        for sentence in sentences:
            clauses = len(re.findall(r'[,;]|(?<=\w)\sand\s|\sbut\s|\sor\s|\sbecause\s', sentence)) + 1
            
            if clauses == 1:
                patterns['simple'] += 1
            elif clauses == 2:
                patterns['compound'] += 1
            elif clauses == 3:
                patterns['complex'] += 1
            else:
                patterns['compound_complex'] += 1
                
        total = len(sentences)
        return {k: v/total if total else 0 for k, v in patterns.items()}

    def _analyze_sentence_complexity(self, sentences):
        """Analyze sentence complexity distribution"""
        complexity_dist = {
            'simple': 0,
            'moderate': 0,
            'complex': 0
        }
        
        for sentence in sentences:
            words = len(word_tokenize(sentence))
            if words < 10:
                complexity_dist['simple'] += 1
            elif words < 20:
                complexity_dist['moderate'] += 1
            else:
                complexity_dist['complex'] += 1
                
        total = len(sentences)
        return {k: v/total if total else 0 for k, v in complexity_dist.items()}

    def _count_literary_device(self, text, device_type):
        """Count occurrences of literary devices"""
        pattern_words = self.literary_patterns.get(device_type, [])
        count = sum(text.lower().count(f" {word} ") for word in pattern_words)
        return count

    def _detect_personification(self, text):
        """Detect instances of personification"""
        words = word_tokenize(text.lower())
        count = 0
        
        for i, word in enumerate(words[:-1]):
            if (word in self.domain_specific_terms['technical'] or 
                word in self.domain_specific_terms['business']):
                if words[i+1] in self.literary_patterns['personification_verbs']:
                    count += 1
        
        return count

    def _detect_emoji_frequency(self, text):
        """Calculate emoji frequency in text"""
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+", flags=re.UNICODE)
        
        emoji_count = len(emoji_pattern.findall(text))
        word_count = len(word_tokenize(text))
        
        return emoji_count / word_count if word_count else 0

    def _determine_punctuation_style(self, text):
        """Determine punctuation style category"""
        punct_count = len(re.findall(r'[,.!?;:]', text))
        word_count = len(word_tokenize(text))
        
        if not word_count:
            return 'minimal'
            
        ratio = punct_count / word_count
        
        if ratio < 0.1:
            return 'minimal'
        elif ratio < 0.2:
            return 'moderate'
        else:
            return 'heavy'

    def count_syllables(self, word):
        """Count syllables in a word"""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        prev_char_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_char_was_vowel:
                count += 1
            prev_char_was_vowel = is_vowel
            
        if word.endswith("e"):
            count = max(1, count - 1)
            
        return count if count > 0 else 1


if __name__ == "__main__":
    # Example usage
    analyzer = WritingStyleAnalyzer()
    
    sample_text = """
    The complex algorithm elegantly processes data, while the server efficiently manages requests.
    This innovative solution represents a breakthrough in system design!
    The code feels frustrated when debugging takes too long...
    """
    
    metrics = analyzer.analyze_writing_style(sample_text)
    print("Writing Style Analysis Results:")
    for key, value in metrics.items():
        print(f"\n{key}:")
        print(value)