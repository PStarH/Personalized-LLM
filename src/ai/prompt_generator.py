class PromptGenerator:
    """
    Advanced prompt generator that combines original functionality with deep linguistic analysis,
    supporting both basic context-based generation and advanced style analysis.
    """

    def __init__(self):
        self._init_context_patterns()
        self._init_sentiment_patterns()
        self._init_style_patterns()
        self._init_linguistic_patterns()

    def _init_context_patterns(self):
        """Initialize context analysis patterns and indicators"""
        self.context_patterns = {
            'casual_chat': {
                'indicators': ['hey', 'hi', 'hello', 'sup', 'what\'s up'],
                'topics': ['daily', 'friends', 'life', 'fun', 'social'],
                'style_weight': 0.3
            },
            'technical_discussion': {
                'indicators': ['code', 'error', 'bug', 'system', 'tech', 'help'],
                'topics': ['programming', 'software', 'hardware', 'development'],
                'style_weight': 0.7
            },
            'emotional_support': {
                'indicators': ['feel', 'sad', 'happy', 'angry', 'frustrated'],
                'topics': ['emotions', 'support', 'advice', 'help'],
                'style_weight': 0.5
            },
            'business_professional': {
                'indicators': ['meeting', 'project', 'deadline', 'client', 'work'],
                'topics': ['business', 'professional', 'work', 'corporate'],
                'style_weight': 0.8
            },
            'creative_discussion': {
                'indicators': ['idea', 'create', 'design', 'art', 'story'],
                'topics': ['creativity', 'design', 'art', 'writing'],
                'style_weight': 0.4
            },
            'academic_discussion': {
                'indicators': ['research', 'study', 'theory', 'analysis'],
                'topics': ['academic', 'education', 'research', 'study'],
                'style_weight': 0.9
            }
        }

    def _init_sentiment_patterns(self):
        """Initialize sentiment analysis patterns"""
        self.sentiment_patterns = {
            'positive': {
                'modifiers': ['enthusiastic', 'happy', 'excited', 'satisfied'],
                'emoji_weight': 0.7,
                'style_adjustments': {'formality': -0.2, 'emoji_frequency': 0.3}
            },
            'negative': {
                'modifiers': ['frustrated', 'angry', 'disappointed', 'sad'],
                'emoji_weight': 0.3,
                'style_adjustments': {'formality': 0.1, 'emoji_frequency': -0.2}
            },
            'neutral': {
                'modifiers': ['calm', 'balanced', 'objective', 'neutral'],
                'emoji_weight': 0.5,
                'style_adjustments': {'formality': 0, 'emoji_frequency': 0}
            },
            'urgent': {
                'modifiers': ['urgent', 'important', 'critical', 'asap'],
                'emoji_weight': 0.2,
                'style_adjustments': {'formality': 0.3, 'emoji_frequency': -0.3}
            }
        }

    def _init_style_patterns(self):
        """Initialize writing style patterns"""
        self.style_patterns = {
            'formal': {
                'template': """
                Context: {context}
                Sentiment: {sentiment}

                Generate a formal yet personable response that:
                - Addresses the main points concisely.
                - Uses professional language.
                - Maintains a respectful tone.
                """,
                'formality_adjustment': 1.0
            },
            'casual': {
                'template': """
                Context: {context}
                Sentiment: {sentiment}

                Generate a friendly and casual response that:
                - Uses informal language.
                - Includes conversational elements.
                - Maintains an approachable tone.
                """,
                'formality_adjustment': -0.3
            },
            'emotional': {
                'template': """
                Context: {context}
                Sentiment: {sentiment}

                Generate an emotionally supportive response that:
                - Acknowledges the user's feelings.
                - Offers empathy and understanding.
                - Uses compassionate language.
                """,
                'formality_adjustment': 0.0
            },
            'technical': {
                'template': """
                Context: {context}
                Sentiment: {sentiment}

                Generate a detailed technical explanation that:
                - Uses precise terminology.
                - Provides clear and logical reasoning.
                - Includes necessary technical details without overwhelming.
                """,
                'formality_adjustment': 0.2
            }
        }

    def _init_linguistic_patterns(self):
        """Initialize linguistic analysis patterns"""
        self.linguistic_patterns = {
            'sentence_structure': ['simple', 'compound', 'complex'],
            'punctuation_usage': ['minimal', 'moderate', 'heavy']
        }

    def generate_prompt(self, relevant_data: str, sentiment: str, writing_style_metrics: Dict[str, float]) -> str:
        """
        Generate an enhanced prompt based on relevant data, sentiment, and writing style.

        Args:
            relevant_data (str): Data retrieved for context.
            sentiment (str): Sentiment of the user's input.
            writing_style_metrics (Dict[str, float]): Aggregated writing style metrics.

        Returns:
            str: Formatted prompt for the LLM.
        """
        # Determine context type
        context_type = self._determine_context_type(relevant_data, writing_style_metrics)

        # Determine sentiment category
        sentiment_category = self._determine_sentiment_category(sentiment)

        # Select style template
        style_template = self.style_patterns.get(context_type, self.style_patterns['formal'])

        # Prepare prompt
        prompt = style_template['template'].format(
            context=relevant_data,
            sentiment=sentiment_category
        )

        # Optionally adjust prompt based on writing style metrics
        optimized_prompt = self._optimize_final_prompt(prompt, writing_style_metrics)

        return optimized_prompt

    def _determine_context_type(self, relevant_data: str, style_metrics: Dict[str, float]) -> str:
        """Determine the context type based on relevant data and style metrics."""
        # Placeholder logic to determine context type
        # This should be replaced with actual implementation
        return 'formal'

    def _determine_sentiment_category(self, sentiment: str) -> str:
        """Map sentiment to predefined categories."""
        sentiment_map = {
            'Positive': 'positive',
            'Negative': 'negative',
            'Neutral': 'neutral',
            'Urgent': 'urgent'
        }
        return sentiment_map.get(sentiment, 'neutral')

    def _optimize_final_prompt(self, prompt: str, style_metrics: Dict[str, float]) -> str:
        """Optimize the final prompt based on writing style metrics."""
        # Placeholder for optimization logic
        # This can include adjusting formality, verbosity, etc.
        return prompt.strip()

if __name__ == "__main__":
    # Initialize the generator
    generator = PromptGenerator()

    # Example usage with aggregated writing style metrics
    relevant_data = "User is having trouble with their Python code and seems frustrated. They've been trying to debug for hours."
    sentiment = "frustrated"
    writing_style = {
        'average_sentence_length': 12,
        'average_word_length': 5,
        'lexical_diversity': 0.6,
        'readability_score': 65.0
    }

    # Generate enhanced prompt
    prompt = generator.generate_prompt(relevant_data, sentiment, writing_style)
    print("Generated Prompt:")
    print(prompt)