class PromptGenerator:
    """
    Advanced prompt generator that creates highly personalized prompts based on
    deep context analysis, sentiment patterns, and writing style metrics.
    """

    def __init__(self):
        self._init_context_patterns()
        self._init_sentiment_patterns()
        self._init_style_patterns()
        
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
                - Maintains professional language and structure
                - Uses industry-appropriate terminology
                - Keeps a balanced, measured tone
                - Includes minimal, professional emojis if any
                - Follows standard business writing conventions
                
                Key points to address:
                {key_points}
                
                Style notes:
                {style_notes}
                """,
                'emoji_set': ['👍', '✅', '📊', '💼'],
                'formality_threshold': 0.7
            },
            'casual': {
                'template': """
                Context: {context}
                Vibe: {sentiment}
                
                Write back in a super casual, friendly way that:
                - Uses natural, conversational language
                - Includes common internet slang/abbreviations
                - Adds emojis for extra flavor
                - Keeps it real and relatable
                - Matches their energy level
                
                Main points to hit:
                {key_points}
                
                Style vibes:
                {style_notes}
                """,
                'emoji_set': ['😊', '😂', '🙌', '💯', '🔥', '✨'],
                'formality_threshold': 0.3
            },
            'technical': {
                'template': """
                Context: {context}
                Current State: {sentiment}
                
                Create a technical yet accessible response that:
                - Balances technical accuracy with clarity
                - Uses appropriate technical terminology
                - Includes practical examples or analogies
                - Maintains a helpful, educational tone
                - Adds relevant technical emojis sparingly
                
                Technical points to cover:
                {key_points}
                
                Style guidelines:
                {style_notes}
                """,
                'emoji_set': ['💻', '🔧', '⚙️', '📱', '💡'],
                'formality_threshold': 0.6
            },
            'emotional': {
                'template': """
                Context: {context}
                Emotional State: {sentiment}
                
                Craft an empathetic, supportive response that:
                - Shows genuine understanding and care
                - Uses warm, compassionate language
                - Validates their feelings
                - Offers gentle support or encouragement
                - Includes appropriate supportive emojis
                
                Key elements to address:
                {key_points}
                
                Tone guidance:
                {style_notes}
                """,
                'emoji_set': ['❤️', '🤗', '💪', '🙏', '✨'],
                'formality_threshold': 0.4
            }
        }

    def _analyze_writing_style(self, writing_style):
        """
        Deeply analyzes writing style metrics to create personalized style guide.
        
        Args:
            writing_style (dict): Writing style metrics
            
        Returns:
            dict: Analyzed style characteristics
        """
        style_analysis = {
            'formality_level': writing_style.get('formality', 0.5),
            'emoji_usage': writing_style.get('emoji_frequency', 0.5),
            'sentence_complexity': writing_style.get('sentence_length', 15),
            'vocabulary_level': writing_style.get('vocabulary_complexity', 0.5),
            'tone_markers': []
        }

        # Analyze formality
        if style_analysis['formality_level'] < 0.3:
            style_analysis['tone_markers'].extend(['very casual', 'informal'])
        elif style_analysis['formality_level'] < 0.6:
            style_analysis['tone_markers'].extend(['conversational', 'balanced'])
        else:
            style_analysis['tone_markers'].extend(['formal', 'professional'])

        # Analyze emoji usage
        if style_analysis['emoji_usage'] > 0.7:
            style_analysis['tone_markers'].append('emoji-heavy')
        elif style_analysis['emoji_usage'] > 0.3:
            style_analysis['tone_markers'].append('moderate-emojis')

        # Analyze sentence complexity
        if style_analysis['sentence_complexity'] < 10:
            style_analysis['tone_markers'].append('short-sentences')
        elif style_analysis['sentence_complexity'] > 20:
            style_analysis['tone_markers'].append('complex-sentences')

        return style_analysis

    def _extract_key_points(self, relevant_data):
        """
        Extracts and organizes key points from relevant data.
        
        Args:
            relevant_data (str): The relevant context data
            
        Returns:
            list: Organized key points
        """
        # Split into sentences and clean
        sentences = [s.strip() for s in relevant_data.split('.') if s.strip()]
        
        key_points = []
        for sentence in sentences:
            # Skip very short or uninformative sentences
            if len(sentence.split()) < 3:
                continue
                
            # Add as key point with importance marker
            importance = 'high' if any(word in sentence.lower() for word in 
                ['important', 'critical', 'essential', 'must', 'need']) else 'normal'
            
            key_points.append({
                'content': sentence,
                'importance': importance
            })
            
        return key_points

    def _determine_style_template(self, context_type, sentiment, style_analysis):
        """
        Determines the most appropriate style template based on multiple factors.
        
        Args:
            context_type (str): The type of conversation context
            sentiment (str): The sentiment analysis result
            style_analysis (dict): Analyzed writing style characteristics
            
        Returns:
            str: The selected style template name
        """
        # Calculate style scores for each template
        scores = {
            'formal': 0,
            'casual': 0,
            'technical': 0,
            'emotional': 0
        }

        # Context-based scoring
        if context_type in ['technical_discussion', 'academic_discussion']:
            scores['technical'] += 2
            scores['formal'] += 1
        elif context_type in ['casual_chat', 'creative_discussion']:
            scores['casual'] += 2
        elif context_type == 'emotional_support':
            scores['emotional'] += 2
            scores['casual'] += 1
        elif context_type == 'business_professional':
            scores['formal'] += 2

        # Sentiment-based scoring
        if sentiment in ['frustrated', 'sad', 'worried']:
            scores['emotional'] += 1.5
        elif sentiment in ['excited', 'happy']:
            scores['casual'] += 1

        # Style-based scoring
        if style_analysis['formality_level'] > 0.7:
            scores['formal'] += 1.5
        elif style_analysis['formality_level'] < 0.3:
            scores['casual'] += 1.5

        # Return template with highest score
        return max(scores.items(), key=lambda x: x[1])[0]

    def generate_prompt(self, relevant_data, sentiment, writing_style):
        """
        Generates a highly personalized prompt based on deep context analysis.
        
        Args:
            relevant_data (str): Retrieved relevant data
            sentiment (str): Sentiment analysis result
            writing_style (dict): Writing style metrics
            
        Returns:
            str: Generated personalized prompt
        """
        # Analyze writing style
        style_analysis = self._analyze_writing_style(writing_style)
        
        # Determine context type
        context_type = 'casual_chat'  # default
        for ctype, patterns in self.context_patterns.items():
            if any(indicator in relevant_data.lower() for indicator in patterns['indicators']):
                context_type = ctype
                break
        
        # Extract key points
        key_points = self._extract_key_points(relevant_data)
        
        # Determine appropriate style template
        template_name = self._determine_style_template(context_type, sentiment, style_analysis)
        template = self.style_patterns[template_name]['template']
        
        # Format style notes
        style_notes = [
            f"- Writing style: {', '.join(style_analysis['tone_markers'])}",
            f"- Emoji usage: {'high' if style_analysis['emoji_usage'] > 0.5 else 'low'}",
            f"- Sentence structure: {'complex' if style_analysis['sentence_complexity'] > 15 else 'simple'}"
        ]
        
        # Format key points
        formatted_key_points = [
            f"- {point['content']}" + (" (Important!)" if point['importance'] == 'high' else "")
            for point in key_points
        ]
        
        # Generate final prompt
        prompt = template.format(
            context=relevant_data,
            sentiment=sentiment,
            key_points="\n".join(formatted_key_points),
            style_notes="\n".join(style_notes)
        )
        
        # Clean up whitespace
        return "\n".join(line.strip() for line in prompt.split('\n') if line.strip())

    def add_context_pattern(self, name, indicators, topics, style_weight):
        """
        Adds a new context pattern for recognition.
        
        Args:
            name (str): Pattern name
            indicators (list): List of indicator words
            topics (list): List of related topics
            style_weight (float): Style weight (0-1)
        """
        self.context_patterns[name] = {
            'indicators': indicators,
            'topics': topics,
            'style_weight': style_weight
        }

    def add_style_pattern(self, name, template, emoji_set, formality_threshold):
        """
        Adds a new style pattern template.
        
        Args:
            name (str): Pattern name
            template (str): Template string
            emoji_set (list): List of appropriate emojis
            formality_threshold (float): Formality threshold (0-1)
        """
        self.style_patterns[name] = {
            'template': template,
            'emoji_set': emoji_set,
            'formality_threshold': formality_threshold
        }
if __name__ == "__main__":
    # Initialize the generator
    generator = PromptGenerator()

    # Example data
    relevant_data = "User is having trouble with their Python code and seems frustrated. They've been trying to debug for hours."
    sentiment = "frustrated"
    writing_style = {
        'formality': 0.3,
        'emoji_frequency': 0.7,
        'sentence_length': 12,
        'vocabulary_complexity': 0.4
    }

    # Generate prompt
    prompt = generator.generate_prompt(relevant_data, sentiment, writing_style)

    # Add custom context pattern if needed
    generator.add_context_pattern(
        name="coding_help",
        indicators=["debug", "error", "code", "programming"],
        topics=["coding", "debugging", "development"],
        style_weight=0.6
    )

    # Add custom style pattern
    generator.add_style_pattern(
        name="debug_helper",
        template="""
        Context: {context}
        Current State: {sentiment}
        
        Write a helpful debugging response that:
        - Shows understanding of their frustration
        - Provides clear technical guidance
        - Uses friendly, encouraging tone
        - Includes relevant code examples
        
        Points to address:
        {key_points}
        
        Style guide:
        {style_notes}
        """,
        emoji_set=['💻', '🐛', '✨', '💡'],
        formality_threshold=0.5
    )