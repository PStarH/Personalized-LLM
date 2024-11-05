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
                - You can use Jocks for the 
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

    def _init_linguistic_patterns(self):
        """Initialize comprehensive linguistic analysis patterns"""
        self.linguistic_patterns = {
            'vocabulary_metrics': {
                'complexity_levels': {
                    'basic': {'weight': 0.3, 'indicators': ['common', 'everyday', 'simple']},
                    'intermediate': {'weight': 0.6, 'indicators': ['field-specific', 'technical']},
                    'advanced': {'weight': 0.9, 'indicators': ['specialized', 'academic', 'theoretical']}
                },
                'domain_specific': set(),
                'richness_threshold': 0.7
            },
            'sentence_structure': {
                'types': {
                    'simple': {'weight': 0.3, 'max_clauses': 1},
                    'compound': {'weight': 0.6, 'max_clauses': 2},
                    'complex': {'weight': 0.8, 'max_clauses': 3},
                    'compound_complex': {'weight': 1.0, 'max_clauses': 4}
                },
                'length_metrics': {
                    'short': {'max_words': 10, 'weight': 0.3},
                    'medium': {'max_words': 20, 'weight': 0.6},
                    'long': {'max_words': 30, 'weight': 0.8},
                    'very_long': {'max_words': float('inf'), 'weight': 1.0}
                }
            },
            'literary_devices': {
                'metaphor': {'weight': 0.7, 'markers': ['like', 'as', 'represents']},
                'simile': {'weight': 0.5, 'markers': ['like', 'as']},
                'personification': {'weight': 0.8, 'markers': ['human_verbs_for_objects']},
                'alliteration': {'weight': 0.6, 'detection': 'consecutive_same_letter'}
            },
            'punctuation_style': {
                'minimal': {'weight': 0.3, 'threshold': 0.2},
                'moderate': {'weight': 0.6, 'threshold': 0.5},
                'heavy': {'weight': 0.9, 'threshold': 0.8},
                'markers': {
                    'emphasis': ['!', '...'],
                    'formal': [';', ':'],
                    'casual': ['...', '!!', '??']
                }
            }
        }

    def _analyze_writing_style(self, writing_style):
        """
        Analyzes writing style combining original metrics with linguistic analysis.
        
        Args:
            writing_style (dict): Writing style metrics
            
        Returns:
            dict: Analyzed style characteristics
        """
        # Original style analysis
        basic_analysis = {
            'formality_level': writing_style.get('formality', 0.5),
            'emoji_usage': writing_style.get('emoji_frequency', 0.5),
            'sentence_complexity': writing_style.get('sentence_length', 15),
            'vocabulary_level': writing_style.get('vocabulary_complexity', 0.5),
            'tone_markers': []
        }

        # Enhanced linguistic analysis
        linguistic_analysis = {
            'vocabulary': self._analyze_vocabulary(writing_style),
            'sentence_structure': self._analyze_sentence_structure(writing_style),
            'literary_devices': self._analyze_literary_devices(writing_style),
            'punctuation': self._analyze_punctuation(writing_style)
        }

        # Combine analyses
        return {**basic_analysis, 'linguistic_features': linguistic_analysis}

    def _analyze_vocabulary(self, writing_style):
        """Analyzes vocabulary characteristics"""
        metrics = writing_style.get('vocabulary_metrics', {})
        return {
            'complexity': self._calculate_vocabulary_complexity(metrics),
            'variety': self._calculate_vocabulary_variety(metrics),
            'domain_specificity': self._analyze_domain_terms(metrics)
        }

    def _analyze_sentence_structure(self, writing_style):
        """Analyzes sentence structure patterns"""
        metrics = writing_style.get('sentence_metrics', {})
        return {
            'avg_length': metrics.get('average_length', 15),
            'complexity_distribution': self._analyze_sentence_complexity(metrics),
            'pattern_variety': self._calculate_pattern_variety(metrics)
        }

    def _analyze_literary_devices(self, writing_style):
        """Analyzes use of literary devices"""
        metrics = writing_style.get('literary_metrics', {})
        return {
            'metaphor_frequency': self._detect_metaphors(metrics),
            'simile_frequency': self._detect_similes(metrics),
            'alliteration_frequency': self._detect_alliteration(metrics)
        }

    def _analyze_punctuation(self, writing_style):
        """Analyzes punctuation patterns"""
        metrics = writing_style.get('punctuation_metrics', {})
        return {
            'style': self._determine_punctuation_style(metrics),
            'emphasis_level': self._calculate_emphasis(metrics),
            'formality_indicators': self._analyze_formal_punctuation(metrics)
        }

    def generate_prompt(self, relevant_data, sentiment, writing_style):
        """
        Generates a highly personalized prompt combining context and linguistic analysis.
        
        Args:
            relevant_data (str): Retrieved relevant data
            sentiment (str): Sentiment analysis result
            writing_style (dict): Writing style metrics
            
        Returns:
            str: Generated personalized prompt
        """
        # Comprehensive style analysis
        style_analysis = self._analyze_writing_style(writing_style)
        
        # Context and key points extraction
        context_type = self._determine_context_type(relevant_data, style_analysis)
        key_points = self._extract_key_points(relevant_data)
        
        # Template selection
        template_name = self._determine_style_template(
            context_type,
            sentiment,
            style_analysis
        )
        template = self.style_patterns[template_name]['template']
        
        # Generate enhanced style notes
        style_notes = self._generate_enhanced_style_notes(style_analysis)
        
        # Format key points with linguistic optimization
        formatted_key_points = self._format_key_points_with_style(
            key_points,
            style_analysis
        )
        
        # Generate and optimize prompt
        prompt = template.format(
            context=relevant_data,
            sentiment=sentiment,
            key_points="\n".join(formatted_key_points),
            style_notes="\n".join(style_notes)
        )
        
        return self._optimize_final_prompt(prompt, style_analysis)

    def add_context_pattern(self, name, indicators, topics, style_weight):
        """Original method for adding context patterns"""
        self.context_patterns[name] = {
            'indicators': indicators,
            'topics': topics,
            'style_weight': style_weight
        }

    def add_style_pattern(self, name, template, emoji_set, formality_threshold):
        """Original method for adding style patterns"""
        self.style_patterns[name] = {
            'template': template,
            'emoji_set': emoji_set,
            'formality_threshold': formality_threshold
        }

    def add_linguistic_pattern(self, category, name, pattern_data):
        """
        Adds a new linguistic pattern for analysis.
        
        Args:
            category (str): Pattern category (vocabulary, sentence, literary, punctuation)
            name (str): Pattern name
            pattern_data (dict): Pattern configuration
        """
        if category in self.linguistic_patterns:
            self.linguistic_patterns[category][name] = pattern_data

if __name__ == "__main__":
    # Initialize the generator
    generator = PromptGenerator()

    # Example usage with enhanced writing style metrics
    relevant_data = "User is having trouble with their Python code and seems frustrated. They've been trying to debug for hours."
    sentiment = "frustrated"
    writing_style = {
        'formality': 0.3,
        'emoji_frequency': 0.7,
        'sentence_length': 12,
        'vocabulary_complexity': 0.4,
        'vocabulary_metrics': {
            'domain_terms': ['Python', 'debug', 'code'],
            'complexity_level': 'intermediate'
        },
        'sentence_metrics': {
            'average_length': 15,
            'pattern_distribution': {'simple': 0.6, 'complex': 0.4}
        },
        'literary_metrics': {
            'metaphor_count': 2,
            'simile_count': 1
        },
        'punctuation_metrics': {
            'emphasis_marks': 3,
            'formal_markers': 1
        }
    }

    # Generate enhanced prompt
    prompt = generator.generate_prompt(relevant_data, sentiment, writing_style)