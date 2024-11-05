import logging
import hashlib
import json
import os
import random
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque

class AdvancedHumanChainManager:
    """
    Advanced manager for generating highly natural human-like thought processes with
    emotional awareness, cognitive biases, personal experiences, and adaptive personality.
    """
    
    MAX_CACHE_SIZE = 1000  # Maximum number of cached thoughts
    
    def __init__(self, model, cache_dir="cache/chains", personality_type="balanced"):
        """
        Initialize with enhanced personality and emotional awareness.
        
        Args:
            model: The fine-tuned language model
            cache_dir (str): Cache directory path
            personality_type (str): Personality archetype to use
        """
        self.model = model
        self.cache_dir = cache_dir
        self._init_personality(personality_type)
        self.emotion_state = self._create_emotion_state()
        os.makedirs(cache_dir, exist_ok=True)
        
        # Track topic familiarity
        self.topic_exposure = defaultdict(int)
        self.confidence_levels = defaultdict(float)
        
        # Initialize cache with deque for eviction
        self.cache = {}
        self.cache_order = deque()
        
    def _init_personality(self, personality_type: str):
        """
        Initialize personality traits and speaking patterns.
        """
        self.personality = {
            "thinking_style": {
                "analytical": random.uniform(0.4, 0.8),
                "intuitive": random.uniform(0.3, 0.7),
                "cautious": random.uniform(0.3, 0.7)
            },
            "expression_patterns": {
                "detail_orientation": random.uniform(0.4, 0.8),
                "emotional_expression": random.uniform(0.3, 0.7),
                "humor_tendency": random.uniform(0.2, 0.5)
            }
        }
        
        # Dynamic language patterns
        self.language_patterns = {
            "reflective": [
                "You know, this reminds me of...",
                "I've been thinking about this...",
                "From my experience...",
                "This is interesting because..."
            ],
            "analytical": [
                "If we break this down...",
                "Looking at this systematically...",
                "Let's consider the key factors...",
                "What stands out to me is..."
            ],
            "uncertain": [
                "I'm not entirely sure, but...",
                "This is just my interpretation...",
                "I could be wrong, but...",
                "From what I understand..."
            ],
            "metacognitive": [
                "Let me think this through...",
                "I'm reconsidering my approach...",
                "Something doesn't quite add up...",
                "Building on that thought..."
            ],
            "emotional": [
                "I'm quite excited about...",
                "This is actually quite challenging...",
                "I feel strongly that...",
                "I'm a bit hesitant about..."
            ]
        }
        
        # Cognitive biases to simulate
        self.cognitive_biases = {
            "confirmation_bias": random.uniform(0.2, 0.4),
            "anchoring_bias": random.uniform(0.2, 0.4),
            "availability_bias": random.uniform(0.2, 0.4),
            "hindsight_bias": random.uniform(0.1, 0.3),
            "dunning_kruger_effect": random.uniform(0.1, 0.3)
        }
        
    def _create_emotion_state(self) -> Dict:
        """
        Create an emotional state tracker.
        """
        return {
            "interest_level": random.uniform(0.5, 1.0),
            "certainty": random.uniform(0.4, 0.8),
            "enthusiasm": random.uniform(0.3, 0.7),
            "recent_emotions": deque(maxlen=10)  # Store recent emotions with timestamps
        }
    
    def _analyze_topic_complexity(self, prompt: str) -> float:
        """
        Analyze the complexity of the topic to adjust response style.
        """
        # Simplified complexity analysis
        complexity_indicators = {
            'technical_terms': len(re.findall(r'\b[a-zA-Z]{10,}\b', prompt)),
            'question_depth': prompt.count('?'),
            'abstract_concepts': len(re.findall(r'\b(theory|concept|framework|system|process)\b', prompt.lower())),
        }
        
        return min(1.0, sum(complexity_indicators.values()) / 10)
    
    def _generate_personal_experience(self, topic: str) -> Optional[str]:
        """
        Generate relevant personal experience or analogy.
        """
        experience_templates = [
            "This reminds me of a similar situation where...",
            "I once encountered something like this when...",
            "It's kind of like when you're...",
            "This is similar to..."
        ]
        
        if random.random() < self.personality["thinking_style"]["intuitive"]:
            return f"{random.choice(experience_templates)} {self._generate_analogy(topic)}"
        return None
    
    def _generate_analogy(self, topic: str) -> str:
        """
        Generate a relevant analogy based on the topic.
        """
        common_experiences = {
            "problem": "trying to solve a puzzle",
            "decision": "choosing a path at a crossroad",
            "learning": "learning to ride a bicycle",
            "growth": "watching a plant grow",
            "complexity": "cooking a complex recipe"
        }
        
        return common_experiences.get(self._extract_core_concept(topic), 
                                   "dealing with something new and unexpected")
    
    def _extract_core_concept(self, text: str) -> str:
        """
        Extract the core concept from text.
        """
        concept_patterns = {
            "problem": r'\b(issue|problem|challenge|difficulty)\b',
            "decision": r'\b(choice|decision|option|alternative)\b',
            "learning": r'\b(learn|understand|grasp|comprehend)\b',
            "growth": r'\b(grow|develop|improve|progress)\b',
            "complexity": r'\b(complex|complicated|intricate|detailed)\b'
        }
        
        for concept, pattern in concept_patterns.items():
            if re.search(pattern, text.lower()):
                return concept
        return "general"
    
    def _adjust_confidence(self, topic: str, success: bool):
        """
        Adjust confidence levels based on successful interactions.
        """
        current_confidence = self.confidence_levels[topic]
        if success:
            self.confidence_levels[topic] = min(1.0, current_confidence + 0.1)
        else:
            self.confidence_levels[topic] = max(0.2, current_confidence - 0.1)
    
    def _add_human_elements(self, thought_process: str, topic_complexity: float) -> str:
        """
        Add sophisticated human-like elements to the thought process.
        """
        # Add metacognitive elements
        if random.random() < self.personality["thinking_style"]["analytical"]:
            thought_process = f"{random.choice(self.language_patterns['metacognitive'])} {thought_process}"
        
        # Add emotional elements based on complexity
        if topic_complexity > 0.7 and random.random() < self.personality["expression_patterns"]["emotional_expression"]:
            emotional_response = random.choice(self.language_patterns['emotional'])
            thought_process = f"{emotional_response} {thought_process}"
        
        # Add personal experience or analogy
        if random.random() < 0.3:
            experience = self._generate_personal_experience(thought_process)
            if experience:
                thought_process += f"\n\n{experience}"
        
        # Add self-correction based on confidence
        if random.random() < (1 - self.emotion_state["certainty"]):
            correction_points = [
                "Actually, let me revise that thought...",
                "On second thought...",
                "Wait, I should clarify something...",
                "Let me refine that idea..."
            ]
            thought_process = f"{random.choice(correction_points)} {thought_process}"
        
        # Add hedging based on topic complexity
        if topic_complexity > 0.5:
            hedging_phrases = [
                "From what I understand...",
                "It seems to me that...",
                "Based on my current knowledge...",
                "I think..."
            ]
            thought_process = f"{random.choice(hedging_phrases)} {thought_process}"
        
        return thought_process
    
    def generate_thought(self, prompt: str, history: List[Dict[str, str]]) -> str:
        """
        Generate a sophisticated human-like thought process.
        """
        topic_complexity = self._analyze_topic_complexity(prompt)
        cache_key = self._generate_cache_key(prompt, history)
        
        # Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Update topic exposure
        topic = self._extract_core_concept(prompt)
        self.topic_exposure[topic] += 1
        
        try:
            # Generate initial thought
            thought_prompt = self._construct_thought_prompt(prompt, history, topic_complexity)
            
            inputs = self.model.tokenizer(thought_prompt, return_tensors="pt", truncation=True)
            outputs = self.model.fine_tuned_model.generate(
                inputs["input_ids"],
                max_length=400,
                temperature=0.8 + (random.random() * 0.2),  # Dynamic temperature
                top_p=0.92,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=self.model.tokenizer.pad_token_id,
                eos_token_id=self.model.tokenizer.eos_token_id
            )
            
            thought_process = self.model.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract core thought process
            if "My thoughts:" in thought_process:
                thought_process = thought_process.split("My thoughts:")[1].strip()
            
            # Enhance with human elements
            final_thought = self._add_human_elements(thought_process, topic_complexity)
            
            # Add cognitive biases
            if random.random() < self.cognitive_biases["confirmation_bias"]:
                final_thought = self._add_confirmation_bias(final_thought, history)
            if random.random() < self.cognitive_biases["hindsight_bias"]:
                final_thought = self._add_hindsight_bias(final_thought)
            if random.random() < self.cognitive_biases["dunning_kruger_effect"]:
                final_thought = self._add_dunning_kruger_bias(final_thought)
            
            # Update emotional state
            self._update_emotion_state(topic_complexity, len(final_thought))
            
            # Cache and return
            self._save_to_cache(cache_key, final_thought)
            self._adjust_confidence(topic, True)
            
            return final_thought
            
        except Exception as e:
            logging.error(f"Error in thought generation: {e}")
            self._adjust_confidence(topic, False)
            return self._generate_fallback_response(topic_complexity)
    
    def _construct_thought_prompt(self, prompt: str, history: List[Dict[str, str]], complexity: float) -> str:
        """
        Construct a context-aware thought prompt.
        """
        context = self._extract_relevant_context(history)
        style = random.choice(['reflective', 'analytical', 'uncertain'])
        
        return (
            f"{random.choice(self.language_patterns[style])}\n\n"
            f"Context: {context}\n"
            f"Current topic complexity: {'high' if complexity > 0.7 else 'moderate' if complexity > 0.4 else 'basic'}\n"
            f"Question: {prompt}\n\n"
            "My thoughts:"
        )
    
    def _update_emotion_state(self, complexity: float, response_length: int):
        """
        Update emotional state based on interaction.
        """
        # Update interest level
        self.emotion_state["interest_level"] = min(1.0, self.emotion_state["interest_level"] + (complexity * 0.05))
        # Decay certainty over time
        self.emotion_state["certainty"] = max(0.2, self.emotion_state["certainty"] - (complexity * 0.03))
        # Update enthusiasm
        self.emotion_state["enthusiasm"] = min(1.0, self.emotion_state["enthusiasm"] + (random.random() * 0.05))
        
        # Decay older emotions
        current_time = datetime.now()
        while self.emotion_state["recent_emotions"]:
            emotion, timestamp = self.emotion_state["recent_emotions"][0]
            if current_time - timestamp > timedelta(minutes=5):
                self.emotion_state["recent_emotions"].popleft()
                logging.debug(f"Removed expired emotion: {emotion}")
            else:
                break
        
        # Add new emotion if response is long
        if response_length > 300:
            new_emotion = ("engaged", current_time)
            self.emotion_state["recent_emotions"].append(new_emotion)
            logging.debug(f"Added new emotion: {new_emotion}")
    
    def _add_confirmation_bias(self, thought: str, history: List[Dict[str, str]]) -> str:
        """
        Add subtle confirmation bias based on conversation history.
        """
        if history:
            recent_opinion = history[-1]['text']
            if random.random() < self.cognitive_biases["confirmation_bias"]:
                return f"{thought}\n\nThis aligns with what we discussed earlier about {recent_opinion[:30]}..."
        return thought
    
    def _add_hindsight_bias(self, thought: str) -> str:
        """
        Add hindsight bias to the thought.
        """
        hindsight_phrases = [
            "In hindsight, it's clear that...",
            "Looking back, I should have...",
            "Now that I think about it, I realize...",
            "Upon reflection, it seems..."
        ]
        return f"{random.choice(hindsight_phrases)} {thought}"
    
    def _add_dunning_kruger_bias(self, thought: str) -> str:
        """
        Add Dunning-Kruger effect bias to the thought.
        """
        overconfidence_phrases = [
            "I am completely certain that...",
            "There's no doubt in my mind that...",
            "I fully believe that...",
            "I'm absolutely sure that..."
        ]
        return f"{random.choice(overconfidence_phrases)} {thought}"
    
    def _save_to_cache(self, key: str, value: str):
        """
        Save a thought to the cache with eviction if necessary.
        """
        if key not in self.cache:
            if len(self.cache_order) >= self.MAX_CACHE_SIZE:
                oldest_key = self.cache_order.popleft()
                del self.cache[oldest_key]
                logging.debug(f"Evicted cache key: {oldest_key}")
            self.cache_order.append(key)
        self.cache[key] = value
    
    def _generate_cache_key(self, prompt: str, history: List[Dict[str, str]]) -> str:
        """
        Generate a unique cache key based on prompt and history.
        """
        history_text = ''.join([entry['text'] for entry in history])
        key_source = prompt + history_text
        return hashlib.sha256(key_source.encode('utf-8')).hexdigest()
    
    def _extract_relevant_context(self, history: List[Dict[str, str]]) -> str:
        """
        Extract relevant context from the conversation history.
        """
        # For simplicity, include the last 3 messages
        relevant_history = history[-3:]
        return ' '.join([entry['text'] for entry in relevant_history])
    
    def _generate_fallback_response(self, complexity: float) -> str:
        """
        Generate a natural fallback response based on complexity.
        """
        if complexity > 0.7:
            return random.choice([
                "I need to think about this more carefully...",
                "This is quite complex. Let me gather my thoughts...",
                "I'm still processing this challenging question..."
            ])
        return random.choice([
            "Let me collect my thoughts on this...",
            "I'm thinking through this...",
            "Give me a moment to organize my thoughts..."
        ])