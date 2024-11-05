import os
from typing import Optional, List, Dict, Union
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import Dataset
import logging
import re
import random
import json
from dataclasses import dataclass
from nltk.tokenize import sent_tokenize
import numpy as np

@dataclass
class PersonalityConfig:
    """Configuration for model personality traits."""
    formality: float = 0.5  # 0: very casual, 1: very formal
    expressiveness: float = 0.5  # 0: reserved, 1: expressive
    humor: float = 0.3  # 0: serious, 1: humorous
    empathy: float = 0.7  # 0: neutral, 1: highly empathetic
    verbosity: float = 0.5  # 0: concise, 1: detailed

class TextProcessor:
    """Handles text processing and augmentation for more natural language."""
    
    def __init__(self):
        self.fillers = [
            "you know", "like", "I mean", "actually", "basically",
            "sort of", "kind of", "in a way", "well"
        ]
        self.interjections = [
            "hmm", "oh", "ah", "wow", "right",
            "yeah", "okay", "hey", "cool"
        ]
        self.hedges = [
            "I think", "probably", "maybe", "might", "could",
            "seems like", "appears to be", "from what I understand"
        ]
        
    def add_natural_elements(self, text: str, personality: PersonalityConfig) -> str:
        """Adds natural language elements based on personality settings."""
        sentences = sent_tokenize(text)
        processed_sentences = []
        
        for sentence in sentences:
            if random.random() < personality.expressiveness:
                # Add interjections at start of some sentences
                if random.random() < 0.3:
                    sentence = f"{random.choice(self.interjections)}, {sentence.lower()}"
            
            if random.random() < (1 - personality.formality):
                # Add fillers for more casual speech
                words = sentence.split()
                if len(words) > 5 and random.random() < 0.3:
                    insert_pos = random.randint(1, len(words) - 1)
                    words.insert(insert_pos, random.choice(self.fillers))
                    sentence = " ".join(words)
            
            if random.random() < personality.empathy:
                # Add hedges for more thoughtful/empathetic responses
                if not any(hedge in sentence.lower() for hedge in self.hedges):
                    if random.random() < 0.3:
                        sentence = f"{random.choice(self.hedges)}, {sentence.lower()}"
            
            processed_sentences.append(sentence)
        
        return " ".join(processed_sentences)

class DialogueContext:
    """Manages conversation context and flow."""
    
    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.conversation_history = []
        self.speaker_patterns = {}
        
    def add_utterance(self, text: str, speaker: str):
        """Adds an utterance to the conversation history."""
        self.conversation_history.append({"speaker": speaker, "text": text})
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
        self._update_speaker_patterns(text, speaker)
    
    def _update_speaker_patterns(self, text: str, speaker: str):
        """Analyzes and updates speaking patterns for each participant."""
        if speaker not in self.speaker_patterns:
            self.speaker_patterns[speaker] = {
                "avg_response_length": [],
                "favorite_words": {},
                "sentence_starters": {}
            }
        
        # Update statistics
        words = text.split()
        self.speaker_patterns[speaker]["avg_response_length"].append(len(words))
        
        # Track frequently used words
        for word in words:
            self.speaker_patterns[speaker]["favorite_words"][word] = \
                self.speaker_patterns[speaker]["favorite_words"].get(word, 0) + 1
        
        # Track sentence starters
        sentences = sent_tokenize(text)
        for sentence in sentences:
            first_word = sentence.split()[0].lower()
            self.speaker_patterns[speaker]["sentence_starters"][first_word] = \
                self.speaker_patterns[speaker]["sentence_starters"].get(first_word, 0) + 1

class EnhancedFineTuner(FineTuner):
    """Enhanced version of FineTuner with human-like language capabilities."""
    
    def __init__(
        self,
        model_name: str,
        personality_config: Optional[PersonalityConfig] = None,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.personality = personality_config or PersonalityConfig()
        self.text_processor = TextProcessor()
        self.dialogue_context = DialogueContext()
        
    def prepare_training_sample(self, text: str, include_context: bool = True) -> str:
        """Prepares a single training sample with context and personality."""
        processed_text = self.text_processor.add_natural_elements(text, self.personality)
        
        if include_context and self.dialogue_context.conversation_history:
            # Add relevant context from conversation history
            context = "\n".join([
                f"{entry['speaker']}: {entry['text']}"
                for entry in self.dialogue_context.conversation_history[-2:]  # Last 2 exchanges
            ])
            processed_text = f"{context}\nAI: {processed_text}"
            
        return processed_text
    
    def prepare_dataset(self, texts: List[str]) -> Dataset:
        """Prepares the training dataset with enhanced human-like features."""
        processed_texts = []
        
        for text in texts:
            if self.do_clean:
                text = self.clean_text(text)
            
            # Process text to add human-like elements
            processed_text = self.prepare_training_sample(text)
            processed_texts.append(processed_text)
            
            if self.do_augment:
                # Create variations with different personality settings
                varied_personality = PersonalityConfig(
                    formality=random.uniform(0, 1),
                    expressiveness=random.uniform(0, 1),
                    humor=random.uniform(0, 1),
                    empathy=random.uniform(0, 1),
                    verbosity=random.uniform(0, 1)
                )
                varied_text = self.prepare_training_sample(
                    text,
                    include_context=random.choice([True, False])
                )
                processed_texts.append(varied_text)
        
        return super().prepare_dataset(processed_texts)
    
    def generate_response(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """Generates a response with personality-aware processing."""
        # Add context from recent conversation
        if self.dialogue_context.conversation_history:
            context_prompt = "\n".join([
                f"{entry['speaker']}: {entry['text']}"
                for entry in self.dialogue_context.conversation_history[-2:]
            ])
            prompt = f"{context_prompt}\nHuman: {prompt}\nAI:"
        
        # Generate base response
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.fine_tuned_model.generate(
            inputs["input_ids"],
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Process response to match personality and add natural elements
        processed_response = self.text_processor.add_natural_elements(
            response,
            self.personality
        )
        
        # Update conversation context
        self.dialogue_context.add_utterance(prompt, "Human")
        self.dialogue_context.add_utterance(processed_response, "AI")
        
        return processed_response

    def save_personality(self, filepath: str):
        """Saves the current personality configuration."""
        with open(filepath, 'w') as f:
            json.dump(vars(self.personality), f)
    
    def load_personality(self, filepath: str):
        """Loads a personality configuration."""
        with open(filepath, 'r') as f:
            personality_dict = json.load(f)
            self.personality = PersonalityConfig(**personality_dict)

if __name__ == "__main__":
    personality = PersonalityConfig(
    formality=0.3,      # More casual
    expressiveness=0.8,  # Quite expressive
    humor=0.6,          # Moderately humorous
    empathy=0.9,        # Highly empathetic
    verbosity=0.4       # Somewhat concise
    )

    tuner = EnhancedFineTuner(
        model_name="your-base-model",
        personality_config=personality
    )

    # Fine-tune with your training texts
    tuner.fine_tune(training_texts)

    # Generate a response
    response = tuner.generate_response("How are you today?")