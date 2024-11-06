import os
from typing import Optional, List, Dict, Union, Type
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
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
        model_type: Optional[str] = None,
        personality_config: Optional[PersonalityConfig] = None,
        tokenizer_name: Optional[str] = None,
        **kwargs
    ):
        """
        Initializes the EnhancedFineTuner.
        
        Args:
            model_name (str): Name of the pre-trained model to fine-tune.
            model_type (Optional[str]): Specific type of the model if auto-detection fails.
            personality_config (Optional[PersonalityConfig]): Configuration for personality traits.
            tokenizer_name (Optional[str]): Name of the tokenizer to use. Defaults to model_name.
            **kwargs: Additional keyword arguments for FineTuner.
        """
        super().__init__(model_name, **kwargs)
        self.personality = personality_config or PersonalityConfig()
        self.text_processor = TextProcessor()
        self.dialogue_context = DialogueContext()
        
        # Dynamically load the model and tokenizer
        self.model_type = model_type
        self.tokenizer_name = tokenizer_name or model_name
        
        try:
            if self.model_type:
                self.model = AutoModel.from_pretrained(self.model_name)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            logging.info(f"Loaded model '{self.model_name}' and tokenizer '{self.tokenizer_name}'.")
        except Exception as e:
            logging.error(f"Error loading model or tokenizer: {e}")
            raise e
        
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
                # Update personality for augmentation
                original_personality = self.personality
                self.personality = varied_personality
                processed_texts.append(varied_text)
                # Restore original personality
                self.personality = original_personality
        
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
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logging.error(f"Error during generation: {e}")
            response = "I'm sorry, I couldn't generate a response at this time."
        
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
        try:
            with open(filepath, 'w') as f:
                json.dump(vars(self.personality), f)
            logging.info(f"Personality configuration saved to {filepath}.")
        except Exception as e:
            logging.error(f"Failed to save personality configuration: {e}")
    
    def load_personality(self, filepath: str):
        """Loads a personality configuration."""
        try:
            with open(filepath, 'r') as f:
                personality_dict = json.load(f)
                self.personality = PersonalityConfig(**personality_dict)
            logging.info(f"Personality configuration loaded from {filepath}.")
        except Exception as e:
            logging.error(f"Failed to load personality configuration: {e}")

    def fine_tune(self, training_texts: List[str]) -> AutoModelForCausalLM:
        """Fine-tunes the model with the provided training texts and returns the fine-tuned model."""
        dataset = self.prepare_dataset(training_texts)
        
        training_args = TrainingArguments(
            output_dir="./fine_tuned_model",
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            learning_rate=self.learning_rate,
            logging_dir="./logs",
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="steps",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        trainer.train()
        
        # Save the fine-tuned model
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        logging.info("Fine-tuning completed and model saved.")
        
        return self.model  # Return the fine-tuned model

if __name__ == "__main__":
    personality = PersonalityConfig(
    formality=0.3,      # More casual
    expressiveness=0.8,  # Quite expressive
    humor=0.6,          # Moderately humorous
    empathy=0.9,        # Highly empathetic
    verbosity=0.4       # Somewhat concise
    )

    tuner = EnhancedFineTuner(
        model_name="gpt2",
        personality_config=personality
    )

    # Example training texts
    training_texts = [
        "Hello! How can I assist you today?",
        "I'm here to help with any questions you might have."
    ]

    # Fine-tune with your training texts
    try:
        fine_tuned_model = tuner.fine_tune(training_texts)  # Capture the returned model
        logging.info("Fine-tuning successful.")
    except Exception as e:
        logging.error(f"Fine-tuning failed: {e}")
        exit(1)

    # Generate a response
    if fine_tuned_model:
        try:
            response = fine_tuned_model.generate_response("How are you today?")
            print(response)
        except Exception as e:
            logging.error(f"Failed to generate response: {e}")
    else:
        logging.error("Fine-tuned model is not available.")