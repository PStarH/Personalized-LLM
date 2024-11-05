import os
import json
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import nltk
from datetime import datetime
from hashlib import sha256

@dataclass
class RetrievedPassage:
    """Dataclass to store retrieved passage information"""
    content: str
    similarity_score: float
    metadata: Dict
    passage_id: str

class EnhancedRetriever:
    """
    Enhanced retriever for RAG applications with improved context handling,
    metadata management, and retrieval strategies.
    """

    def __init__(
        self, 
        embedding_model_name: str = "all-MiniLM-L6-v2",
        embedding_storage_path: str = "embeddings/history_embeddings.npy",
        metadata_storage_path: str = "embeddings/metadata.json",
        similarity_threshold: float = 0.6,
        max_tokens_per_passage: int = 300
    ):
        """
        Initialize the enhanced retriever with improved configuration.

        Args:
            embedding_model_name (str): Name of the SentenceTransformer model
            embedding_storage_path (str): Path to store embeddings (.npy)
            metadata_storage_path (str): Path to store metadata (.json)
            similarity_threshold (float): Minimum similarity score for retrieval
            max_tokens_per_passage (int): Maximum tokens per passage
        """
        self.embedding_model_name = embedding_model_name
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
            logging.info(f"Loaded embedding model: {embedding_model_name}")
        except Exception as e:
            logging.error(f"Failed to load embedding model '{embedding_model_name}': {e}")
            raise e
        self.embedding_storage_path = embedding_storage_path
        self.metadata_storage_path = metadata_storage_path
        self.similarity_threshold = similarity_threshold
        self.max_tokens_per_passage = max_tokens_per_passage
        
        # Initialize storage
        self.passages: List[str] = []
        self.embeddings: np.ndarray = np.array([])
        self.metadata: List[Dict] = []
        self.passage_ids: List[str] = []
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            logging.info("Downloaded NLTK 'punkt' tokenizer.")
        except Exception as e:
            logging.warning(f"Failed to download NLTK data: {e}")
        
        self._load_embeddings()

    def _generate_passage_id(self, passage: str) -> str:
        """Generate a unique ID for a passage."""
        return sha256(passage.encode()).hexdigest()[:16]

    def _load_embeddings(self):
        """Load embeddings and metadata from storage."""
        if os.path.exists(self.embedding_storage_path) and os.path.exists(self.metadata_storage_path):
            try:
                self.embeddings = np.load(self.embedding_storage_path)
                with open(self.metadata_storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.passages = data.get('passages', [])
                    self.metadata = data.get('metadata', [])
                    self.passage_ids = data.get('passage_ids', [])
                logging.info(f"Loaded {len(self.passages)} passages with metadata.")
            except Exception as e:
                logging.error(f"Failed to load embeddings or metadata: {e}")
        else:
            logging.info("No existing embeddings or metadata found. Starting fresh.")

    def _save_embeddings(self):
        """Save embeddings and metadata to storage."""
        try:
            os.makedirs(os.path.dirname(self.embedding_storage_path), exist_ok=True)
            np.save(self.embedding_storage_path, self.embeddings)
            with open(self.metadata_storage_path, 'w', encoding='utf-8') as f:
                data = {
                    'passages': self.passages,
                    'metadata': self.metadata,
                    'passage_ids': self.passage_ids
                }
                json.dump(data, f, ensure_ascii=False, indent=4)
            logging.info(f"Saved {len(self.passages)} passages and embeddings.")
        except Exception as e:
            logging.error(f"Failed to save embeddings or metadata: {e}")

    def _smart_split_text(self, text: str) -> List[Tuple[str, Dict]]:
        """
        Intelligently split text into passages while preserving context.
        
        Returns:
            List[Tuple[str, Dict]]: List of (passage, metadata) tuples
        """
        sentences = sent_tokenize(text)
        passages = []
        current_passage = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(sentence.split())
            
            if current_tokens + sentence_tokens > self.max_tokens_per_passage and current_passage:
                # Join the current passage and create metadata
                passage_text = ' '.join(current_passage)
                metadata = {
                    'token_count': current_tokens,
                    'timestamp': datetime.now().isoformat(),
                    'source_type': 'text',
                }
                passages.append((passage_text, metadata))
                
                # Reset for next passage
                current_passage = [sentence]
                current_tokens = sentence_tokens
            else:
                current_passage.append(sentence)
                current_tokens += sentence_tokens
        
        # Handle the last passage
        if current_passage:
            passage_text = ' '.join(current_passage)
            metadata = {
                'token_count': current_tokens,
                'timestamp': datetime.now().isoformat(),
                'source_type': 'text',
            }
            passages.append((passage_text, metadata))
        
        return passages

    def add_document(self, content: str, source_metadata: Optional[Dict] = None):
        """
        Add a new document with metadata to the retriever.
        
        Args:
            content (str): Document content
            source_metadata (Dict, optional): Additional metadata about the source
        """
        passages_with_metadata = self._smart_split_text(content)
        
        for passage, metadata in passages_with_metadata:
            # Combine with source metadata
            if source_metadata:
                metadata.update(source_metadata)
            
            passage_id = self._generate_passage_id(passage)
            
            # Skip if passage already exists
            if passage_id in self.passage_ids:
                logging.info(f"Passage already exists with ID {passage_id}. Skipping.")
                continue
            
            # Compute embedding
            try:
                embedding = self.embedding_model.encode(passage)
            except Exception as e:
                logging.error(f"Failed to encode passage: {e}")
                continue
            
            # Add to storage
            self.passages.append(passage)
            if self.embeddings.size:
                self.embeddings = np.vstack([self.embeddings, embedding])
            else:
                self.embeddings = np.array([embedding])
            self.metadata.append(metadata)
            self.passage_ids.append(passage_id)
            logging.info(f"Added passage ID {passage_id}.")

        self._save_embeddings()

    def update_passage(self, passage_id: str, new_content: Optional[str] = None, new_metadata: Optional[Dict] = None):
        """
        Update an existing passage's content and/or metadata.
        
        Args:
            passage_id (str): ID of the passage to update
            new_content (str, optional): New content for the passage
            new_metadata (Dict, optional): New metadata to update
        """
        if passage_id not in self.passage_ids:
            logging.warning(f"Passage ID {passage_id} not found.")
            return
        
        idx = self.passage_ids.index(passage_id)
        
        if new_content:
            old_content = self.passages[idx]
            self.passages[idx] = new_content
            try:
                new_embedding = self.embedding_model.encode(new_content)
                self.embeddings[idx] = new_embedding
                logging.info(f"Updated content and embedding for passage ID {passage_id}.")
            except Exception as e:
                logging.error(f"Failed to encode updated passage: {e}")
                # Revert to old content if encoding fails
                self.passages[idx] = old_content
        if new_metadata:
            self.metadata[idx].update(new_metadata)
            logging.info(f"Updated metadata for passage ID {passage_id}.")
        
        self._save_embeddings()

    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        filter_criteria: Optional[Dict] = None
    ) -> List[RetrievedPassage]:
        """
        Retrieve relevant passages with enhanced filtering and ranking.
        
        Args:
            query (str): Query text
            top_k (int): Number of passages to retrieve
            filter_criteria (Dict, optional): Metadata-based filters
            
        Returns:
            List[RetrievedPassage]: Retrieved passages with metadata and scores
        """
        if not self.passages:
            logging.info("No passages available for retrieval.")
            return []

        # Encode query
        try:
            query_embedding = self.embedding_model.encode(query)
            logging.info("Encoded query successfully.")
        except Exception as e:
            logging.error(f"Failed to encode query: {e}")
            return []
        
        # Calculate similarities
        try:
            similarities = cosine_similarity([query_embedding], self.embeddings)[0]
            logging.info("Computed cosine similarities.")
        except Exception as e:
            logging.error(f"Failed to compute cosine similarities: {e}")
            return []
        
        # Apply similarity threshold
        valid_indices = np.where(similarities >= self.similarity_threshold)[0]
        
        # Apply metadata filters if specified
        if filter_criteria:
            filtered_indices = []
            for idx in valid_indices:
                metadata = self.metadata[idx]
                if all(metadata.get(k) == v for k, v in filter_criteria.items()):
                    filtered_indices.append(idx)
            valid_indices = filtered_indices
            logging.info(f"Applied filter criteria. {len(valid_indices)} passages match the filters.")
        
        if len(valid_indices) == 0:
            logging.info("No passages meet the similarity threshold and filter criteria.")
            return []
        
        # Sort by similarity
        ranked_indices = valid_indices[np.argsort(similarities[valid_indices])[-top_k:][::-1]]
        
        # Prepare results
        results = []
        for idx in ranked_indices:
            result = RetrievedPassage(
                content=self.passages[idx],
                similarity_score=float(similarities[idx]),
                metadata=self.metadata[idx],
                passage_id=self.passage_ids[idx]
            )
            results.append(result)
            logging.debug(f"Retrieved passage ID {result.passage_id} with similarity {result.similarity_score:.2f}.")
        
        return results

    def get_contextual_prompt(self, query: str, top_k: int = 3) -> str:
        """
        Generate a contextual prompt for the LLM using retrieved passages.
        
        Args:
            query (str): User query
            top_k (int): Number of passages to include
            
        Returns:
            str: Formatted prompt with context
        """
        retrieved_passages = self.retrieve(query, top_k=top_k)
        
        if not retrieved_passages:
            return f"Query: {query}\nNo relevant context found."
        
        context_str = "\n\nRelevant Context:\n"
        for i, passage in enumerate(retrieved_passages, 1):
            context_str += f"\nPassage {i} (Similarity: {passage.similarity_score:.2f}):\n{passage.content}\n"
        
        prompt = f"""Query: {query}

Available Context:
{context_str}

Please provide a comprehensive response using the provided context. If the context doesn't fully address the query, please indicate any gaps in the available information."""
        
        logging.info("Generated contextual prompt.")
        return prompt

    def clear_data(self, filter_criteria: Optional[Dict] = None):
        """
        Clear data with optional filtering.
        
        Args:
            filter_criteria (Dict, optional): Metadata-based filters for selective clearing
        """
        if not filter_criteria:
            self.passages = []
            self.embeddings = np.array([])
            self.metadata = []
            self.passage_ids = []
            if os.path.exists(self.embedding_storage_path):
                os.remove(self.embedding_storage_path)
                logging.info(f"Removed embedding file at {self.embedding_storage_path}.")
            if os.path.exists(self.metadata_storage_path):
                os.remove(self.metadata_storage_path)
                logging.info(f"Removed metadata file at {self.metadata_storage_path}.")
            logging.info("All data cleared.")
        else:
            indices_to_keep = []
            for i, metadata in enumerate(self.metadata):
                if not all(metadata.get(k) == v for k, v in filter_criteria.items()):
                    indices_to_keep.append(i)
            
            self.passages = [self.passages[i] for i in indices_to_keep]
            self.embeddings = self.embeddings[indices_to_keep]
            self.metadata = [self.metadata[i] for i in indices_to_keep]
            self.passage_ids = [self.passage_ids[i] for i in indices_to_keep]
            self._save_embeddings()
            logging.info("Data cleared based on filter criteria.")

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize retriever
    retriever = EnhancedRetriever()
    
    # Example document with metadata
    document = """
    Artificial Intelligence (AI) is transforming the world in unprecedented ways.
    Machine learning, a subset of AI, focuses on data-driven learning and pattern recognition.
    Deep learning systems utilize neural networks with multiple layers for complex tasks.
    Natural Language Processing enables machines to understand and process human language effectively.
    Modern AI applications span across healthcare, finance, and autonomous systems.
    """
    
    metadata = {
        'source': 'AI textbook',
        'author': 'Dr. Smith',
        'year': 2024,
        'domain': 'artificial intelligence'
    }
    
    # Add document
    retriever.add_document(document, metadata)
    
    # Example query with filtering
    query = "How does machine learning relate to AI?"
    filter_criteria = {'domain': 'artificial intelligence'}
    
    # Get contextual prompt
    prompt = retriever.get_contextual_prompt(query)
    print("\nGenerated Prompt for LLM:")
    print(prompt)
    
    # Retrieve with filtering
    results = retriever.retrieve(query, filter_criteria=filter_criteria)
    print("\nRetrieved Passages:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Passage (Score: {result.similarity_score:.2f}):")
        print(f"Content: {result.content}")
        print(f"Metadata: {result.metadata}")