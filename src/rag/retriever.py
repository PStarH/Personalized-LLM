import os
import json
import logging
import sqlite3
import threading
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
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
        metadata_db_path: str = "embeddings/metadata.db",
        faiss_index_path: str = "embeddings/faiss.index",
        similarity_threshold: float = 0.6,
        max_tokens_per_passage: int = 300,
        faiss_nlist: int = 100  # Number of clusters for IVF
    ):
        """
        Initialize the enhanced retriever with improved configuration.

        Args:
            embedding_model_name (str): Name of the SentenceTransformer model
            embedding_storage_path (str): Path to store embeddings (.npy)
            metadata_db_path (str): Path to store metadata SQLite DB
            faiss_index_path (str): Path to store FAISS index
            similarity_threshold (float): Minimum similarity score for retrieval
            max_tokens_per_passage (int): Maximum tokens per passage
            faiss_nlist (int): Number of clusters for IVF index
        """
        self.embedding_model_name = embedding_model_name
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
            logging.info(f"Loaded embedding model: {embedding_model_name}")
        except Exception as e:
            logging.error(f"Failed to load embedding model '{embedding_model_name}': {e}")
            raise e
        self.embedding_storage_path = embedding_storage_path
        self.metadata_db_path = metadata_db_path
        self.faiss_index_path = faiss_index_path
        self.similarity_threshold = similarity_threshold
        self.max_tokens_per_passage = max_tokens_per_passage
        self.faiss_nlist = faiss_nlist
        
        # Initialize storage
        self.passages: List[str] = []
        self.metadata: List[Dict] = []
        self.passage_ids: List[str] = []
        self.embeddings = None
        self.index = None  # FAISS index
        self.lock = threading.Lock()
        
        # Setup SQLite for metadata
        self._setup_metadata_db()
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            logging.info("Downloaded NLTK 'punkt' tokenizer.")
        except Exception as e:
            logging.warning(f"Failed to download NLTK data: {e}")
        
        self._load_embeddings()

    def _setup_metadata_db(self):
        """Set up the SQLite database for metadata."""
        try:
            os.makedirs(os.path.dirname(self.metadata_db_path), exist_ok=True)
            self.conn = sqlite3.connect(self.metadata_db_path, check_same_thread=False)
            self.cursor = self.conn.cursor()
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    passage_id TEXT PRIMARY KEY,
                    content TEXT,
                    similarity_score REAL,
                    metadata TEXT
                )
            ''')
            self.conn.commit()
            logging.info("Initialized SQLite metadata database.")
        except Exception as e:
            logging.error(f"Failed to set up metadata database: {e}")
            raise e

    def _generate_passage_id(self, passage: str) -> str:
        """Generate a unique ID for a passage."""
        return sha256(passage.encode()).hexdigest()[:16]

    def _load_embeddings(self):
        """Load embeddings and metadata from storage and initialize FAISS index."""
        try:
            if os.path.exists(self.embedding_storage_path) and os.path.exists(self.faiss_index_path):
                self.embeddings = np.load(self.embedding_storage_path).astype('float32')
                self.index = faiss.read_index(self.faiss_index_path)
                logging.info(f"Loaded embeddings from {self.embedding_storage_path} and FAISS index from {self.faiss_index_path}.")
            else:
                self.embeddings = np.empty((0, self.embedding_model.get_sentence_embedding_dimension()), dtype='float32')
                # Initialize FAISS index with IVF
                quantizer = faiss.IndexFlatIP(self.embedding_model.get_sentence_embedding_dimension())
                self.index = faiss.IndexIVFFlat(quantizer, self.embedding_model.get_sentence_embedding_dimension(), self.faiss_nlist, faiss.METRIC_INNER_PRODUCT)
                logging.info("Initialized new FAISS IVF index.")
                if self.embeddings.shape[0] > 0:
                    faiss.normalize_L2(self.embeddings)
                    self.index.train(self.embeddings)
                    self.index.add(self.embeddings)
                    logging.info(f"FAISS index trained and added {self.embeddings.shape[0]} embeddings.")
            
            # Load metadata from SQLite
            self.cursor.execute("SELECT passage_id, content, similarity_score, metadata FROM metadata")
            rows = self.cursor.fetchall()
            for row in rows:
                self.passage_ids.append(row[0])
                self.passages.append(row[1])
                self.metadata.append(json.loads(row[3]))
            logging.info(f"Loaded {len(self.passages)} passages with metadata from database.")
        except Exception as e:
            logging.error(f"Failed to load embeddings or metadata: {e}")
            raise e

    def _save_embeddings(self):
        """Save embeddings and FAISS index to storage."""
        with self.lock:
            try:
                if self.embeddings is not None and self.embeddings.size > 0:
                    os.makedirs(os.path.dirname(self.embedding_storage_path), exist_ok=True)
                    np.save(self.embedding_storage_path, self.embeddings)
                    faiss.write_index(self.index, self.faiss_index_path)
                    logging.info(f"Saved embeddings to {self.embedding_storage_path} and FAISS index to {self.faiss_index_path}.")
                else:
                    logging.warning("No embeddings to save.")
            except Exception as e:
                logging.error(f"Failed to save embeddings or FAISS index: {e}")

    def _save_metadata(self, passage_id: str, content: str, similarity_score: float, metadata: Dict):
        """Save a single passage's metadata to the SQLite database."""
        try:
            metadata_json = json.dumps(metadata)
            self.cursor.execute('''
                INSERT INTO metadata (passage_id, content, similarity_score, metadata)
                VALUES (?, ?, ?, ?)
            ''', (passage_id, content, similarity_score, metadata_json))
            self.conn.commit()
            logging.info(f"Saved metadata for passage ID {passage_id}.")
        except sqlite3.IntegrityError:
            logging.warning(f"Passage ID {passage_id} already exists in metadata database.")
        except Exception as e:
            logging.error(f"Failed to save metadata for passage ID {passage_id}: {e}")

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

    def add_documents(self, contents: List[str], source_metadata: Optional[Dict] = None, batch_size: int = 32):
        """
        Add multiple documents with metadata to the retriever in batches.

        Args:
            contents (List[str]): List of document contents
            source_metadata (Dict, optional): Additional metadata about the sources
            batch_size (int): Number of passages to process in each batch
        """
        passages_with_metadata = []
        for content in contents:
            splits = self._smart_split_text(content)
            for passage, metadata in splits:
                if source_metadata:
                    metadata.update(source_metadata)
                passages_with_metadata.append((passage, metadata))
        
        num_batches = (len(passages_with_metadata) + batch_size - 1) // batch_size
        for i in range(num_batches):
            batch = passages_with_metadata[i*batch_size:(i+1)*batch_size]
            self._process_batch(batch)
    
        self._save_embeddings()

    def _process_batch(self, batch: List[Tuple[str, Dict]]):
        """Process a batch of passages."""
        passages = []
        embeddings = []
        metadata_batch = []
        passage_ids = []
        
        for passage, metadata in batch:
            passage_id = self._generate_passage_id(passage)
            if passage_id in self.passage_ids:
                logging.info(f"Passage already exists with ID {passage_id}. Skipping.")
                continue
            try:
                embedding = self.embedding_model.encode(passage).astype('float32')
                faiss.normalize_L2(embedding.reshape(1, -1))
                passages.append(passage)
                embeddings.append(embedding)
                metadata_batch.append(metadata)
                passage_ids.append(passage_id)
            except Exception as e:
                logging.error(f"Failed to encode passage: {e}")
                continue
        
        if embeddings:
            with self.lock:
                new_embeddings = np.vstack(embeddings)
                self.embeddings = np.vstack([self.embeddings, new_embeddings]) if self.embeddings.size else new_embeddings
                self.index.add(new_embeddings)
                self.passages.extend(passages)
                self.metadata.extend(metadata_batch)
                self.passage_ids.extend(passage_ids)
                for pid, passage, meta in zip(passage_ids, passages, metadata_batch):
                    self._save_metadata(pid, passage, 0.0, meta)
                logging.info(f"Added {len(embeddings)} new passages.")

    def update_passage(self, passage_id: str, new_content: Optional[str] = None, new_metadata: Optional[Dict] = None):
        """
        Update an existing passage's content and/or metadata.
        
        Args:
            passage_id (str): ID of the passage to update
            new_content (str, optional): New content for the passage
            new_metadata (Dict, optional): New metadata to update
        """
        with self.lock:
            if passage_id not in self.passage_ids:
                logging.warning(f"Passage ID {passage_id} not found.")
                return
            
            idx = self.passage_ids.index(passage_id)
            
            if new_content:
                old_content = self.passages[idx]
                self.passages[idx] = new_content
                try:
                    new_embedding = self.embedding_model.encode(new_content).astype('float32')
                    faiss.normalize_L2(new_embedding.reshape(1, -1))
                    self.embeddings[idx] = new_embedding
                    # Update FAISS index incrementally
                    self.index.replace_ids(np.array([idx]), np.array([new_embedding]))
                    logging.info(f"Updated content and embedding for passage ID {passage_id}.")
                    
                    # Update metadata in database
                    self.cursor.execute('''
                        UPDATE metadata
                        SET content = ?, similarity_score = ?
                        WHERE passage_id = ?
                    ''', (new_content, 0.0, passage_id))
                    self.conn.commit()
                except Exception as e:
                    logging.error(f"Failed to encode updated passage: {e}")
                    # Revert to old content if encoding fails
                    self.passages[idx] = old_content
            if new_metadata:
                self.metadata[idx].update(new_metadata)
                logging.info(f"Updated metadata for passage ID {passage_id}.")
                
                # Update metadata in database
                try:
                    metadata_json = json.dumps(self.metadata[idx])
                    self.cursor.execute('''
                        UPDATE metadata
                        SET metadata = ?
                        WHERE passage_id = ?
                    ''', (metadata_json, passage_id))
                    self.conn.commit()
                except Exception as e:
                    logging.error(f"Failed to update metadata in database for passage ID {passage_id}: {e}")
            
            self._save_embeddings()

    def delete_passage(self, passage_id: str):
        """
        Delete a passage from the retriever.
        
        Args:
            passage_id (str): ID of the passage to delete
        """
        with self.lock:
            if passage_id not in self.passage_ids:
                logging.warning(f"Passage ID {passage_id} not found.")
                return
            
            idx = self.passage_ids.index(passage_id)
            
            # Remove from lists
            del self.passages[idx]
            del self.metadata[idx]
            del self.passage_ids[idx]
            self.embeddings = np.delete(self.embeddings, idx, axis=0)
            self.index.reset()  # Reset and rebuild index
            if self.embeddings.size > 0:
                faiss.normalize_L2(self.embeddings)
                self.index.train(self.embeddings)
                self.index.add(self.embeddings)
                logging.info(f"Deleted passage ID {passage_id} and rebuilt FAISS index.")
            
            # Remove from metadata database
            try:
                self.cursor.execute('''
                    DELETE FROM metadata
                    WHERE passage_id = ?
                ''', (passage_id,))
                self.conn.commit()
                logging.info(f"Deleted metadata for passage ID {passage_id} from database.")
            except Exception as e:
                logging.error(f"Failed to delete metadata from database for passage ID {passage_id}: {e}")
            
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
        if not self.embeddings.size:
            logging.info("No passages available for retrieval.")
            return []

        # Encode query
        try:
            query_embedding = self.embedding_model.encode(query).astype('float32')
            faiss.normalize_L2(query_embedding.reshape(1, -1))
            logging.info("Encoded query successfully.")
        except Exception as e:
            logging.error(f"Failed to encode query: {e}")
            return []
        
        # Perform FAISS search
        try:
            D, I = self.index.search(query_embedding.reshape(1, -1), top_k * 2)  # Retrieve more to account for filtering
            logging.info("Performed FAISS search.")
        except Exception as e:
            logging.error(f"Failed to perform FAISS search: {e}")
            return []
        
        retrieved = []
        for distance, idx in zip(D[0], I[0]):
            if idx == -1 or idx >= len(self.passages):
                continue
            similarity_score = float(distance)
            if similarity_score < self.similarity_threshold:
                continue
            metadata = self.metadata[idx]
            
            # Apply filter criteria if specified
            if filter_criteria:
                if not all(metadata.get(k) == v for k, v in filter_criteria.items()):
                    continue
            
            retrieved.append(RetrievedPassage(
                content=self.passages[idx],
                similarity_score=similarity_score,
                metadata=metadata,
                passage_id=self.passage_ids[idx]
            ))
            if len(retrieved) >= top_k:
                break
        
        logging.info(f"Retrieved {len(retrieved)} passages.")
        return retrieved

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
        with self.lock:
            if not filter_criteria:
                self.passages = []
                self.metadata = []
                self.passage_ids = []
                self.embeddings = np.empty((0, self.embedding_model.get_sentence_embedding_dimension()), dtype='float32')
                self.index.reset()
                logging.info("Cleared all passages and embeddings.")
                
                # Clear metadata database
                try:
                    self.cursor.execute('DELETE FROM metadata')
                    self.conn.commit()
                    logging.info("Cleared all metadata from database.")
                except Exception as e:
                    logging.error(f"Failed to clear metadata database: {e}")
                
                # Remove embedding and FAISS index files
                if os.path.exists(self.embedding_storage_path):
                    os.remove(self.embedding_storage_path)
                    logging.info(f"Removed embedding file at {self.embedding_storage_path}.")
                if os.path.exists(self.faiss_index_path):
                    os.remove(self.faiss_index_path)
                    logging.info(f"Removed FAISS index file at {self.faiss_index_path}.")
            else:
                indices_to_keep = []
                for i, metadata in enumerate(self.metadata):
                    if not all(metadata.get(k) == v for k, v in filter_criteria.items()):
                        indices_to_keep.append(i)
                
                self.passages = [self.passages[i] for i in indices_to_keep]
                self.embeddings = self.embeddings[indices_to_keep]
                self.metadata = [self.metadata[i] for i in indices_to_keep]
                self.passage_ids = [self.passage_ids[i] for i in indices_to_keep]
                
                # Rebuild FAISS index
                try:
                    self.index.reset()
                    if self.embeddings.size > 0:
                        faiss.normalize_L2(self.embeddings)
                        self.index.train(self.embeddings)
                        self.index.add(self.embeddings)
                    logging.info("Rebuilt FAISS index after clearing data based on filters.")
                except Exception as e:
                    logging.error(f"Failed to rebuild FAISS index: {e}")
                
                # Update metadata database
                try:
                    filter_conditions = ' AND '.join([f'metadata LIKE "%{k}: {v}%"' for k, v in filter_criteria.items()])
                    self.cursor.execute(f'DELETE FROM metadata WHERE NOT ({filter_conditions})')
                    self.conn.commit()
                    logging.info("Cleared metadata from database based on filter criteria.")
                except Exception as e:
                    logging.error(f"Failed to clear metadata from database based on filters: {e}")
            
            self._save_embeddings()

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
    retriever.add_documents([document], metadata)
    
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