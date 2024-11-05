import json
import os
import threading
from typing import List, Optional, Dict, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict
import logging
from pathlib import Path

@dataclass
class Thought:
    """Represents a single thought with metadata."""
    content: str
    timestamp: str
    tags: List[str]
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class HistoryEmbedder:
    """
    Enhanced version of HistoryEmbedder with improved functionality for embedding,
    searching, and managing thought history with metadata and tags.
    """

    def __init__(
        self,
        history_dir: str = "history",
        max_history_size: int = 100,
        model_name: str = 'all-MiniLM-L6-v2',
        auto_backup: bool = True,
        backup_interval: int = 10
    ):
        """
        Initialize the HistoryEmbedder with enhanced configuration options.

        Args:
            history_dir (str): Directory for history storage and backups
            max_history_size (int): Maximum number of thoughts to retain
            model_name (str): Name of the sentence transformer model to use
            auto_backup (bool): Enable automatic backups
            backup_interval (int): Number of changes before automatic backup
        """
        self.history_dir = Path(history_dir)
        self.history_file = self.history_dir / "history.json"
        self.backup_dir = self.history_dir / "backups"
        self.max_history_size = max_history_size
        self.auto_backup = auto_backup
        self.backup_interval = backup_interval
        
        # Create necessary directories
        self.history_dir.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)
        
        # Initialize logging
        logging.basicConfig(
            filename=self.history_dir / "history.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        self.thoughts: List[Thought] = []
        self.tags_index: Dict[str, List[int]] = defaultdict(list)
        self.lock = threading.Lock()
        self.changes_since_backup = 0
        
        # Initialize the embedding model
        try:
            self.model = SentenceTransformer(model_name)
            logging.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            logging.error(f"Failed to load model {model_name}: {e}")
            raise
        
        self._load_history()

    def embed(
        self,
        thought_content: str,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Embeds a thought with enhanced metadata and tags.

        Args:
            thought_content (str): The thought content to embed
            tags (List[str]): Optional list of tags for categorization
            metadata (Dict[str, Any]): Optional metadata for the thought

        Returns:
            bool: Success status of the embedding operation
        """
        if not thought_content.strip():
            logging.warning("Attempted to embed empty thought")
            return False

        with self.lock:
            try:
                # Generate embedding
                embedding = self.model.encode(thought_content).tolist()
                
                # Create thought object
                thought = Thought(
                    content=thought_content,
                    timestamp=datetime.now().isoformat(),
                    tags=tags or [],
                    metadata=metadata or {},
                    embedding=embedding
                )
                
                # Manage history size
                if len(self.thoughts) >= self.max_history_size:
                    removed_thought = self.thoughts.pop(0)
                    self._update_indices(removed_thought, remove=True)
                    logging.info(f"Removed oldest thought due to size limit")
                
                # Add new thought
                self.thoughts.append(thought)
                self._update_indices(thought)
                
                # Handle backup
                self.changes_since_backup += 1
                if self.auto_backup and self.changes_since_backup >= self.backup_interval:
                    self._create_backup()
                
                self._save_history()
                logging.info(f"Successfully embedded thought with {len(tags or [])} tags")
                return True
                
            except Exception as e:
                logging.error(f"Failed to embed thought: {e}")
                return False

    def search_by_similarity(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.3,
        tags: List[str] = None
    ) -> List[Tuple[Thought, float]]:
        """
        Enhanced similarity search with filtering and minimum similarity threshold.

        Args:
            query (str): Search query
            top_k (int): Maximum number of results
            min_similarity (float): Minimum similarity threshold
            tags (List[str]): Optional tags to filter results

        Returns:
            List[Tuple[Thought, float]]: Thoughts and their similarity scores
        """
        with self.lock:
            if not self.thoughts:
                return []

            # Generate query embedding
            query_embedding = self.model.encode(query)
            
            # Filter thoughts by tags if specified
            candidate_thoughts = self.thoughts
            if tags:
                tagged_indices = set.intersection(
                    *[set(self.tags_index[tag]) for tag in tags]
                )
                candidate_thoughts = [self.thoughts[i] for i in tagged_indices]
            
            # Calculate similarities and filter
            results = []
            for thought in candidate_thoughts:
                similarity = self._cosine_similarity(
                    query_embedding,
                    np.array(thought.embedding)
                )
                if similarity >= min_similarity:
                    results.append((thought, similarity))
            
            # Sort and return top results
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

    def get_thoughts_by_tags(self, tags: List[str], require_all: bool = True) -> List[Thought]:
        """
        Retrieve thoughts that match specified tags.

        Args:
            tags (List[str]): Tags to filter by
            require_all (bool): If True, thoughts must have all tags

        Returns:
            List[Thought]: Matching thoughts
        """
        with self.lock:
            if not tags:
                return self.thoughts.copy()
            
            if require_all:
                indices = set.intersection(*[set(self.tags_index[tag]) for tag in tags])
            else:
                indices = set.union(*[set(self.tags_index[tag]) for tag in tags])
            
            return [self.thoughts[i] for i in sorted(indices)]

    def _update_indices(self, thought: Thought, remove: bool = False):
        """Update tag indices when adding or removing thoughts."""
        thought_index = len(self.thoughts) - 1
        for tag in thought.tags:
            if remove:
                self.tags_index[tag].remove(thought_index)
                if not self.tags_index[tag]:
                    del self.tags_index[tag]
            else:
                self.tags_index[tag].append(thought_index)

    def _create_backup(self):
        """Create a timestamped backup of the history."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"history_backup_{timestamp}.json"
        try:
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        'thoughts': [asdict(thought) for thought in self.thoughts],
                        'tags_index': dict(self.tags_index)
                    },
                    f,
                    ensure_ascii=False,
                    indent=4
                )
            self.changes_since_backup = 0
            logging.info(f"Created backup: {backup_file}")
        except Exception as e:
            logging.error(f"Failed to create backup: {e}")

    def _save_history(self):
        """Save the current history to disk."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        'thoughts': [asdict(thought) for thought in self.thoughts],
                        'tags_index': dict(self.tags_index)
                    },
                    f,
                    ensure_ascii=False,
                    indent=4
                )
        except Exception as e:
            logging.error(f"Failed to save history: {e}")
            raise

    def _load_history(self):
        """Load history from disk."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.thoughts = [
                        Thought(**thought_data)
                        for thought_data in data['thoughts']
                    ]
                    self.tags_index = defaultdict(
                        list,
                        {k: list(v) for k, v in data['tags_index'].items()}
                    )
                logging.info(f"Loaded history with {len(self.thoughts)} thoughts")
            except Exception as e:
                logging.error(f"Failed to load history: {e}")
                raise

    def analyze_history(self) -> Dict[str, Any]:
        """
        Analyze the history to generate insights.

        Returns:
            Dict[str, Any]: Analysis results including tag frequencies,
                           temporal patterns, etc.
        """
        with self.lock:
            analysis = {
                'total_thoughts': len(self.thoughts),
                'unique_tags': len(self.tags_index),
                'tag_frequencies': dict(sorted(
                    [(tag, len(indices)) for tag, indices in self.tags_index.items()],
                    key=lambda x: x[1],
                    reverse=True
                )),
                'average_thought_length': sum(len(t.content) for t in self.thoughts) / len(self.thoughts) if self.thoughts else 0,
                'thoughts_per_day': defaultdict(int)
            }
            
            # Analyze temporal patterns
            for thought in self.thoughts:
                day = thought.timestamp.split('T')[0]
                analysis['thoughts_per_day'][day] += 1
            
            return analysis

    @property
    def stats(self) -> Dict[str, Any]:
        """
        Get current statistics about the history.

        Returns:
            Dict[str, Any]: Statistics about the history
        """
        with self.lock:
            return {
                'total_thoughts': len(self.thoughts),
                'unique_tags': len(self.tags_index),
                'storage_size': os.path.getsize(self.history_file) if self.history_file.exists() else 0,
                'last_modified': datetime.fromtimestamp(os.path.getmtime(self.history_file)).isoformat() if self.history_file.exists() else None,
                'backup_count': len(list(self.backup_dir.glob('*.json')))
            }

if __name__ == "__main__":
    # Initialize the embedder
    embedder = HistoryEmbedder(
        history_dir="my_history",
        max_history_size=1000,
        auto_backup=True
    )

    # Add a thought with tags and metadata
    embedder.embed(
        "This is an important thought about AI",
        tags=["AI", "important"],
        metadata={"priority": "high", "project": "AI Research"}
    )

    # Search for similar thoughts with tag filtering
    results = embedder.search_by_similarity(
        "AI technology",
        top_k=5,
        min_similarity=0.3,
        tags=["AI"]
    )

    # Get thoughts by tags
    ai_thoughts = embedder.get_thoughts_by_tags(["AI", "important"])

    # Analyze history
    analysis = embedder.analyze_history()
    print(f"Total thoughts: {analysis['total_thoughts']}")
    print(f"Most common tags: {analysis['tag_frequencies']}")

    # Get current stats
    stats = embedder.stats
    print(f"Storage size: {stats['storage_size']} bytes")