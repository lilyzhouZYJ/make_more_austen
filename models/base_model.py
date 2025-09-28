from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional


class BaseBigramModel(ABC):
    """
    Abstract base class for all bigram language models in the Austen text generation project.
    
    This class defines the interface that all bigram models must implement to ensure
    compatibility and extensibility.
    """
    
    def __init__(self):
        self.vocab = {}            # set of vocabulary
        self.word_to_idx = {}      # map of words to indices
        self.idx_to_word = {}      # map of indices to words
        self.all_words = []        # list of all tokens from the dataset
        self.is_trained = False    # flag to indicate if model is trained
    
    @abstractmethod
    def train(self, file_path: str, max_training_data_size: Optional[int] = None) -> None:
        """
        Train the model on the given text file.
        
        Args:
            file_path: Path to the training text file
            max_training_data_size: Maximum number of words to use for training (None for all data)
        """
        pass
    
    @abstractmethod
    def generate_text(self, max_length: int = 50, start_word: str = '<START>', 
                     temperature: float = 1.0) -> str:
        """
        Generate text using the trained model.
        
        Args:
            max_length: Maximum number of words to generate
            start_word: Word to start generation with
            temperature: Controls randomness (higher = more random)
            
        Returns:
            Generated text string
        """
        pass
    
    @abstractmethod
    def get_model_stats(self) -> Dict:
        """
        Get statistics about the trained model.
        
        Returns:
            Dictionary containing model statistics
        """
        pass
    
    @abstractmethod
    def calculate_loss(self) -> Dict:
        """
        Calculate the model's loss and perplexity.
        
        Returns:
            Dictionary containing loss metrics
        """
        pass
    
    def generate_sentences(self, num_sentences: int = 5, max_words_per_sentence: int = 20, 
                          temperature: float = 1.0) -> List[str]:
        """
        Generate multiple complete sentences.
        
        Args:
            num_sentences: Number of sentences to generate
            max_words_per_sentence: Maximum words per sentence
            temperature: Controls randomness
            
        Returns:
            List of generated sentences
        """
        sentences = []
        for _ in range(num_sentences):
            sentence = self.generate_text(
                max_length=max_words_per_sentence, 
                start_word='<START>',
                temperature=temperature)
            sentences.append(sentence)
        return sentences
    
    def get_most_likely_words(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get the most likely words to follow a given word.
        
        Args:
            word: Input word
            top_k: Number of top predictions to return
            
        Returns:
            List of (word, probability) tuples
        """
        if word not in self.word_to_idx:
            print(f"Word '{word}' not in vocabulary")
            return []
        
        word_idx = self.word_to_idx[word]
        return self._get_next_word_probabilities(word_idx, top_k)
    
    @abstractmethod
    def _get_next_word_probabilities(self, word_idx: int, top_k: int) -> List[Tuple[str, float]]:
        """
        Internal method to get next word probabilities for a given word index.
        
        Args:
            word_idx: Index of the input word
            top_k: Number of top predictions to return
            
        Returns:
            List of (word, probability) tuples
        """
        pass
    
    def _load_and_preprocess_data(self, file_path: str, max_training_data_size: Optional[int] = None) -> None:
        """
        Load and preprocess data from file.
        
        Args:
            file_path: Path to the text file
            max_training_data_size: Maximum number of words to use for training
        """
        import re
        
        print("Loading data file...")
        with open(file_path, 'r') as f:
            data = f.read()
        
        # Basic preprocessing: lowercase, remove extra whitespace, split into sentences
        data = data.lower()
        data = re.sub(r'\s+', ' ', data)  # Replace multiple whitespace with single space
        
        # Split into sentences (simple approach using periods)
        sentences = []
        for sentence in data.split('.'):
            sentence = sentence.strip()
            if sentence:
                sentences.append(sentence + '.')
        
        print(f"Found {len(sentences)} sentences")

        # Split each sentence into words (using whitespace / punctuation);
        # add start and end tokens
        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence)
            words = ['<START>'] + words + ['<END>']
            self.all_words.extend(words)
            
            # Limit training data size if specified
            if max_training_data_size and len(self.all_words) >= max_training_data_size:
                self.all_words = self.all_words[:max_training_data_size]
                print(f"Limited training data to {len(self.all_words)} words")
                break

        print(f"Tokenized into {len(self.all_words)} words")
        
        # Build vocabulary
        self.vocab = list(set(self.all_words))
        self.vocab.sort()
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(self.vocab)}
        
        print(f"Vocabulary size: {len(self.vocab)}")
