import torch
import torch.nn.functional as F
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional

class NeuralBigramModel:
    """
    Neural network-based bigram language model.
    
    This model implements a simple neural network that learns to predict the next word
    given the current word. It uses an embedding layer to represent words as dense vectors
    and a linear layer to predict the next word probabilities.
    """
    
    def __init__(self, embedding_dim: int = 64, learning_rate: float = 0.1, num_epochs: int = 100):
        """
        Initialize the neural bigram model.
        
        Args:
            embedding_dim: Dimension of word embeddings
            learning_rate: Learning rate for training
            num_epochs: Number of training epochs
        """
        self.vocab = {}            # set of vocabulary
        self.word_to_idx = {}      # map of words to indices
        self.idx_to_word = {}      # map of indices to words
        self.all_words = []        # list of all tokens from the dataset
        self.is_trained = False    # flag to indicate if model is trained
        
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        # Neural network components (initialized after vocabulary is built)
        self.embeddings = None
        self.linear = None
        self.optimizer = None
        
        # Training data
        self.X = None  # input words (indices)
        self.Y = None  # target words (indices)
        
    def train(self, file_path: str, max_training_data_size: Optional[int] = None) -> None:
        """
        Train the neural bigram model on the given text file.
        
        Args:
            file_path: Path to the training text file
            max_training_data_size: Maximum number of words to use for training
        """
        print("Training neural bigram model...")
        
        # Load and preprocess data
        self._load_and_preprocess_data(file_path, max_training_data_size)
        
        # Prepare training data
        self._prepare_training_data()
        
        # Initialize neural network
        self._initialize_network()
        
        # Train the model
        self._train_neural_network()
        
        self.is_trained = True
        print("Neural bigram training completed!")
    
    def _load_and_preprocess_data(self, file_path: str, max_training_data_size: Optional[int] = None) -> None:
        """
        Load and preprocess data from file.
        
        Args:
            file_path: Path to the text file
            max_training_data_size: Maximum number of words to use for training
        """
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
    
    def _prepare_training_data(self) -> None:
        """
        Prepare training data (X, Y pairs) from the word sequence.
        """
        print("Preparing training data...")
        
        # Create X, Y pairs for training
        # X[i] is the current word, Y[i] is the next word
        X = []
        Y = []
        
        for i in range(len(self.all_words) - 1):
            current_word = self.all_words[i]
            next_word = self.all_words[i + 1]
            
            X.append(self.word_to_idx[current_word])
            Y.append(self.word_to_idx[next_word])
        
        self.X = torch.tensor(X, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.long)
        
        print(f"Prepared {len(X)} training examples")
    
    def _initialize_network(self) -> None:
        """
        Initialize the neural network components.
        """
        print("Initializing neural network...")
        
        vocab_size = len(self.vocab)
        
        # Create embeddings for each word
        self.embeddings = torch.randn(vocab_size, self.embedding_dim, requires_grad=True)
        
        # Create linear layer to predict next word probabilities
        self.linear = torch.randn(self.embedding_dim, vocab_size, requires_grad=True)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam([self.embeddings, self.linear], lr=self.learning_rate)
        
        print(f"Initialized network with {vocab_size} vocabulary size and {self.embedding_dim} embedding dimension")
    
    def _train_neural_network(self) -> None:
        """
        Train the neural network using gradient descent.
        """
        print(f"Training neural network for {self.num_epochs} epochs...")
        
        for epoch in range(self.num_epochs):
            # Forward pass
            # Get embeddings for input words
            emb = self.embeddings[self.X]  # (N, embedding_dim)
            
            # Predict logits for next words
            logits = emb @ self.linear  # (N, vocab_size)
            
            # Calculate loss (cross-entropy)
            loss = F.cross_entropy(logits, self.Y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Print progress
            if epoch % 10 == 0 or epoch == self.num_epochs - 1:
                print(f"Epoch {epoch:3d}: Loss = {loss.item():.4f}")
    
    def generate_text(self, max_length: int = 50, start_word: str = '<START>', 
                     temperature: float = 1.0) -> str:
        """
        Generate text using the trained neural network.
        
        Args:
            max_length: Maximum number of words to generate
            start_word: Word to start generation with
            temperature: Controls randomness (higher = more random)
            
        Returns:
            Generated text string
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before generating text")
        
        generated_words = []
        current_word = start_word
        
        for _ in range(max_length):
            if current_word != '<START>':
                generated_words.append(current_word)
            
            if current_word == '<END>':
                break
                
            # Get probabilities for next word using neural network
            current_idx = self.word_to_idx[current_word]
            
            # Get embedding for current word
            emb = self.embeddings[current_idx]  # (embedding_dim,)
            
            # Get logits for next word
            logits = emb @ self.linear  # (vocab_size,)
            
            # Apply temperature
            logits = logits / temperature
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=0)
            
            # Sample next word
            next_idx = torch.multinomial(probs, 1).item()
            current_word = self.idx_to_word[next_idx]
        
        # Clean up the generated text
        text = ' '.join(generated_words)
        text = text.replace('<END>', '').strip()
        return text
    
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
    
    def _get_next_word_probabilities(self, word_idx: int, top_k: int) -> List[Tuple[str, float]]:
        """
        Get the most likely words to follow a given word using the neural network.
        
        Args:
            word_idx: Index of the input word
            top_k: Number of top predictions to return
            
        Returns:
            List of (word, probability) tuples
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get embedding for current word
        emb = self.embeddings[word_idx]  # (embedding_dim,)
        
        # Get logits for next word
        logits = emb @ self.linear  # (vocab_size,)
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=0)
        
        # Get top-k most likely next words
        top_probs, top_indices = torch.topk(probs, top_k)
        
        results = []
        for prob, idx in zip(top_probs, top_indices):
            word = self.idx_to_word[idx.item()]
            results.append((word, prob.item()))
        
        return results
    
    def get_model_stats(self) -> Dict:
        """
        Get statistics about the trained model.
        
        Returns:
            Dictionary containing model statistics
        """
        stats = {
            'model_type': 'Neural Bigram',
            'all_words_size': len(self.all_words),
            'vocabulary_size': len(self.vocab),
            'embedding_dimension': self.embedding_dim,
            'learning_rate': self.learning_rate,
            'training_epochs': self.num_epochs,
            'most_common_words': []
        }
        
        # Count word frequencies
        word_freq = Counter(self.all_words)
        stats['most_common_words'] = word_freq.most_common(5)
        
        return stats
    
    def calculate_loss(self) -> Dict:
        """
        Calculate the model's loss and perplexity on the training data.
        
        Returns:
            Dictionary containing loss metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before calculating loss")
        
        print("Calculating neural network model loss...")
        
        # Forward pass to get current loss
        emb = self.embeddings[self.X]  # (N, embedding_dim)
        logits = emb @ self.linear  # (N, vocab_size)
        loss = F.cross_entropy(logits, self.Y)
        
        # Calculate perplexity
        perplexity = torch.exp(loss).item()
        
        return {
            'average_loss': loss.item(),
            'perplexity': perplexity
        }