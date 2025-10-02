import torch
import torch.nn.functional as F
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional

class MLPModel:
    """
    Multi-Layer Perceptron (MLP) language model.
    
    This model implements a word-level MLP that learns to predict the next word
    given a sequence of previous words. It uses word embeddings, multiple hidden
    layers with Tanh activation, and outputs probability distributions over the vocabulary.
    """
    
    def __init__(self, block_size: int = 3, embedding_dim: int = 64, hidden_dim: int = 128, 
                 learning_rate: float = 0.1, num_epochs: int = 100, mini_batch_size: Optional[int] = None):
        """
        Initialize the MLP model.
        
        Args:
            block_size: Number of words to use as context (input sequence length)
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of hidden layers
            learning_rate: Learning rate for training
            num_epochs: Number of training epochs
            mini_batch_size: Mini-batch size for training (None for full batch)
        """
        self.vocab = {}            # set of vocabulary
        self.word_to_idx = {}      # map of words to indices
        self.idx_to_word = {}      # map of indices to words
        self.all_words = []        # list of all words from the dataset
        self.is_trained = False    # flag to indicate if model is trained
        
        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.mini_batch_size = mini_batch_size
        
        # Neural network components (initialized after vocabulary is built)
        self.embeddings = None
        self.W1 = None  # First layer weights
        self.b1 = None  # First layer bias
        self.W2 = None  # Second layer weights
        self.b2 = None  # Second layer bias
        self.optimizer = None
        
        # Training data
        self.X = None  # input word sequences (indices)
        self.Y = None  # target words (indices)
        
    def train(self, file_path: str, max_training_data_size: Optional[int] = None) -> None:
        """
        Train the MLP model on the given text file.
        
        Args:
            file_path: Path to the training text file
            max_training_data_size: Maximum number of words to use for training
        """
        print("Training MLP model...")
        
        # Load and preprocess data
        self._load_and_preprocess_data(file_path, max_training_data_size)
        
        # Prepare training data
        self._prepare_training_data()
        
        # Initialize neural network
        self._initialize_network()
        
        # Train the model
        self._train_neural_network()
        
        self.is_trained = True
        print("MLP training completed!")
    
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
            # Add padding in front so that we can start generating with empty input ('<start><start><start>')
            words = ['<START>'] * (self.block_size) + words + ['<END>']
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
        print(f"Preparing training data, block size (context length) is {self.block_size}")
        
        # Create X, Y pairs for training
        # X[i] is a sequence of block_size words, Y[i] is the next word
        X = []
        Y = []
        
        for i in range(len(self.all_words) - self.block_size):
            # Get block_size words as context
            context = self.all_words[i:i + self.block_size]
            # Get the next word as target
            target = self.all_words[i + self.block_size]
            
            # Convert to indices
            X.append([self.word_to_idx[word] for word in context])
            Y.append(self.word_to_idx[target])
        
        self.X = torch.tensor(X, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.long)
        
        print(f"Prepared {len(X)} training examples")
        print(f"Input shape: {self.X.shape}, Target shape: {self.Y.shape}")
    
    def _initialize_network(self) -> None:
        """
        Initialize the neural network components.
        """
        print("Initializing neural network...")
        
        vocab_size = len(self.vocab)
        
        # Create embeddings for each word: each row is the embedding for a word
        self.embeddings = torch.randn(vocab_size, self.embedding_dim, requires_grad=True)
        
        # First layer (embedding_dim * block_size -> hidden_dim)
        self.W1 = torch.randn(self.embedding_dim * self.block_size, self.hidden_dim, requires_grad=True)
        self.b1 = torch.zeros(self.hidden_dim, requires_grad=True)
        
        # Second layer (hidden_dim -> vocab_size)
        self.W2 = torch.randn(self.hidden_dim, vocab_size, requires_grad=True)
        self.b2 = torch.zeros(vocab_size, requires_grad=True)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam([
            self.embeddings, self.W1, self.b1, self.W2, self.b2
        ], lr=self.learning_rate)
        
        print(f"Initialized network with:")
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  Block size: {self.block_size}")
        print(f"  Embedding dimension: {self.embedding_dim}")
        print(f"  Hidden dimension: {self.hidden_dim}")
    
    def _train_neural_network(self) -> None:
        """
        Train the neural network using gradient descent with optional mini-batching.
        """
        print(f"Training neural network for {self.num_epochs} epochs...")

        # Check if we are using mini-batching
        use_mini_batch = self.mini_batch_size is not None
        if use_mini_batch:
            print(f"Using mini-batch size: {self.mini_batch_size}")
        else:
            print("Using full batch training")
        
        total_samples = len(self.X)
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            if use_mini_batch:
                # Shuffle data for each epoch
                indices = torch.randperm(total_samples)
                
                # Process data in mini-batches
                for start_idx in range(0, total_samples, self.mini_batch_size):
                    end_idx = min(start_idx + self.mini_batch_size, total_samples)
                    batch_indices = indices[start_idx:end_idx]
                    
                    # Get batch data
                    X_batch = self.X[batch_indices]
                    Y_batch = self.Y[batch_indices]
                    
                    # Forward pass
                    emb = self.embeddings[X_batch]  # (batch_size, block_size, embedding_dim)
                    emb_flat = emb.view(emb.shape[0], -1)  # (batch_size, block_size * embedding_dim)
                    h = torch.tanh(emb_flat @ self.W1 + self.b1)  # (batch_size, hidden_dim)
                    logits = h @ self.W2 + self.b2  # (batch_size, vocab_size)
                    loss = F.cross_entropy(logits, Y_batch)
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                avg_loss = epoch_loss / num_batches
            else:
                # Full batch training
                # Forward pass
                emb = self.embeddings[self.X]  # (N, block_size, embedding_dim)
                emb_flat = emb.view(emb.shape[0], -1)  # (N, block_size * embedding_dim)
                h = torch.tanh(emb_flat @ self.W1 + self.b1)  # (N, hidden_dim)
                logits = h @ self.W2 + self.b2  # (N, vocab_size)
                loss = F.cross_entropy(logits, self.Y)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                avg_loss = loss.item()
            
            # Print progress
            if epoch % 10 == 0 or epoch == self.num_epochs - 1:
                print(f"Epoch {epoch:3d}: Loss = {avg_loss:.4f}")
    
    def generate_text(self, max_length: int = 50, sentence_start: str = None, 
                     temperature: float = 1.0) -> str:
        """
        Generate text using the trained MLP model.
        
        Args:
            max_length: Maximum number of words to generate
            sentence_start: Words to start the sentence with
            temperature: Controls randomness (higher = more random)
            
        Returns:
            Generated text string
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before generating text")

        # Produce the first context
        if sentence_start is None:
            sentence_start_words = ['<START>'] * self.block_size
        else:
            sentence_start_words = sentence_start.split()
            if len(sentence_start_words) < self.block_size:
                # Add padding to the start of the context
                sentence_start_words = ['<START>'] * (self.block_size - len(sentence_start_words)) + sentence_start_words
            elif len(sentence_start_words) > self.block_size:
                # Truncate the context to the block size
                sentence_start_words = sentence_start_words[-self.block_size:]

        generated_words = []
        curr_context = [self.word_to_idx[word] for word in sentence_start_words]

        for _ in range(max_length):
            # Get embeddings for current context
            emb = self.embeddings[curr_context] # (block_size, embedding_dim)
            
            # Flatten embeddings for MLP input
            emb_flat = emb.view(1, -1)  # (1, block_size * embedding_dim)
            
            # First layer with tanh activation
            h = torch.tanh(emb_flat @ self.W1 + self.b1)  # (1, hidden_dim)
            
            # Second layer (output layer)
            logits = h @ self.W2 + self.b2  # (1, vocab_size)

            # Apply temperature
            logits = logits / temperature

            # Convert to probabilities
            probs = F.softmax(logits, dim=1)
            
            # Sample next word
            next_idx = torch.multinomial(probs, num_samples=1).item()
            next_word = self.idx_to_word[next_idx]
            
            generated_words.append(next_word)
            
            # Stop if we hit an end token
            if next_word == '<END>':
                break

            # Update context
            curr_context = curr_context[1:] + [next_idx]
        
        # Clean up the generated text
        text = ' '.join(generated_words)
        text = text.replace('<END>', '').strip()
        return text
    
    def generate_sentences(self, num_sentences: int = 5, max_words_per_sentence: int = 20, 
                          temperature: float = 1.0) -> List[str]:
        """
        Generate multiple text samples.
        
        Args:
            num_sentences: Number of text samples to generate
            max_words_per_sentence: Maximum words per sample
            temperature: Controls randomness
            
        Returns:
            List of generated text samples
        """
        sentences = []
        for _ in range(num_sentences):
            sentence = self.generate_text(
                max_length=max_words_per_sentence, 
                sentence_start=None,
                temperature=temperature)
            sentences.append(sentence)
        return sentences
    
    def get_most_likely_words(self, context: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get the most likely words to follow a given context.
        
        Args:
            context: Input word context (must be block_size words)
            top_k: Number of top predictions to return
            
        Returns:
            List of (word, probability) tuples
        """
        context_words = context.lower().split()
        if len(context_words) != self.block_size:
            raise ValueError(f"Context must be exactly {self.block_size} words long")
        
        for word in context_words:
            if word not in self.word_to_idx:
                raise ValueError(f"Word '{word}' not in vocabulary")
        
        return self._get_next_word_probabilities(context_words, top_k)
    
    def _get_next_word_probabilities(self, context_words: List[str], top_k: int) -> List[Tuple[str, float]]:
        """
        Get the most likely words to follow a given context using the neural network.
        
        Args:
            context_words: Input word context
            top_k: Number of top predictions to return
            
        Returns:
            List of (word, probability) tuples
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert context to indices
        context_indices = [self.word_to_idx[word] for word in context_words]
        context_tensor = torch.tensor([context_indices], dtype=torch.long)
        
        # Forward pass
        emb = self.embeddings[context_tensor]  # (1, block_size, embedding_dim)
        emb_flat = emb.view(1, -1)  # (1, block_size * embedding_dim)
        
        h = torch.tanh(emb_flat @ self.W1 + self.b1)  # (1, hidden_dim)
        logits = h @ self.W2 + self.b2  # (1, vocab_size)
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=1).squeeze()  # (vocab_size,)
        
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
            'model_type': 'MLP',
            'all_words_size': len(self.all_words),
            'vocabulary_size': len(self.vocab),
            'block_size': self.block_size,
            'embedding_dimension': self.embedding_dim,
            'hidden_dimension': self.hidden_dim,
            'learning_rate': self.learning_rate,
            'training_epochs': self.num_epochs,
            'mini_batch_size': self.mini_batch_size,
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
        
        print("Calculating MLP model loss...")
        
        # Forward pass to get current loss
        emb = self.embeddings[self.X]  # (N, block_size, embedding_dim)
        emb_flat = emb.view(emb.shape[0], -1)  # (N, block_size * embedding_dim)
        h = torch.tanh(emb_flat @ self.W1 + self.b1)  # (N, hidden_dim)
        logits = h @ self.W2 + self.b2  # (N, vocab_size)
        loss = F.cross_entropy(logits, self.Y)
        
        # Calculate perplexity
        perplexity = torch.exp(loss).item()
        
        return {
            'average_loss': loss.item(),
            'perplexity': perplexity
        }
