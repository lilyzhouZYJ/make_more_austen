import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional


class ProbabilisticBigramModel:
    """
    Probabilistic bigram language model.
    
    This model uses a simple count-based approach to learn bigram probabilities.
    It counts occurrences of word pairs in the training text and normalizes
    them to create a probability matrix.
    """
    
    def __init__(self):
        """
        Initialize the probabilistic bigram model.
        """
        self.vocab = {}            # set of vocabulary
        self.word_to_idx = {}      # map of words to indices
        self.idx_to_word = {}      # map of indices to words
        self.all_words = []        # list of all tokens from the dataset
        self.is_trained = False    # flag to indicate if model is trained
        
        self.bigram_counts = defaultdict(int)  # count of bigrams
        self.bigram_probs = None              # probability matrix
    
    def train(self, file_path: str, max_training_data_size: Optional[int] = None) -> None:
        """
        Train the probabilistic bigram model on the given text file.
        
        Args:
            file_path: Path to the training text file
            max_training_data_size: Maximum number of words to use for training
        """
        print("Training probabilistic bigram model...")
        
        # Load and preprocess data
        self._load_and_preprocess_data(file_path, max_training_data_size)
        
        # Build bigrams
        self._build_bigrams()
        
        # Create probability matrix
        self._create_bigram_matrix()
        
        self.is_trained = True
        print("Probabilistic bigram training completed!")
    
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
    
    def _build_bigrams(self) -> None:
        """
        Build bigrams from all_words.
        """
        print("Building bigrams...")
        
        for i in range(len(self.all_words) - 1):
            current_word = self.all_words[i]
            next_word = self.all_words[i + 1]
            bigram = (current_word, next_word)
            self.bigram_counts[bigram] += 1
        
        print(f"Found {len(self.bigram_counts)} unique bigrams")
    
    def _create_bigram_matrix(self) -> None:
        """
        Create bigram probability matrix.
        """
        print("Creating bigram probability matrix...")
        
        vocab_size = len(self.vocab)
        self.bigram_probs = torch.zeros((vocab_size, vocab_size))
        
        # Fill the matrix with counts
        for (word1, word2), count in self.bigram_counts.items():
            idx1 = self.word_to_idx[word1]
            idx2 = self.word_to_idx[word2]
            self.bigram_probs[idx1, idx2] = count
        
        # Normalize rows to get probabilities
        self.bigram_probs = F.normalize(self.bigram_probs, p=1, dim=1)
        print("Bigram probability matrix created")
    
    def generate_text(self, max_length: int = 50, start_word: str = '<START>', 
                     temperature: float = 1.0) -> str:
        """
        Generate text using the bigram model.
        
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
                
            # Get probabilities for next word
            current_idx = self.word_to_idx[current_word]
            probs = self.bigram_probs[current_idx]
            
            # Apply temperature for more diverse generation
            if temperature != 1.0:
                probs = probs / temperature
                probs = F.softmax(probs, dim=0)
            
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
        Get the most likely words to follow a given word.
        
        Args:
            word_idx: Index of the input word
            top_k: Number of top predictions to return
            
        Returns:
            List of (word, probability) tuples
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        probs = self.bigram_probs[word_idx]
        
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
            'model_type': 'Probabilistic Bigram',
            'all_words_size': len(self.all_words),
            'vocabulary_size': len(self.vocab),
            'total_bigram_count': len(self.bigram_counts),
            'most_common_bigrams': Counter(self.bigram_counts).most_common(5),
            'most_common_words': []
        }
        
        # Count word frequencies
        word_freq = Counter()
        for (word1, word2), count in self.bigram_counts.items():
            word_freq[word1] += count
        
        stats['most_common_words'] = word_freq.most_common(5)
        return stats
    
    def calculate_loss(self) -> Dict:
        """
        Calculate the negative log-likelihood loss of the model.
        This measures how well the model predicts the actual bigrams in the training data.
        Lower loss means better predictions.
        
        Returns:
            Dictionary containing loss metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before calculating loss")
        
        print("Calculating probabilistic model loss...")
        
        total_loss = 0.0
        total_bigrams = 0
        
        for (word1, word2), count in self.bigram_counts.items():
            word1_idx = self.word_to_idx[word1]
            word2_idx = self.word_to_idx[word2]
            
            # Get the probability assigned by the model to this bigram
            prob = self.bigram_probs[word1_idx, word2_idx].item()
            
            # Avoid log(0) by adding small epsilon
            eps = 1e-8
            prob = max(prob, eps)
            
            # Negative log-likelihood for this bigram (weighted by count)
            loss = -torch.log(torch.tensor(prob)) * count
            total_loss += loss.item()
            total_bigrams += count
        
        # Average loss per bigram
        avg_loss = total_loss / total_bigrams
        
        # Calculate perplexity (exp of average loss)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'average_loss': avg_loss,
            'perplexity': perplexity
        }
    
    def visualize_bigram_heatmap(self, words_to_show: int = 20, figsize: tuple = (12, 8)) -> None:
        """
        Visualize the bigram probability matrix as a heatmap.
        
        Args:
            words_to_show: Number of words to show in heatmap
            figsize: Figure size for the plot
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before creating heatmap")
        
        # Get the most frequent words for visualization
        word_freq = Counter()
        for (word1, word2), count in self.bigram_counts.items():
            word_freq[word1] += count
        
        top_words = [word for word, freq in word_freq.most_common(words_to_show)]
        
        # Create submatrix for visualization
        submatrix = torch.zeros((words_to_show, words_to_show))
        for i, word1 in enumerate(top_words):
            if word1 in self.word_to_idx:
                idx1 = self.word_to_idx[word1]
                for j, word2 in enumerate(top_words):
                    if word2 in self.word_to_idx:
                        idx2 = self.word_to_idx[word2]
                        submatrix[i, j] = self.bigram_probs[idx1, idx2]
        
        # Create heatmap
        plt.figure(figsize=figsize)
        plt.imshow(submatrix.numpy(), cmap='Blues', aspect='auto')
        plt.colorbar(label='Probability')
        plt.xlabel('Next Word')
        plt.ylabel('Current Word')
        plt.title('Bigram Probability Heatmap (Top {} Words)'.format(words_to_show))
        
        # Set tick labels
        plt.xticks(range(words_to_show), top_words, rotation=45, ha='right')
        plt.yticks(range(words_to_show), top_words)
        
        plt.tight_layout()
        plt.savefig('bigram_heatmap.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Heatmap saved as 'bigram_heatmap.png'")