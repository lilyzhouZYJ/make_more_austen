import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import re
import argparse
from collections import defaultdict, Counter

class WordBigram:
    def __init__(self):
        self.all_words = []        # list of all tokens from the dataset
        self.vocab = {}            # set of vocabulary
        self.word_to_idx = {}      # map of words to indices
        self.idx_to_word = {}      # map of indices to words

        self.bigram_counts = defaultdict(int) # count of bigrams
        self.bigram_probs = None

    ###########################################################################
    # Data loading / preprocessing; building bigram probability matrix
    ###########################################################################

    def load_and_preprocess_data(self, file_path):
        """
        Load the data file and preprocess it by:
        - lowercase
        - removing extra whitespace
        - splitting into sentences
        - tokenizing each sentence into words
        - adding start and end tokens
        - building vocabulary
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

        print(f"Tokenized into {len(self.all_words)} words")
        
        # Build vocabulary
        self.vocab = list(set(self.all_words))
        self.vocab.sort()
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(self.vocab)}
        
        print(f"Vocabulary size: {len(self.vocab)}")
    
    def build_bigrams(self):
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
    
    def create_bigram_matrix(self):
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
    
    def train(self, file_path):
        """
        Train the bigram model on the given text file.
        """
        print("Training bigram model...")
        self.load_and_preprocess_data(file_path)
        self.build_bigrams()
        self.create_bigram_matrix()
        print("Training completed!")

    ###########################################################################
    # Text generation
    ###########################################################################

    def generate_text(self, max_length=50, start_word='<START>', temperature=1.0):
        """
        Generate text using the bigram model.
        """
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
    
    def generate_sentences(self, num_sentences=5, max_words_per_sentence=20, temperature=1.0):
        """
        Generate multiple complete sentences.
        """
        sentences = []
        for _ in range(num_sentences):
            sentence = self.generate_text(
                max_length=max_words_per_sentence, 
                start_word='<START>',
                temperature=temperature)
            sentences.append(sentence)
        return sentences
    
    def get_most_likely_words(self, word, top_k=5):
        """
        Get the most likely words to follow a given word.
        """
        if word not in self.word_to_idx:
            print(f"Word '{word}' not in vocabulary")
            return []
        
        word_idx = self.word_to_idx[word]
        probs = self.bigram_probs[word_idx]
        
        # Get top-k most likely next words
        top_probs, top_indices = torch.topk(probs, top_k)
        
        results = []
        for prob, idx in zip(top_probs, top_indices):
            word = self.idx_to_word[idx.item()]
            results.append((word, prob.item()))
        
        return results
    
    def visualize_bigram_heatmap(self, words_to_show=20, figsize=(12, 8)):
        """
        Visualize the bigram probability matrix as a heatmap.
        """
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
    
    def get_model_stats(self):
        """
        Get statistics about the trained model.
        """
        stats = {
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
    
    def calculate_loss(self):
        """
        Calculate the negative log-likelihood loss of the model.
        This measures how well the model predicts the actual bigrams in the training data.
        Lower loss means better predictions.
        """
        print("Calculating model loss...")
        
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

def main():
    parser = argparse.ArgumentParser(description='Austen Word-Level Bigram Model')
    
    # Text generation arguments
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for text generation (default: 1.0)')
    parser.add_argument('--sentences', type=int, default=5,
                        help='Number of sentences to generate (default: 5)')
    parser.add_argument('--max-words', type=int, default=20,
                        help='Maximum words per sentence (default: 20)')
    
    # Helper options
    parser.add_argument('--stats', action='store_true',
                        help='Show model statistics')
    parser.add_argument('--loss', action='store_true',
                        help='Show model loss and perplexity')
    parser.add_argument('--heatmap', action='store_true',
                        help='Generate and save bigram heatmap')
    parser.add_argument('--predictions-for-word', type=str, metavar='WORD',
                        help='Show top 5 words that follow the specified word')
    
    # Model options
    parser.add_argument('--start-word', type=str, default='<START>',
                        help='Starting word for generation (default: <START>)')
    parser.add_argument('--heatmap-words', type=int, default=15,
                        help='Number of words to show in heatmap (default: 15)')
    
    args = parser.parse_args()
    
    print("=== Austen Word-Level Bigram Model ===")
    print(f"Temperature: {args.temperature}")
    print(f"Generating {args.sentences} sentences with max {args.max_words} words each")
    print()
    
    # Train model
    model = WordBigram()
    model.train('austen.txt')
    
    # Show statistics if requested
    if args.stats:
        stats = model.get_model_stats()
        print("\n=== Model Statistics ===")
        print(f"Vocabulary size: {stats['vocabulary_size']} words")
        print(f"Total tokens: {stats['all_words_size']}")
        print(f"Total bigrams: {stats['total_bigram_count']}")
        print(f"Most common words: {stats['most_common_words']}")
        print(f"Most common bigrams: {stats['most_common_bigrams']}")
    
    # Show loss if requested
    if args.loss:
        loss_info = model.calculate_loss()
        print("\n=== Model Loss ===")
        print(f"Average loss (negative log-likelihood): {loss_info['average_loss']:.4f}")
        print(f"Perplexity: {loss_info['perplexity']:.2f}")
    
    # Generate text
    print(f"\n=== Generated Text (Temperature = {args.temperature}) ===")
    sentences = model.generate_sentences(num_sentences=args.sentences, 
                                       max_words_per_sentence=args.max_words,
                                       temperature=args.temperature)
    
    for i, sentence in enumerate(sentences, 1):
        print(f"{i}. {sentence}")
    
    # Show word predictions if requested
    if args.predictions_for_word:
        word = args.predictions_for_word.lower()
        print(f"\n=== Word Predictions for '{word}' ===")
        if word in model.word_to_idx:
            predictions = model.get_most_likely_words(word, top_k=5)
            pred_words = [f"{pred[0]} ({pred[1]:.3f})" for pred in predictions]
            print(f"Top 5 words after '{word}': {', '.join(pred_words)}")
        else:
            print(f"Word '{word}' not found in vocabulary.")
            print("Note: Make sure the word is lowercase and exists in the Austen text.")
    
    # Generate heatmap if requested
    if args.heatmap:
        print("\n=== Creating Bigram Heatmap ===")
        try:
            model.visualize_bigram_heatmap(words_to_show=args.heatmap_words)
            print(f"Heatmap saved as 'bigram_heatmap.png'")
        except Exception as e:
            print(f"Could not create visualization: {e}")
    
    print("\n=== Generation Complete ===")
    print(f"Generated {args.sentences} sentences with temperature {args.temperature}")
    print("Use --help to see all available options")
    print()

if __name__ == "__main__":
    main()