# Write like Jane Austen!

This project implements a multi-model text generation system trained on Jane Austen's works. It is inspired by [Andrej Karpathy's makemore library](https://github.com/karpathy/makemore/tree/master), but while makemore is a character-level framework, this project extends it to work at the word-level.

## Features

- **Multiple supported models:** choose any of the supported models for your text generation.
- **Word-level tokenization:** processes text at the word level.
- **Extensible architecture:** easy to add new model types.
- **Memory management:** configurable training data size limits.
- **Text generation:** generate Austen'style text with configurable temperature for creativity control.
- **Visualization:** generate heatmaps of bigram probabilities and word embeddings.
- **Interactive predictions**: Get the most likely words to follow any given word

## Supported Model Types

**Probabilistic bigram model:**
- bigram model that uses count-based probabilities to generate predictions
- fast training and inference, memory efficient
- good baseline for comparison

**Neural-network bigram model:**
- uses word embeddings and neural network
- more expressive than count-based bigram model
- configurable architecture (embedding dimension, learning rate, epochs)

**MLP (Multi-Layer Perceptron) model:**
- word-level language model that uses multiple words as context
- more sophisticated than bigram models by considering longer sequences
- supports mini-batch training for better memory efficiency and faster convergence
- configurable context length (block size), hidden dimensions, and training parameters

## Files

- `main.py`: Main entry point with command-line interface
- `models/`: Directory containing model implementations
  - `probabilistic_bigram.py`: Count-based bigram model
  - `neural_bigram.py`: Neural network bigram model
  - `mlp.py`: Multi-layer perceptron model
- `austen.txt`: The Jane Austen text dataset
- `requirements.txt`: Python dependencies

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command-Line Interface

The system supports a comprehensive command-line interface with multiple model options:

```bash
# Basic usage - generate 5 sentences with probabilistic model
python main.py --model bigram-probabilistic

# Use neural network model with default settings
python main.py --model bigram-neural

# Use MLP model with default settings
python main.py --model mlp

# Neural model with custom parameters
python main.py --model bigram-neural --embedding-dim 128 --learning-rate 0.05 --epochs 200

# MLP model with custom parameters
python main.py --model mlp --block-size 4 --hidden-dim 256 --embedding-dim 128

# MLP model with mini-batch training
python main.py --model mlp --mini-batch-size 32 --epochs 50

# Limit training data to save memory
python main.py --model bigram-probabilistic --max-training-data-size 50000

# Generate creative text with higher temperature
python main.py --model bigram-neural --temperature 1.5 --sentences 3

# Show model statistics and loss
python main.py --model bigram-probabilistic --stats --loss

# Generate heatmap visualization (probabilistic model only)
python main.py --model bigram-probabilistic --heatmap

# Visualize word embeddings (neural model only)
python main.py --model bigram-neural --embeddings

# Show word predictions for a specific word
python main.py --model bigram-probabilistic --predictions-for-word catherine

# Show context predictions for MLP model (requires block_size words)
python main.py --model mlp --predictions-for-word "and the book"

# Combine multiple options
python main.py --model bigram-neural --temperature 1.2 --sentences 4 --max-words 18 --stats --loss --predictions-for-word elizabeth
```

### Available Command-Line Arguments

#### Model Selection
- `--model {bigram-probabilistic,bigram-neural,mlp}`: Type of model to use (required)
- `--data-file FILE`: Path to training data file (default: austen.txt)

#### Data Management
- `--max-training-data-size SIZE`: Maximum number of words to use for training (default: all data)

#### Training Parameters (neural and MLP models)
- `--embedding-dim DIM`: Embedding dimension (default: 64)
- `--learning-rate RATE`: Learning rate (default: 0.1)
- `--epochs EPOCHS`: Number of training epochs (default: 100)

#### MLP Model Parameters (MLP model only)
- `--block-size SIZE`: Context length (number of words to use as input) (default: 3)
- `--hidden-dim DIM`: Hidden layer dimension (default: 128)
- `--mini-batch-size SIZE`: Mini-batch size for training (default: None, uses full batch)

#### Text Generation
- `--temperature TEMPERATURE`: Temperature for text generation (default: 1.0)
- `--sentences SENTENCES`: Number of sentences to generate (default: 5)
- `--max-words MAX_WORDS`: Maximum words per sentence (default: 20)
- `--start-word START_WORD`: Starting word for generation (default: <START>)

#### Analysis
- `--stats`: Show model statistics
- `--loss`: Show model loss and perplexity
- `--predictions-for-word WORD`: Show top 5 words that follow the specified word

#### Visualization
- `--heatmap`: Generate and save bigram heatmap (probabilistic model only)
- `--embeddings`: Visualize word embeddings (neural model only)
- `--heatmap-words WORDS`: Number of words to show in visualizations (default: 15)

### Programmatic Usage

You can also use the models programmatically:

```python
from models.probabilistic_bigram import ProbabilisticBigramModel
from models.neural_bigram import NeuralBigramModel
from models.mlp import MLPModel

# Probabilistic model
prob_model = ProbabilisticBigramModel()
prob_model.train('austen.txt', max_training_data_size=50000)

# Neural network model
neural_model = NeuralBigramModel(embedding_dim=128, learning_rate=0.05, num_epochs=200)
neural_model.train('austen.txt', max_training_data_size=50000)

# MLP model
mlp_model = MLPModel(block_size=4, hidden_dim=256, embedding_dim=128, mini_batch_size=64)
mlp_model.train('austen.txt', max_training_data_size=50000)

# Generate text
text = neural_model.generate_text(max_length=20, start_word='she', temperature=1.2)
print(text)

# Generate text with MLP model (using context)
mlp_text = mlp_model.generate_text(max_length=20, sentence_start='she was', temperature=1.2)
print(mlp_text)

# Get word predictions
predictions = neural_model.get_most_likely_words('catherine', top_k=5)

# Get context predictions for MLP model
mlp_predictions = mlp_model.get_most_likely_words('and the', top_k=5)

# Generate multiple sentences
sentences = neural_model.generate_sentences(num_sentences=3, max_words_per_sentence=15, temperature=0.8)

# Get model statistics
stats = neural_model.get_model_stats()
print(f"Vocabulary size: {stats['vocabulary_size']}")

# Calculate loss
loss_info = neural_model.calculate_loss()
print(f"Perplexity: {loss_info['perplexity']:.2f}")
```

## Model Architectures

### Probabilistic Bigram Model

The probabilistic model uses a count-based approach:

1. **Text Preprocessing**: Converts text to lowercase, splits into sentences, and tokenizes words
2. **Vocabulary Building**: Creates a mapping between words and integer indices
3. **Bigram Counting**: Counts occurrences of word pairs (bigrams) in the training text
4. **Probability Matrix**: Creates a V×V matrix where V is vocabulary size, with each entry representing the probability of one word following another
5. **Text Generation**: Samples words based on the learned probabilities

### Neural Network Bigram Model

The neural model uses learnable embeddings:

1. **Text Preprocessing**: Same as probabilistic model
2. **Vocabulary Building**: Same as probabilistic model
3. **Embedding Layer**: Each word is represented as a dense vector (embedding)
4. **Linear Layer**: Maps embeddings to next-word probabilities
5. **Training**: Uses gradient descent to learn optimal embeddings and weights
6. **Text Generation**: Uses trained neural network to predict next words

### MLP Model

The MLP model is a more sophisticated word-level language model:

1. **Text Preprocessing**: Same as other models, with special handling for context padding
2. **Vocabulary Building**: Same as other models
3. **Context Window**: Uses a configurable number of previous words (block_size) as context
4. **Embedding Layer**: Each word in the context is embedded into dense vectors
5. **Hidden Layer**: Concatenated embeddings pass through a hidden layer with tanh activation
6. **Output Layer**: Maps hidden representation to vocabulary probabilities
7. **Mini-batch Training**: Supports both full-batch and mini-batch training for efficiency
8. **Text Generation**: Uses sliding window approach to generate text with longer context awareness

## Model Statistics

- **Vocabulary Size**: 17,493 unique words
- **Total Bigrams**: 251,167 unique word pairs
- **Training Tokens**: 1,050,812 total words
- **Sentences**: 43,580 sentences from Austen's works

## Temperature Effects

Temperature controls the creativity and randomness of text generation:

- **Temperature < 1.0 (e.g., 0.5, 0.8)**: More conservative, predictable text
- **Temperature = 1.0**: Balanced creativity (default)
- **Temperature > 1.0 (e.g., 1.5, 2.0)**: More creative, surprising text

```bash
# Conservative generation
python main.py --model bigram-probabilistic --temperature 0.8

# Creative generation  
python main.py --model bigram-neural --temperature 1.5
```

## Model Loss

The model calculates its performance using negative log-likelihood loss:

- **Average Loss**: Measures how well the model predicts actual bigrams in the training data (lower is better)
- **Perplexity**: The exponential of average loss - represents the "surprise" of the model (lower is better)

```bash
# Show model loss
python main.py --model bigram-probabilistic --loss
```

Example output:
```
=== Model Loss ===
Average loss (negative log-likelihood): 4.1529
Perplexity: 63.62
```

## Example Output

### Probabilistic Model
```
Generated text: "she must be sure you mean to think to which"

Word predictions for 'catherine':
Top 5 words after 'catherine': s (0.140), was (0.080), and (0.058), <END> (0.046), had (0.042)
```

### Neural Network Model
```
Generated text: "she had been very much pleased with her own"

Word predictions for 'elizabeth':
Top 5 words after 'elizabeth': was (0.082), <END> (0.064), and (0.061), s (0.060), had (0.060)
```

### MLP Model
```
Generated text: "he had a considerable independence besides two good livings"

Context predictions for 'and the book':
Top 5 words after 'and the book': her (0.968), ladies (0.024), many (0.006), about (0.001), room (0.000)
```

## Visualization

### Probabilistic Model
The probabilistic model can generate a heatmap showing the probability relationships between the most common words:

```bash
python main.py --model probabilistic --heatmap
```

This creates a visualization saved as `bigram_heatmap.png`.

### Neural Network Model
The neural model can visualize word embeddings using t-SNE:

```bash
python main.py --model neural --embeddings
```

This creates a visualization saved as `neural_bigram_embeddings.png`.

## References

- [Neural Networks: Zero to Hero - Makemore Part 1: Bigrams](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part1_bigrams.ipynb)
- [Neural Networks: Zero to Hero - Makemore Part 2: MLP](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part2_mlp.ipynb)
- Jane Austen dataset was downloaded from kaggle, [Jane Austen and Charles Dickens](https://www.kaggle.com/datasets/joshmcadams/jane-austin-and-charles-dickens/data)