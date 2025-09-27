# Austen Word-Level Bigram Model

A word-level bigram language model trained on Jane Austen's novels, implemented in PyTorch. This project is inspired by [Andrej Karpathy's nn-zero-to-hero series](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part1_bigrams.ipynb), but adapted to work at the word level instead of character level.

## Features

- **Word-level tokenization**: Processes text at the word level for more meaningful language modeling
- **Bigram probability matrix**: Uses PyTorch tensors for efficient probability calculations
- **Text generation**: Generate Austen-style text with configurable temperature for creativity control
- **Visualization**: Create heatmaps of bigram probabilities
- **Interactive predictions**: Get the most likely words to follow any given word

## Files

- `bigram.py`: Main implementation with command-line interface
- `austen.txt`: The Jane Austen text dataset
- `requirements.txt`: Python dependencies
- `test_model.py`: Test suite to verify correctness

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command-Line Interface

The model now supports a comprehensive command-line interface:

```bash
# Basic usage - generate 5 sentences with default settings
python bigram.py

# Generate 3 sentences with more conservative (lower temperature) text
python bigram.py --temperature 0.8 --sentences 3

# Generate creative text with longer sentences
python bigram.py --temperature 1.5 --max-words 25 --sentences 2

# Show model statistics
python bigram.py --stats

# Show model loss and perplexity
python bigram.py --loss

# Generate heatmap visualization
python bigram.py --heatmap

# Show word predictions for a specific word
python bigram.py --predictions-for-word catherine

# Combine multiple options
python bigram.py --temperature 1.2 --sentences 4 --max-words 18 --stats --loss --predictions-for-word elizabeth
```

### Available Command-Line Arguments

- `--temperature TEMPERATURE`: Temperature for text generation (default: 1.0)
- `--sentences SENTENCES`: Number of sentences to generate (default: 5)
- `--max-words MAX_WORDS`: Maximum words per sentence (default: 20)
- `--stats`: Show model statistics
- `--loss`: Show model loss and perplexity
- `--heatmap`: Generate and save bigram heatmap
- `--predictions-for-word WORD`: Show top 5 words that follow the specified word
- `--start-word START_WORD`: Starting word for generation (default: <START>)
- `--heatmap-words HEATMAP_WORDS`: Number of words to show in heatmap (default: 15)

### Programmatic Usage

You can also use the model programmatically:

```python
from bigram import WordBigram

# Initialize and train the model
model = WordBigram()
model.train('austen.txt')

# Generate text
text = model.generate_text(max_length=20, start_word='she', temperature=1.2)
print(text)

# Get word predictions
predictions = model.get_most_likely_words('catherine', top_k=5)

# Generate multiple sentences
sentences = model.generate_sentences(num_sentences=3, max_words_per_sentence=15, temperature=0.8)
```

## Model Architecture

The model uses a simple bigram approach:

1. **Text Preprocessing**: Converts text to lowercase, splits into sentences, and tokenizes words
2. **Vocabulary Building**: Creates a mapping between words and integer indices
3. **Bigram Counting**: Counts occurrences of word pairs (bigrams) in the training text
4. **Probability Matrix**: Creates a V×V matrix where V is vocabulary size, with each entry representing the probability of one word following another
5. **Text Generation**: Samples words based on the learned probabilities

## Key Differences from Character-Level Model

- **Tokenization**: Uses words instead of characters as the basic unit
- **Vocabulary Size**: Much larger vocabulary (17,493 words vs. ~27 characters)
- **Context**: Each prediction considers only the previous word (bigram model)
- **Output Quality**: Generated text is more coherent at the word level but may lack longer-term structure

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
python bigram.py --temperature 0.8

# Creative generation  
python bigram.py --temperature 1.5
```

## Model Loss

The model calculates its performance using negative log-likelihood loss:

- **Average Loss**: Measures how well the model predicts actual bigrams in the training data (lower is better)
- **Perplexity**: The exponential of average loss - represents the "surprise" of the model (lower is better)

```bash
# Show model loss
python bigram.py --loss
```

Example output:
```
=== Model Loss ===
Average loss (negative log-likelihood): 4.1529
Perplexity: 63.62
```

## Example Output

```
Generated text: "she must be sure you mean to think to which"

Word predictions for 'catherine':
Top 5 words after 'catherine': s (0.140), was (0.080), and (0.058), <END> (0.046), had (0.042)

Word predictions for 'elizabeth':
Top 5 words after 'elizabeth': was (0.082), <END> (0.064), and (0.061), s (0.060), had (0.060)
```

## Visualization

The model can generate a heatmap showing the probability relationships between the most common words:

```python
model.visualize_bigram_heatmap(words_to_show=20)
```

This creates a visualization saved as `bigram_heatmap.png`.

## References

- [Neural Networks: Zero to Hero - Makemore Part 1: Bigrams](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part1_bigrams.ipynb)
- Jane Austen dataset was downloaded from kaggle, [Jane Austen and Charles Dickens](https://www.kaggle.com/datasets/joshmcadams/jane-austin-and-charles-dickens/data)