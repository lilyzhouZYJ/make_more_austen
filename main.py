#!/usr/bin/env python3
"""
Main entry point for the Austen text generation project.

This script provides a command-line interface for training and using different
language models to generate Austen-style text.
"""

import argparse
import sys
from typing import Optional

from models.probabilistic_bigram import ProbabilisticBigramModel
from models.neural_bigram import NeuralBigramModel


def get_model(model_type: str, **kwargs):
    """
    Factory function to create model instances.
    
    Args:
        model_type: Type of model to create ('probabilistic' or 'neural')
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Model instance
        
    Raises:
        ValueError: If model_type is not recognized
    """
    if model_type == 'probabilistic':
        return ProbabilisticBigramModel()
    elif model_type == 'neural':
        embedding_dim = kwargs.get('embedding_dim', 64)
        learning_rate = kwargs.get('learning_rate', 0.1)
        num_epochs = kwargs.get('num_epochs', 100)
        return NeuralBigramModel(
            embedding_dim=embedding_dim,
            learning_rate=learning_rate,
            num_epochs=num_epochs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Available types: 'probabilistic', 'neural'")


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description='Austen Text Generation with Multiple Language Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate text with probabilistic bigram model (default)
  python main.py
  
  # Use neural network model with custom parameters
  python main.py --model neural --embedding-dim 128 --learning-rate 0.05 --epochs 200
  
  # Limit training data to save memory
  python main.py --max-training-data-size 50000
  
  # Generate creative text with higher temperature
  python main.py --model neural --temperature 1.5 --sentences 3
  
  # Show model statistics and loss
  python main.py --stats --loss
        """
    )
    
    # Model selection arguments
    parser.add_argument('--model', type=str, default='probabilistic',
                       choices=['probabilistic', 'neural'],
                       help='Type of model to use (default: probabilistic)')
    
    # Data arguments
    parser.add_argument('--data-file', type=str, default='austen.txt',
                       help='Path to training data file (default: austen.txt)')
    parser.add_argument('--max-training-data-size', type=int, default=None,
                       help='Maximum number of words to use for training (default: all data)')
    
    # Neural network specific arguments
    parser.add_argument('--embedding-dim', type=int, default=64,
                       help='Embedding dimension for neural model (default: 64)')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                       help='Learning rate for neural model (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs for neural model (default: 100)')
    
    # Text generation arguments
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for text generation (default: 1.0)')
    parser.add_argument('--sentences', type=int, default=5,
                       help='Number of sentences to generate (default: 5)')
    parser.add_argument('--max-words', type=int, default=20,
                       help='Maximum words per sentence (default: 20)')
    parser.add_argument('--start-word', type=str, default='<START>',
                       help='Starting word for generation (default: <START>)')
    
    # Analysis arguments
    parser.add_argument('--stats', action='store_true',
                       help='Show model statistics')
    parser.add_argument('--loss', action='store_true',
                       help='Show model loss and perplexity')
    parser.add_argument('--predictions-for-word', type=str, metavar='WORD',
                       help='Show top 5 words that follow the specified word')
    
    # Visualization arguments
    parser.add_argument('--heatmap', action='store_true',
                       help='Generate and save bigram heatmap (probabilistic model only)')
    parser.add_argument('--heatmap-words', type=int, default=15,
                       help='Number of words to show in heatmap (default: 15)')
    
    args = parser.parse_args()
    
    print("=== Austen Text Generation ===")
    print(f"Model type: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Generating {args.sentences} sentences with max {args.max_words} words each")
    if args.max_training_data_size:
        print(f"Training data limit: {args.max_training_data_size} words")
    print()
    
    try:
        # Create and train model
        model_kwargs = {
            'embedding_dim': args.embedding_dim,
            'learning_rate': args.learning_rate,
            'num_epochs': args.epochs
        }
        
        model = get_model(args.model, **model_kwargs)
        model.train(args.data_file, args.max_training_data_size)
        
        # Show statistics if requested
        if args.stats:
            stats = model.get_model_stats()
            print("\n=== Model Statistics ===")
            print(f"Model type: {stats['model_type']}")
            print(f"Vocabulary size: {stats['vocabulary_size']} words")
            print(f"Total tokens: {stats['all_words_size']}")
            print(f"Most common words: {stats['most_common_words']}")
            
            if 'total_bigram_count' in stats:
                print(f"Total bigrams: {stats['total_bigram_count']}")
                print(f"Most common bigrams: {stats['most_common_bigrams']}")
            
            if 'embedding_dimension' in stats:
                print(f"Embedding dimension: {stats['embedding_dimension']}")
                print(f"Learning rate: {stats['learning_rate']}")
                print(f"Training epochs: {stats['training_epochs']}")
        
        # Show loss if requested
        if args.loss:
            loss_info = model.calculate_loss()
            print("\n=== Model Loss ===")
            print(f"Average loss (negative log-likelihood): {loss_info['average_loss']:.4f}")
            print(f"Perplexity: {loss_info['perplexity']:.2f}")
        
        # Generate text
        print(f"\n=== Generated Text (Temperature = {args.temperature}) ===")
        sentences = model.generate_sentences(
            num_sentences=args.sentences, 
            max_words_per_sentence=args.max_words,
            temperature=args.temperature
        )
        
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
        
        # Generate visualizations
        if args.heatmap and args.model == 'probabilistic':
            print("\n=== Creating Bigram Heatmap ===")
            try:
                model.visualize_bigram_heatmap(words_to_show=args.heatmap_words)
                print(f"Heatmap saved as 'bigram_heatmap.png'")
            except Exception as e:
                print(f"Could not create heatmap: {e}")
        elif args.heatmap and args.model == 'neural':
            print("Heatmap visualization is only available for probabilistic model.")
        
        print("\n=== Generation Complete ===")
        print(f"Generated {args.sentences} sentences with {args.model} model (temperature {args.temperature})")
        print("Use --help to see all available options")
        print()
        
    except FileNotFoundError:
        print(f"Error: Data file '{args.data_file}' not found.")
        print("Please make sure the file exists or specify a different file with --data-file.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
