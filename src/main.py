import os
import argparse
from ai.prompt_generator import PromptGenerator
from ai.fine_tuner import FineTuner
from crawler.web_crawler import WebCrawler
from nlp.sentiment_analysis import SentimentAnalyzer
from nlp.writing_style import WritingStyleAnalyzer
from rag.retriever import Retriever
from history.embed_history import HistoryEmbedder
from cot.chain_manager import ChainManager
from utils.file_reader import FileReader
import logging

def parse_arguments():
    parser = argparse.ArgumentParser(description="Personalized-LLM Main Application")
    parser.add_argument(
        '-d', '--directories',
        nargs='+',
        required=True,
        help='One or more directories containing user files.'
    )
    parser.add_argument(
        '-w', '--writing_styles',
        nargs='+',
        required=True,
        help='One or more directories containing writing style documents.'
    )
    parser.add_argument(
        '-m', '--model_name',
        type=str,
        required=True,
        help='Name of the pre-trained model to fine-tune (e.g., gpt2, bert-base-uncased).'
    )
    parser.add_argument(
        '-e', '--num_train_epochs',
        type=int,
        default=3,
        help='Number of training epochs.'
    )
    parser.add_argument(
        '-b', '--batch_size',
        type=int,
        default=4,
        help='Batch size per device during training.'
    )
    parser.add_argument(
        '-lr', '--learning_rate',
        type=float,
        default=5e-5,
        help='Learning rate for the optimizer.'
    )
    parser.add_argument(
        '-c', '--block_size',
        type=int,
        default=128,
        help='Maximum sequence length after tokenization.'
    )
    parser.add_argument(
        '--max_train_samples',
        type=int,
        default=None,
        help='Maximum number of training samples.'
    )
    parser.add_argument(
        '--max_eval_samples',
        type=int,
        default=None,
        help='Maximum number of evaluation samples.'
    )
    parser.add_argument(
        '--no_clean',
        action='store_true',
        help='Disable data cleaning.'
    )
    parser.add_argument(
        '--augment',
        action='store_true',
        help='Enable data augmentation.'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging.'
    )
    return parser.parse_args()

def setup_logging(verbose: bool):
    """
    Sets up the logging configuration.

    Args:
        verbose (bool): If True, set logging level to INFO, else WARNING.
    """
    logging_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=logging_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def validate_writing_style_files(writing_style_dirs):
    valid_files = []
    supported_extensions = FileReader.SUPPORTED_EXTENSIONS
    for directory in writing_style_dirs:
        if not os.path.isdir(directory):
            logging.warning(f"Writing style directory not found: {directory}")
            continue

        files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and
               os.path.splitext(f)[1].lower() in supported_extensions
        ]

        if not files:
            logging.warning(f"No supported writing style files found in directory: {directory}")
            continue

        valid_files.extend(files)

    if not valid_files:
        logging.error("No valid writing style files found. Exiting.")
        exit(1)

    return valid_files

def aggregate_writing_styles(writing_style_contents):
    """
    Aggregates multiple writing style analyses into a single metric.
    This is a simple average; more sophisticated methods can be implemented.

    Args:
        writing_style_contents (List[str]): List of writing style document contents.

    Returns:
        dict: Aggregated writing style metrics.
    """
    analyzer = WritingStyleAnalyzer()
    aggregated_metrics = {
        'average_sentence_length': 0.0,
        'average_word_length': 0.0,
        'lexical_diversity': 0.0,
        'readability_score': 0.0
    }
    count = len(writing_style_contents)

    for content in writing_style_contents:
        metrics = analyzer.analyze_writing_style(content)
        for key in aggregated_metrics:
            aggregated_metrics[key] += metrics.get(key, 0.0)

    for key in aggregated_metrics:
        aggregated_metrics[key] = aggregated_metrics[key] / count if count > 0 else 0.0

    return aggregated_metrics

def main():
    # Parse command-line arguments
    args = parse_arguments()
    setup_logging(args.verbose)
    user_directories = args.directories
    writing_style_dirs = args.writing_styles
    model_name = args.model_name
    num_train_epochs = args.num_train_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    block_size = args.block_size
    max_train_samples = args.max_train_samples
    max_eval_samples = args.max_eval_samples
    do_clean = not args.no_clean
    do_augment = args.augment

    # Validate and collect writing style files
    writing_style_files = validate_writing_style_files(writing_style_dirs)

    # Read and aggregate writing style contents
    writing_style_contents = []
    for file_path in writing_style_files:
        content = FileReader.read_file(file_path)
        if content:
            writing_style_contents.append(content)
            logging.info(f"Loaded writing style file: {file_path}")
        else:
            logging.warning(f"Skipping file due to read error: {file_path}")

    if not writing_style_contents:
        logging.error("No content to analyze for writing styles. Exiting.")
        exit(1)

    aggregated_writing_style = aggregate_writing_styles(writing_style_contents)
    logging.info(f"Aggregated Writing Style Metrics: {aggregated_writing_style}")

    # Initialize components
    crawler = WebCrawler()
    sentiment_analyzer = SentimentAnalyzer()
    retriever = Retriever()
    prompt_generator = PromptGenerator()
    fine_tuner = FineTuner(
        model_name=model_name,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        block_size=block_size,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
        do_clean=do_clean,
        do_augment=do_augment
    )
    history_embedder = HistoryEmbedder()
    chain_manager = ChainManager()

    # Collect training data from user files
    training_texts = []

    for directory in user_directories:
        if not os.path.isdir(directory):
            logging.warning(f"User files directory not found: {directory}")
            continue

        user_files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and
               os.path.splitext(f)[1].lower() in FileReader.SUPPORTED_EXTENSIONS
        ]

        if not user_files:
            logging.warning(f"No supported user files found in directory: {directory}")
            continue

        for file_path in user_files:
            content = FileReader.read_file(file_path)
            if not content:
                logging.warning(f"Skipping file due to read error: {file_path}")
                continue

            training_texts.append(content)
            logging.info(f"Added training text from: {file_path}")

    if training_texts:
        # Perform fine-tuning with all collected training texts
        fine_tuned_model = fine_tuner.fine_tune(training_texts)
        logging.info("Fine-tuning completed successfully.")

        # Generate responses using the fine-tuned model
        for file_path in user_files:
            content = FileReader.read_file(file_path)
            if not content:
                logging.warning(f"Skipping file due to read error: {file_path}")
                continue

            # Step 2: Sentiment Analysis
            sentiment = sentiment_analyzer.analyze_sentiment(content)
            logging.info(f"Sentiment for {file_path}: {sentiment}")

            # Step 3: Web Crawling for RAG
            crawled_data = crawler.retrieve_data(query=content)

            # Step 4: Retrieval-Augmented Generation
            relevant_data = retriever.retrieve(crawled_data, query=content)

            # Step 5: Prompt Generation with Aggregated Writing Style
            prompt = prompt_generator.generate_prompt(relevant_data, sentiment, aggregated_writing_style)

            # Step 6: Chain of Thought Management
            thought = chain_manager.generate_thought(prompt, fine_tuned_model)

            # Step 7: History Embedding
            history_embedder.embed(thought)

            # Step 8: Generate Final Response
            response = fine_tuned_model.generate_response(thought)
            print(f"Response for {file_path}:\n{response}\n")
    else:
        logging.error("No training texts collected. Skipping fine-tuning.")

if __name__ == "__main__":
    main()
