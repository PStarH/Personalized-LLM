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
    return parser.parse_args()

def validate_writing_style_files(writing_style_dirs):
    valid_files = []
    supported_extensions = FileReader.SUPPORTED_EXTENSIONS
    for directory in writing_style_dirs:
        if not os.path.isdir(directory):
            print(f"Writing style directory not found: {directory}")
            continue

        files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and
               os.path.splitext(f)[1].lower() in supported_extensions
        ]

        if not files:
            print(f"No supported writing style files found in directory: {directory}")
            continue

        valid_files.extend(files)

    if not valid_files:
        print("No valid writing style files found. Exiting.")
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
            aggregated_metrics[key] += metrics[key]

    for key in aggregated_metrics:
        aggregated_metrics[key] = aggregated_metrics[key] / count if count > 0 else 0.0

    return aggregated_metrics

def main():
    # Parse command-line arguments
    args = parse_arguments()
    user_directories = args.directories
    writing_style_dirs = args.writing_styles

    # Validate and collect writing style files
    writing_style_files = validate_writing_style_files(writing_style_dirs)

    # Read and aggregate writing style contents
    writing_style_contents = []
    for file_path in writing_style_files:
        content = FileReader.read_file(file_path)
        if content:
            writing_style_contents.append(content)
        else:
            print(f"Skipping file due to read error: {file_path}")

    if not writing_style_contents:
        print("No content to analyze for writing styles. Exiting.")
        exit(1)

    aggregated_writing_style = aggregate_writing_styles(writing_style_contents)
    print(f"Aggregated Writing Style Metrics: {aggregated_writing_style}")

    # Initialize components
    crawler = WebCrawler()
    sentiment_analyzer = SentimentAnalyzer()
    writing_style_analyzer = WritingStyleAnalyzer()  # May not be needed anymore
    retriever = Retriever()
    prompt_generator = PromptGenerator()
    fine_tuner = FineTuner()
    history_embedder = HistoryEmbedder()
    chain_manager = ChainManager()

    for directory in user_directories:
        if not os.path.isdir(directory):
            print(f"User files directory not found: {directory}")
            continue

        user_files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and
               os.path.splitext(f)[1].lower() in FileReader.SUPPORTED_EXTENSIONS
        ]

        if not user_files:
            print(f"No supported user files found in directory: {directory}")
            continue

        for file_path in user_files:
            content = FileReader.read_file(file_path)
            if not content:
                print(f"Skipping file due to read error: {file_path}")
                continue

            # Step 2: Sentiment Analysis
            sentiment = sentiment_analyzer.analyze_sentiment(content)
            print(f"Sentiment for {file_path}: {sentiment}")

            # Step 3: Web Crawling for RAG
            crawled_data = crawler.retrieve_data(query=content)

            # Step 4: Retrieval-Augmented Generation
            relevant_data = retriever.retrieve(crawled_data, query=content)

            # Step 5: Prompt Generation with Aggregated Writing Style
            prompt = prompt_generator.generate_prompt(relevant_data, sentiment, aggregated_writing_style)

            # Step 6: Fine-Tuning the LLM
            fine_tuned_model = fine_tuner.fine_tune(prompt)

            # Step 7: Chain of Thought Management
            thought = chain_manager.generate_thought(prompt, fine_tuned_model)

            # Step 8: History Embedding
            history_embedder.embed(thought)

            # Step 9: Generate Final Response
            response = fine_tuned_model.generate_response(thought)
            print(f"Response for {file_path}:\n{response}\n")

if __name__ == "__main__":
        main()
