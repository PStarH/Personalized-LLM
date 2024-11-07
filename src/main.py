import os
import argparse
import logging
from ai.prompt_generator import PromptGenerator
from ai.fine_tuner import EnhancedFineTuner, PersonalityConfig
from ai.llm_providers import (
    LLMProvider,
    LLMProviderRegistry
)
from crawler.web_crawler import WebCrawler
from nlp.sentiment_analysis import EnhancedSentimentAnalyzer
from nlp.writing_style import WritingStyleAnalyzer
from rag.retriever import EnhancedRetriever
from history.embed_history import HistoryEmbedder
from cot.chain_manager import AdvancedHumanChainManager
from utils.file_reader import FileReader
import json
from datetime import datetime
import importlib.util
from typing import Type

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
    parser.add_argument(
        '--personalize',
        type=str,
        help='Path to a JSON file containing personality configuration.'
    )
    parser.add_argument(
        '--llm_provider',
        type=str,
        default='openai',
        choices=LLMProviderRegistry.list_providers(),
        help='LLM provider to use (e.g., ollama, openai, anthropic, google, custom).'
    )
    parser.add_argument(
        '--provider_module',
        type=str,
        default=None,
        help='Path to the custom provider module (if using a custom provider).'
    )
    parser.add_argument(
        '--provider_class',
        type=str,
        default=None,
        help='Name of the custom provider class in the module.'
    )
    parser.add_argument(
        '--api_key',
        type=str,
        default=None,
        help='API key for the selected LLM provider, if required.'
    )
    parser.add_argument(
        '--credentials_path',
        type=str,
        default=None,
        help='Path to Google Cloud credentials JSON file (if using Google provider).'
    )
    parser.add_argument(
        '--model_provider',
        type=str,
        default=None,
        help='Model provider name for LangChain integrations (e.g., huggingface, cohere).'
    )
    parser.add_argument(
        '--model_kwargs',
        type=str,
        default=None,
        help='JSON string of additional keyword arguments for the model provider.'
    )
    parser.add_argument(
        '--enable_embedding',
        action='store_true',
        help='Enable text embedding functionality.'
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

def load_custom_provider(module_path: str, class_name: str) -> Type[LLMProvider]:
    """
    Dynamically loads a custom LLM provider from the specified module.

    Args:
        module_path (str): File path to the custom provider module.
        class_name (str): Name of the provider class within the module.

    Returns:
        Type[LLMProvider]: The loaded provider class.

    Raises:
        Exception: If the module or class cannot be loaded.
    """
    try:
        spec = importlib.util.spec_from_file_location("custom_provider", module_path)
        custom_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_module)
        provider_class = getattr(custom_module, class_name)
        if not issubclass(provider_class, LLMProvider):
            raise TypeError(f"{class_name} does not inherit from LLMProvider.")
        # Register the custom provider
        LLMProviderRegistry.register_provider('custom', provider_class)
        logging.info(f"Custom provider '{class_name}' loaded and registered successfully.")
        return provider_class
    except Exception as e:
        logging.error(f"Failed to load custom provider: {e}")
        raise

def get_llm_provider(provider_name: str, api_key: str = None, credentials_path: str = None, provider_module: str = None, provider_class: str = None, **kwargs) -> LLMProvider:
    """
    Factory method to get the appropriate LLM provider instance.

    Args:
        provider_name (str): Name of the LLM provider.
        api_key (str, optional): API key for the provider if required.
        credentials_path (str, optional): Path to Google Cloud credentials (if Google provider).
        provider_module (str, optional): Path to the custom provider module.
        provider_class (str, optional): Name of the custom provider class.
        **kwargs: Additional keyword arguments for the provider.

    Returns:
        LLMProvider: An instance of a subclass of LLMProvider.
    """
    if provider_name == 'custom':
        if not provider_module or not provider_class:
            logging.error("Both --provider_module and --provider_class must be specified for custom providers.")
            exit(1)
        load_custom_provider(provider_module, provider_class)

    provider_cls = LLMProviderRegistry.get_provider(provider_name)

    try:
        if provider_name == 'ollama':
            return provider_cls(model=kwargs.get('model', 'default'), api_url=kwargs.get('api_url', 'http://localhost:11434'))
        elif provider_name == 'openai':
            if not api_key:
                logging.error("API key is required for OpenAI provider.")
                exit(1)
            return provider_cls(api_key=api_key, model=kwargs.get('model', 'text-davinci-003'))
        elif provider_name == 'anthropic':
            if not api_key:
                logging.error("API key is required for Anthropic provider.")
                exit(1)
            return provider_cls(api_key=api_key, model=kwargs.get('model', 'claude-v1.3'))
        elif provider_name == 'google':
            if not credentials_path:
                logging.error("Credentials path is required for Google provider.")
                exit(1)
            return provider_cls(credentials_path=credentials_path, model=kwargs.get('model', 'text-bison-001'))
        elif provider_name == 'custom':
            # Assuming custom provider may require additional kwargs
            return provider_cls(**kwargs)
        else:
            # For any other providers, assume they require api_key
            if issubclass(provider_cls, LLMProvider):
                try:
                    # Check if the provider's constructor accepts api_key
                    if 'api_key' in provider_cls.__init__.__code__.co_varnames:
                        return provider_cls(api_key=api_key)
                    else:
                        return provider_cls()
                except TypeError as e:
                    logging.error(f"Error initializing provider '{provider_name}': {e}")
                    exit(1)
            else:
                logging.error(f"Provider '{provider_name}' is not a valid LLMProvider subclass.")
                exit(1)
    except Exception as e:
        logging.error(f"Failed to initialize provider '{provider_name}': {e}")
        exit(1)

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
    llm_provider_name = args.llm_provider
    api_key = args.api_key
    provider_module = args.provider_module
    provider_class = args.provider_class
    credentials_path = args.credentials_path
    model_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else {}
    enable_embedding = args.enable_embedding

    # Initialize LLM Provider
    llm_provider = get_llm_provider(
        provider_name=llm_provider_name, 
        api_key=api_key, 
        credentials_path=credentials_path,
        provider_module=provider_module, 
        provider_class=provider_class,
        model=model_name,
        **model_kwargs
    )

    # Load personality configuration if provided
    if args.personalize:
        try:
            with open(args.personalize, 'r') as f:
                personality_data = json.load(f)
            personality = PersonalityConfig(**personality_data)
            logging.info(f"Loaded personality configuration from {args.personalize}")
        except Exception as e:
            logging.error(f"Failed to load personality configuration: {e}")
            logging.info("Using default personality configuration.")
            personality = PersonalityConfig()
    else:
        personality = PersonalityConfig(
            formality=0.3,
            expressiveness=0.8,
            humor=0.6,
            empathy=0.9,
            verbosity=0.4
        )

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
    sentiment_analyzer = EnhancedSentimentAnalyzer()
    retriever = EnhancedRetriever()
    prompt_generator = PromptGenerator()
    history_embedder = HistoryEmbedder()
    fine_tuner = EnhancedFineTuner(
        model_name=model_name,
        personality_config=personality,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        block_size=block_size,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
        do_clean=do_clean,
        do_augment=do_augment
    )
    chain_manager = AdvancedHumanChainManager(
        model=fine_tuner.model,  # Pass the fine-tuned model directly
        cache_dir="cache/chains",
        personality_type="balanced"
    )

    # Initialize Embedding Provider if enabled
    if enable_embedding:
        try:
            embedding = llm_provider.embed_text("Initialize Embedding")
            if embedding is not None:
                logging.info("Embedding functionality is enabled and working.")
            else:
                logging.warning("Embedding functionality is enabled but failed to initialize.")
        except Exception as e:
            logging.error(f"Embedding initialization failed: {e}")
            exit(1)

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
        try:
            # Perform fine-tuning with all collected training texts
            logging.info("Starting fine-tuning process...")
            fine_tuned_model = fine_tuner.fine_tune(training_texts)
            logging.info("Fine-tuning completed successfully.")

            # Save the fine-tuned model
            model_save_path = "fine_tuned_model"
            fine_tuned_model.save_pretrained(model_save_path)
            logging.info(f"Fine-tuned model saved at: {model_save_path}")

        except Exception as e:
            logging.error(f"An error occurred during fine-tuning: {e}")
            exit(1)

        # Generate responses using the fine-tuned model
        for directory in user_directories:
            user_files = [
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, f)) and
                   os.path.splitext(f)[1].lower() in FileReader.SUPPORTED_EXTENSIONS
            ]

            for file_path in user_files:
                content = FileReader.read_file(file_path)
                if not content:
                    logging.warning(f"Skipping file due to read error: {file_path}")
                    continue

                # Step 2: Sentiment Analysis
                sentiment_result = sentiment_analyzer.analyze_sentiment(content, include_aspects=True)
                if sentiment_result:
                    sentiment = sentiment_result.sentiment
                    logging.info(f"Sentiment for {file_path}: {sentiment}")
                else:
                    sentiment = "Neutral"
                    logging.info(f"Sentiment analysis failed for {file_path}. Defaulting to Neutral.")

                # Step 3: Web Crawling for RAG
                crawled_data = crawler.retrieve_data(query=content)

                # Step 4: Retrieval-Augmented Generation
                relevant_data = retriever.retrieve(crawled_data, query=content)

                # Step 5: Prompt Generation with Aggregated Writing Style
                prompt = prompt_generator.generate_prompt(relevant_data, sentiment, aggregated_writing_style)

                # Step 6: Chain of Thought Management
                recognized_history = [{"text": t.content} for t in history_embedder.thoughts]
                thought_process = chain_manager.generate_thought(prompt, recognized_history)

                # Update history with the new thought
                history_embedder.add_thought(
                    thought_content=thought_process,
                    tags=["chain_of_thought"],
                    metadata={"source": "ChainManager"}
                )

                # Step 7: History Embedding
                if enable_embedding:
                    embedding = llm_provider.embed_text(thought_process)
                    if embedding:
                        history_embedder.embed(
                            thought_content=thought_process,
                            tags=["generated_thought_embedding"],
                            metadata={"timestamp": datetime.now().isoformat(), "embedding": embedding}
                        )
                    else:
                        logging.warning(f"Failed to embed thought for {file_path}. Skipping embedding step.")
                else:
                    history_embedder.embed(
                        thought_content=thought_process,
                        tags=["generated_thought"],
                        metadata={"timestamp": datetime.now().isoformat()}
                    )

                # Step 8: Generate Final Response
                try:
                    response = llm_provider.generate_response(thought_process)
                    print(f"Response for {file_path}:\n{response}\n")
                except Exception as e:
                    logging.error(f"Failed to generate response for {file_path}: {e}")
    else:
        logging.error("No training texts collected. Skipping fine-tuning.")

if __name__ == "__main__":
    main()

