# Personalized-LLM

*Making LLM results more accurate through enhanced data retrieval, contextual understanding, and personalized responses.*

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## Introduction

**Personalized-LLM** is a sophisticated framework designed to enhance the accuracy and relevance of Large Language Models (LLMs) by integrating advanced data retrieval methods, contextual analysis, and personalized response generation. Leveraging web crawling for updated information, Retrieval-Augmented Generation (RAG) by reading local files, embedding history, and employing a Chain of Thoughts methodology, Personalized-LLM ensures that responses are not only accurate but also tailored to the user's specific needs and preferences.

## Features

- **Web Crawling:** Retrieves the latest information from the web using Bing to ensure up-to-date responses.
- **Retrieval-Augmented Generation (RAG):** Enhances response generation by reading and analyzing local files on the user's computer.
- **History Embedding:** Maintains a history of interactions to provide contextually aware responses.
- **Chain of Thoughts (CoT):** Implements a thought process management system to simulate human-like reasoning.
- **Fine-Tuning with NLP:** Utilizes Natural Language Processing techniques to analyze semantic feelings and optimize prompts for the LLM.
- **Sentiment Analysis:** Determines the sentiment of user inputs to tailor responses appropriately.
- **Writing Style Analysis:** Aggregates writing style metrics to customize the response style.
- **Personality Configuration:** Allows customization of the model's personality traits for more personable interactions.
- **Caching Mechanism:** Implements an efficient caching system to optimize performance and response times.
- **Batch Processing Optimization:** Enhances processing efficiency for handling multiple inputs simultaneously.

## Technologies Used

**Personalized-LLM** leverages a combination of cutting-edge technologies, libraries, and frameworks to deliver its comprehensive feature set:

### Programming Language
- **Python 3.7+:** The primary language used for developing Personalized-LLM due to its extensive support for machine learning and natural language processing tasks.

### Machine Learning & NLP Libraries
- **Transformers (Hugging Face):** Utilized for accessing and fine-tuning state-of-the-art LLMs such as GPT-2, enabling personalized response generation. Transformers provide a vast array of pre-trained models and functionalities for natural language understanding and generation.
- **SentenceTransformers:** Facilitates the creation of high-quality sentence embeddings, crucial for effective retrieval and similarity matching in RAG. It extends the capabilities of Transformers by allowing easy computation of dense vector representations of sentences.
- **FAISS (Facebook AI Similarity Search):** Employed for efficient similarity search and clustering of high-dimensional vectors, enhancing the performance of the retrieval system. FAISS optimizes the search process in large-scale embedding spaces.
- **NLTK (Natural Language Toolkit):** Provides tools for text processing, including tokenization and sentiment analysis, essential for understanding and generating human-like text. NLTK offers a comprehensive suite of libraries and datasets for various NLP tasks.
- **TensorFlow/PyTorch:** Backend frameworks used by Transformers and SentenceTransformers for model training and inference. They provide the foundational infrastructure for building and deploying deep learning models.

### Data Handling & Storage
- **SQLite:** Manages metadata storage, ensuring efficient retrieval and organization of data related to retrieved passages and history embeddings. SQLite offers a lightweight, disk-based database solution without the need for a separate server.
- **JSON:** Standard format for storing configurations, history backups, and other structured data within the application. JSON's flexibility allows for easy serialization and deserialization of complex data structures.

### Web Technologies
- **Requests:** Facilitates HTTP requests for web crawling tasks, enabling the retrieval of real-time information from the internet. Requests simplifies the process of interacting with web pages and APIs.
- **BeautifulSoup:** Parses HTML content fetched from web pages, extracting relevant textual information for further processing. BeautifulSoup allows for easy navigation and manipulation of HTML and XML documents.

### Caching & Optimization
- **FAISS Indexing:** Enhances the speed and scalability of similarity searches within large datasets. By organizing embeddings into indexes, FAISS reduces the computational overhead of similarity queries.
- **LRU Cache (Least Recently Used):** Implements caching mechanisms to store frequent queries and responses, reducing redundant computations and improving response times. This caching strategy ensures that the most recently accessed data remains readily available.

### Additional Tools
- **Docx:** Handles reading and parsing of `.docx` files, allowing the system to analyze documents in Microsoft Word format. Docx enables seamless integration with commonly used document formats.
- **Logging:** Comprehensive logging setup ensures that all operations are tracked, aiding in debugging and performance monitoring. Logs provide insights into the system's behavior and facilitate troubleshooting.
- **Dataclasses:** Simplifies the creation of data structures for managing thoughts, sentiments, and other entities within the application. Dataclasses reduce boilerplate code and enhance code readability.

### Development & Testing
- **Git:** Version control system used for managing codebase changes and collaboration. Git enables efficient tracking of modifications and supports collaborative development workflows.
- **Virtualenv:** Manages project-specific Python environments, ensuring dependency isolation. Virtualenv prevents dependency conflicts by creating isolated environments for different projects.
- **Pip:** Python package installer used for managing project dependencies listed in `requirements.txt`. Pip streamlines the installation and management of Python packages essential for the project.

### Deployment
- **Docker (Optional):** Containerization for deploying Personalized-LLM in various environments with ease, ensuring consistency across development and production setups. Docker containers encapsulate the application and its dependencies, facilitating scalable and portable deployments.

### Testing Frameworks
- **PyTest:** Utilized for writing and running automated tests to ensure code reliability and functionality. PyTest offers a simple yet powerful framework for unit and integration testing.
- **Mocking Libraries:** Employed to simulate external dependencies and isolate components during testing. Mocking ensures that tests focus on specific functionalities without external interference.

### Continuous Integration/Continuous Deployment (CI/CD)
- **GitHub Actions:** Configured to automate testing, building, and deployment processes upon code changes. GitHub Actions streamlines the CI/CD pipeline, enhancing development efficiency and reliability.

### Documentation Tools
- **Sphinx:** Used for generating comprehensive documentation from the codebase, ensuring that all functionalities are well-documented and accessible. Sphinx supports reStructuredText and integrates with various documentation formats.

By integrating these technologies, Personalized-LLM achieves a balance between performance, scalability, and flexibility, enabling it to deliver highly personalized and contextually relevant responses.

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/PStarH/Personalized-LLM.git
   cd Personalized-LLM
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK Data:**
   ```bash
   python -m nltk.downloader punkt stopwords
   ```

## Usage

### Running the Main Application

```bash
python src/main.py -d /path/to/user/files -w /path/to/writing/styles -m gpt2
```

#### Command-Line Arguments

- `-d`, `--directories`: One or more directories containing user files.
- `-w`, `--writing_styles`: One or more directories containing writing style documents.
- `-m`, `--model_name`: Name of the pre-trained model to fine-tune (e.g., gpt2, bert-base-uncased).
- `-e`, `--num_train_epochs`: Number of training epochs (default: 3).
- `-b`, `--batch_size`: Batch size per device during training (default: 4).
- `-lr`, `--learning_rate`: Learning rate for the optimizer (default: 5e-5).
- `-c`, `--block_size`: Maximum sequence length after tokenization (default: 128).
- `--max_train_samples`: Maximum number of training samples.
- `--max_eval_samples`: Maximum number of evaluation samples.
- `--no_clean`: Disable data cleaning.
- `--augment`: Enable data augmentation.
- `--verbose`: Enable verbose logging.
- `--personalize`: Path to a JSON file containing personality configuration.

#### Example

```bash
python src/main.py \
    -d ./user_files \
    -w ./writing_styles \
    -m gpt2 \
    -e 5 \
    --augment \
    --verbose \
    --personalize ./configs/personality.json
```

### Generating Responses

After fine-tuning, the model can generate responses based on user inputs:

```bash
python src/main.py
```

**Sample Output:**
```bash
Retrieved Passages:

1. Passage (Score: 0.85):
Content: Artificial Intelligence (AI) is transforming the world in unprecedented ways.

...

Response for ./user_files/input1.txt:
[Generated response here]
```

## Project Structure

```plaintext
Personalized-LLM/
├── src/
│   ├── ai/
│   │   ├── fine_tuner.py
│   │   └── prompt_generator.py
│   ├── cot/
│   │   └── chain_manager.py
│   ├── crawler/
│   │   └── web_crawler.py
│   ├── history/
│   │   └── embed_history.py
│   ├── history_embedder.py
│   ├── main.py
│   ├── nlp/
│   │   ├── sentiment_analysis.py
│   │   └── writing_style.py
│   └── rag/
│       └── retriever.py
├── utils/
│   └── file_reader.py
├── README.md
├── requirements.txt
└── configs/
    └── personality.json
```

### Description of Key Components

- **src/**: Contains the main source code of the application.
  - **ai/**: Modules related to artificial intelligence functionalities.
    - `fine_tuner.py`: Handles the fine-tuning of language models.
    - `prompt_generator.py`: Generates prompts based on context and sentiment.
  - **cot/**: Modules for Chain of Thoughts processing.
    - `chain_manager.py`: Manages complex reasoning chains for response generation.
  - **crawler/**: Web crawling utilities.
    - `web_crawler.py`: Fetches and processes data from the web.
  - **history/**: Manages interaction history.
    - `embed_history.py`: Embeds and stores historical interactions.
  - `history_embedder.py`: Additional functionalities for embedding history.
  - `main.py`: Entry point for running the application.
  - **nlp/**: Natural Language Processing modules.
    - `sentiment_analysis.py`: Analyzes sentiment of user inputs.
    - `writing_style.py`: Analyzes and adjusts writing styles.
  - **rag/**: Retrieval-Augmented Generation modules.
    - `retriever.py`: Retrieves relevant passages based on queries.
- **utils/**: Utility scripts and helper functions.
  - `file_reader.py`: Reads and processes different file formats.
- **configs/**: Configuration files.
  - `personality.json`: Defines personality traits for the LLM.
- `requirements.txt`: Lists all Python dependencies required for the project.
- `README.md`: Documentation for the project.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. **Fork the Repository**

2. **Create a New Branch**
   ```bash
   git checkout -b feature/YourFeatureName
   ```

3. **Commit Your Changes**
   ```bash
   git commit -m "Add your message here"
   ```

4. **Push to the Branch**
   ```bash
   git push origin feature/YourFeatureName
   ```

5. **Open a Pull Request**

For significant changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Author

[PStarH](https://github.com/PStarH) – Your GitHub handle
