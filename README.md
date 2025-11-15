# LLM-TOOL-KIT

## Description

This project focuses on Natural Language Processing (NLP) for Arabic text. It utilizes transformer models and other NLP techniques for tasks such as data processing and semantic analysis. The project uses MongoDB for data storage and includes utilities for logging and potentially reward functions for reinforcement learning or custom training loops.

## Features

*   Semantic analysis
*   Training Embedding Model
*   Training Tokenizer
*   Finetuning Reasoning Model
*   Finetuning SFT Models
*   Data processing pipelines
*   Integration with MongoDB
*   Customizable logging
*   Potential for reinforcement learning or custom model training

## Installation

1.  **Clone the repository:**
    ```bash
    git clone 'https://github.com/ghyathmoussa/llm-tool-kit.git'
    cd `llm-tool-kit`
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up environment variables:**
    Create a `.env` file in the project root and add the following variables:
    ```env
    MONGO_URI=<your_mongodb_uri>
    MONGO_DB_NAME=<your_mongodb_database_name>
    MONGO_COLLECTION_NAME=<your_mongodb_collection_name>
    GROQ_API_KEY=<your_groq_api_key>  # Optional: only needed if using Groq API for synthetic data generation
    ```

## Usage

* Tokenizer Model

```
python3 tokenizer_model.py \
    --tokenizer-name "custom-arabic-tokenizer" \
    --vocab-size 32000 \
    --max-length 4096 \
    --model-path "/path/to/tokenizer.json" \
    --texts-source "data1.jsonl" \
    --min-frequency 2 \
    --batch-size 1000 \
    --max-samples 10000 \
    --special-tokens "[PAD]" "[UNK]" "[CLS]" "[SEP]" "[MASK]"
```

* Embeddings Model

```
python3 embeddings_model.py \
     --data_path data1.jsonl \
     --batch_size 16 \
    --learning_rate 0.001 \
    --epochs 5
```

* Finetune Model

Example of data structure

```
[
  {
    "problem": "Solve for x: 2x + 5 = 11",
    "solution": "x = 3"
  },
  {
    "problem": "If a train travels at 60 mph for 2 hours, and then at 40 mph for 1 hour, what is the total distance traveled?",
    "solution": "Total distance = 160 miles"
  }
]
```

Running Code:

```
python finetune_model.py \
  --ft-type reasoning \
  --use-quantization lora \
  --model-name Qwen/QwQ-32B-Preview \
  --output-dir ./fine-tuned-model \
  --dataset-name ./dataset.json \
  --batch-size 8 \
  --gradient-accumulation_steps 2 \
  --learning-rate 2e-4 \
  --num_train-epochs 3 \
  --max-steps -1 \
  --prompt "" \
  --max-length 4096 \
  --padding-side right \
  --beta 0.04 \
  --num-generations 4 \
  --max-completion-length 128
```

* Generating Synthetic Data

**Using vLLM Inference API (no API key required):**

```bash
python models/generate_synthetic_data.py \
  --input-file path/to/dataset.jsonl \
  --output-file data/synthetic_finetuning_data.jsonl \
  --qa-per-chunk 3 \
  --api-url http://localhost:8000/v1 \
  --llm-model your-model-name
```

**Using Groq API:**

```bash
python models/generate_synthetic_data.py \
  --input-file path/to/dataset.jsonl \
  --output-file data/synthetic_finetuning_data.jsonl \
  --qa-per-chunk 3 \
  --llm-api-key <your_groq_api_key> \
  --llm-model llama3-8b-8192
```

**Note:** When using `--api-url` for vLLM or other custom inference endpoints, no API key is required. The script will automatically detect the custom endpoint and use it instead of the default Groq API.

* Preprocess Data (Chunking)

```
python3 preprocess_data.py \
    --input-dir path/to/data/directory \
    --output-file path/to/output.jsonl \
    --max-tokens 2048 \
    --skip-header "Header text to skip"
```
## Configuration

The project configuration is managed in `config.py`. This file loads environment variables from a `.env` file located in the project root. Key configuration variables include:

*   `MONGO_URI`: MongoDB connection string.
*   `MONGO_DB_NAME`: Name of the MongoDB database.
*   `MONGO_COLLECTION_NAME`: Name of the MongoDB collection.
*   `GROQ_API_KEY`: (Optional) Groq API key for synthetic data generation. Not required if using vLLM or other custom inference endpoints.

Ensure these are correctly set in your `.env` file before running the project.

## Data

The project appears to use the [`Tashkeela-arabic-diacritized-text-utf8-0.3`](https://sourceforge.net/projects/tashkeela/files/Tashkeela-arabic-diacritized-text-utf8-0.3.zip/download) dataset, which can be found in the `Tashkeela-arabic-diacritized-text-utf8-0.3/` directory. Processed data or data used for modeling may be stored in json file.

The `source_data/` directory might contain raw or initial datasets, and the `data/` directory could be used for processed or intermediate data files.



## Docker

A `Dockerfile` is provided for containerizing the application. To build and run the Docker image:

1.  **Build the image:**
    ```bash
    docker build -t nlp-project .
    ```
2.  **Run the container:**
    ```bash
    docker run -d --env-file .env -p <host_port>:<container_port> nlp-project
    ```
    (You might need to adjust port mappings and ensure the container has access to necessary resources like MongoDB if it's not running within the same Docker network).

## Project Structure

```
.
├── Dockerfile
├── LICENSE
├── README.md
├── config.py
├── data/
├── evals/
├── helpers/
├── logs/
├── models/
├── notebooks/
│   ├── process_data.ipynb
│   └── semantic_process.ipynb
├── outputs/
├── requirements.txt
├── source_data/
├── Tashkeela-arabic-diacritized-text-utf8-0.3/
└── utils/
    ├── __init__.py
    ├── logger.py
    └── reward_functions.py
```

## Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

Please ensure your code adheres to the project's coding standards and includes tests where appropriate.

## License

This project is licensed under the terms of the LICENSE file.