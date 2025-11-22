# ✨ LLMium

> **The Complete Toolkit for Intelligent Language Model Development & Deployment**

Unlock the full potential of language models with LLMium—a modern, comprehensive toolkit designed for researchers, engineers, and data scientists. Whether you're fine-tuning cutting-edge models, generating synthetic training data, or evaluating quality with sophisticated reward systems, LLMium makes it effortless.

## 🎯 What is LLMium?

LLMium is a production-ready framework for building, training, and deploying language models with support for **Arabic & multilingual NLP**. It combines state-of-the-art transformer techniques, intelligent data processing, and quality evaluation into one seamless experience.

## 🚀 Features

**Core Capabilities:**
- 🧠 **Smart Model Fine-tuning** — Reasoning models, SFT, and custom training loops
- 📊 **Semantic Analysis** — Advanced NLP for Arabic and multilingual text
- 🎯 **Synthetic Data Generation** — Auto-generate QA pairs powered by LLMs (vLLM, Groq, OpenAI)
- ⭐ **Quality Assurance** — Multi-language reward models (Arabic, English, Multilingual)
- 🔤 **Tokenizer Training** — Build custom tokenizers for specialized vocabularies
- 📈 **Embedding Models** — Train and fine-tune semantic embeddings
- 🗄️ **MongoDB Integration** — Seamless data persistence and retrieval
- ⚡ **Parallel Processing** — Fast, concurrent API requests for scale
- 🐳 **Docker Ready** — One-command deployment with containerization
- 📝 **Comprehensive Logging** — Track everything with structured logging

## 📦 Installation

### Quick Start (3 steps)

**1. Clone the repository:**
```bash
git clone https://github.com/ghyathmoussa/llm-tool-kit.git
cd llm-tool-kit
```

**2. Set up your environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Configure environment variables:**

Create a `.env` file in the project root:
```env
# MongoDB (required for data persistence)
MONGO_URI=mongodb+srv://user:password@cluster.mongodb.net
MONGO_DB_NAME=llmium_db
MONGO_COLLECTION_NAME=training_data

# API Keys (optional, depending on features used)
GROQ_API_KEY=your_groq_api_key              # For synthetic data generation
HF_API_TOKEN=your_huggingface_token         # For reward model evaluation

# Performance tuning (optional)
MAX_PARALLEL_REQUESTS=5                      # Parallel API requests (default: 1)
```

✅ **You're all set!** Proceed to [Usage](#usage) to get started.

## 💡 Usage

### 1️⃣ Train a Custom Tokenizer

Build a specialized tokenizer optimized for your domain or language:

```bash
python3 models/tokenizer_model.py \
    --tokenizer-name "custom-arabic-tokenizer" \
    --vocab-size 32000 \
    --max-length 4096 \
    --model-path "./tokenizer.json" \
    --texts-source "data1.jsonl" \
    --min-frequency 2 \
    --batch-size 1000 \
    --max-samples 10000 \
    --special-tokens "[PAD]" "[UNK]" "[CLS]" "[SEP]" "[MASK]"
```

---

### 2️⃣ Train Embedding Models

Create semantic embeddings from your text data:

```bash
python3 models/embeddings_model.py \
    --data_path data1.jsonl \
    --batch_size 16 \
    --learning_rate 0.001 \
    --epochs 5
```

---

### 3️⃣ Fine-tune Reasoning & SFT Models

Adapt pre-trained models for your specific tasks. Start with your dataset:

**Dataset Format:**
```json
[
  {
    "problem": "Solve for x: 2x + 5 = 11",
    "solution": "x = 3"
  },
  {
    "problem": "If a train travels at 60 mph for 2 hours, then at 40 mph for 1 hour, what is the total distance?",
    "solution": "Total distance = 160 miles"
  }
]
```

**Fine-tune with LoRA optimization:**
```bash
python models/finetune_model.py \
  --ft-type reasoning \
  --use-quantization lora \
  --model-name Qwen/QwQ-32B-Preview \
  --output-dir ./fine-tuned-model \
  --dataset-name ./dataset.json \
  --batch-size 8 \
  --gradient-accumulation_steps 2 \
  --learning-rate 2e-4 \
  --num_train-epochs 3 \
  --max-length 4096 \
  --padding-side right \
  --beta 0.04 \
  --num-generations 4 \
  --max-completion-length 128
```

---

### 4️⃣ Generate Synthetic Training Data

Auto-generate QA pairs from raw text—choose your inference backend:

**Option A: Local vLLM (No API key needed):**
```bash
python models/generate_synthetic_data.py \
  --input-file path/to/dataset.jsonl \
  --output-file data/synthetic_finetuning_data.jsonl \
  --qa-per-chunk 3 \
  --api-url http://localhost:8000/v1 \
  --llm-model your-model-name
```

**Option B: Groq API (Fast & Cost-effective):**
```bash
python models/generate_synthetic_data.py \
  --input-file path/to/dataset.jsonl \
  --output-file data/synthetic_finetuning_data.jsonl \
  --qa-per-chunk 3 \
  --llm-api-key $GROQ_API_KEY \
  --llm-model llama3-8b-8192
```

**Option C: Parallel Processing (Speed boost):**
```bash
python models/generate_synthetic_data.py \
  --input-file path/to/dataset.jsonl \
  --output-file data/synthetic_finetuning_data.jsonl \
  --qa-per-chunk 3 \
  --api-url http://localhost:8000/v1 \
  --llm-model your-model-name \
  --max-parallel-requests 5
```

---

### 5️⃣ Quality Evaluation with Reward Models

Filter and score synthetic data using intelligent multi-language reward models:

**Quick Usage (Arabic data with quality threshold):**
```bash
python models/generate_synthetic_data.py \
  --input-file data/arabic_texts.jsonl \
  --output-file data/filtered.jsonl \
  --qa-per-chunk 3 \
  --use-reward-model \
  --reward-language arabic \
  --reward-threshold 0.6
```

**Using Local vLLM (Faster & Cheaper):**
```bash
python models/generate_synthetic_data.py \
  --input-file data/input.jsonl \
  --output-file data/output.jsonl \
  --use-reward-model \
  --reward-api-type vllm \
  --reward-api-endpoint http://localhost:8001/v1
```

**Programmatic Python API:**
```python
from models.reward_model import create_reward_evaluator

evaluator = create_reward_evaluator(
    language="arabic",
    reward_threshold=0.6,
    api_type="huggingface"
)

qa_pairs = [
    {"question": "سؤال؟", "answer": "إجابة تفصيلية"},
]
filtered_pairs, scores = evaluator.evaluate_qa_pairs(qa_pairs)
```

**Output with quality scores:**
```json
{
  "instruction": "السؤال؟",
  "input": "",
  "output": "الإجابة",
  "reward_score": 0.85,
  "source_document_info": {...}
}
```

**Reward Model Options:**

| Argument | Values | Purpose |
|----------|--------|---------|
| `--reward-language` | `multilingual`, `arabic`, `english` | Evaluate in target language |
| `--reward-threshold` | `0.0–1.0` (default: 0.5) | Quality cutoff score |
| `--reward-api-type` | `huggingface`, `openai`, `vllm`, `local` | Evaluation backend |
| `--reward-api-endpoint` | URL | Custom endpoint (for vLLM/local) |

---

### 6️⃣ Preprocess & Chunk Text

Split large documents into tokenizable chunks:

```bash
python3 models/preprocess_data.py \
    --input-dir path/to/data/directory \
    --output-file path/to/output.jsonl \
    --max-tokens 2048 \
    --skip-header "Header text to skip"
```
---

## ⚙️ Configuration & Environment

All settings are managed in `config.py` and loaded from `.env`. Here's what you can customize:

| Variable | Purpose | Required | Example |
|----------|---------|----------|---------|
| `MONGO_URI` | MongoDB connection | ✅ Yes | `mongodb+srv://...` |
| `MONGO_DB_NAME` | Database name | ✅ Yes | `llmium_db` |
| `MONGO_COLLECTION_NAME` | Collection name | ✅ Yes | `training_data` |
| `GROQ_API_KEY` | For LLM synthetic data | ❌ Optional | `gsk_...` |
| `HF_API_TOKEN` | For reward evaluation | ❌ Optional | `hf_...` |
| `MAX_PARALLEL_REQUESTS` | Concurrent API calls | ❌ Optional | `5` (default: 1) |

---

## 📂 Project Structure

```
llmium/
├── README.md                         # You are here ✨
├── requirements.txt                  # Python dependencies
├── config.py                         # Configuration loader
├── Dockerfile                        # Container deployment
│
├── models/                           # Core ML modules
│   ├── finetune_model.py            # Model fine-tuning pipeline
│   ├── generate_synthetic_data.py    # QA pair generation
│   ├── reward_model.py              # Quality evaluation
│   ├── embeddings_model.py          # Semantic embedding training
│   ├── tokenizer_model.py           # Custom tokenizer creation
│   └── preprocess_data.py           # Text chunking & preprocessing
│
├── notebooks/                        # Jupyter notebooks for exploration
│   ├── model.ipynb
│   ├── process_data.ipynb
│   ├── rl_model.ipynb
│   └── semantic_process.ipynb
│
├── data/                            # Processed & intermediate data
│   ├── data1.jsonl
│   ├── processed_data.jsonl
│   └── english/
│
├── source_data/                     # Raw datasets (Arabic Islamic texts)
│   ├── أحكام القرآن للجصاص.txt
│   ├── البحر المحيط.txt
│   └── ... 50+ Islamic legal texts
│
├── helpers/                         # Utility functions
│   ├── get_prompt.py
│   └── json2mongo.py
│
├── evals/                           # Evaluation scripts
│   └── reasoning_eval.py
│
└── utils/                           # Core utilities
    ├── logger.py                    # Structured logging
    └── reward_functions.py          # Evaluation metrics
```



## 🐳 Docker Deployment

Deploy LLMium in seconds with Docker:

**Build the container:**
```bash
docker build -t llmium:latest .
```

**Run with environment variables:**
```bash
docker run -d \
  --name llmium \
  --env-file .env \
  -p 8000:8000 \
  llmium:latest
```

**Using Docker Compose (recommended):**
```bash
docker-compose up -d
```

> 📌 **Note:** Ensure MongoDB is accessible from the container (same network or remote connection).

---

## 🤝 Contributing

We welcome contributions from researchers, engineers, and enthusiasts!

**How to contribute:**

1. 🍴 Fork the repository
2. 🌿 Create a feature branch: `git checkout -b feature/amazing-feature`
3. ✏️ Make your changes and commit: `git commit -m 'Add amazing feature'`
4. 📤 Push to your fork: `git push origin feature/amazing-feature`
5. 🔄 Open a Pull Request

**Guidelines:**
- Write clean, well-documented code
- Include tests for new functionality
- Follow Python PEP 8 style guide
- Add docstrings to functions and classes
- Update README for new features

---

## 📄 License

This project is licensed under the terms specified in the `LICENSE` file.

---

## 🙌 Acknowledgments

- **Base Dataset:** [Tashkeela Arabic Diacritized Corpus](https://sourceforge.net/projects/tashkeela/)
- **Models:** Powered by transformers, HuggingFace, and Groq APIs

---

## 📚 Resources

- 📖 [HuggingFace Documentation](https://huggingface.co/docs)
- 🚀 [Groq API Guide](https://console.groq.com/docs)
- 🔧 [vLLM Setup](https://docs.vllm.ai/)

---

## 💬 Support

Need help? 

- 📋 Check [Discussions](https://github.com/ghyathmoussa/llm-tool-kit/discussions) for Q&A
- 🐛 Report bugs via [Issues](https://github.com/ghyathmoussa/llm-tool-kit/issues)
- 📧 Contact the maintainers

---

**Made with ❤️ by Ghyath Moussa**