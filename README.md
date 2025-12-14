# LoCoMo Agent - Long-Term Conversational Memory Evaluation

A minimal viable product (MVP) agent for running [LoCoMo](https://github.com/snap-research/locomo) evaluation - a benchmark for evaluating very long-term conversational memory of LLM agents.

## 🎯 Overview

This project implements a simple agent that can:
1. **Load LoCoMo dataset** - Automatically downloads and parses the benchmark data
2. **Answer questions** - Uses RAG or full-context approaches to answer questions about long conversations
3. **Evaluate performance** - Computes metrics (Exact Match, F1, BLEU, ROUGE-L) per question category

## 🚀 Quick Start

### 1. Install Dependencies (using uv)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv sync
```

### 2. Set up API Key

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=sk-your-api-key-here
```

### 3. Run Quick Test

```bash
source .venv/bin/activate
python scripts/quick_test.py
```

### 4. Run Full Evaluation

```bash
# RAG mode (recommended for long conversations)
python scripts/run_evaluation.py --mode rag --model gpt-4o-mini

# Full context mode (truncates long conversations)
python scripts/run_evaluation.py --mode full_context --model gpt-4o-mini

# Test with fewer conversations
python scripts/run_evaluation.py --max-conversations 2
```

## 📁 Project Structure

```
long-mem-agent/
├── src/locomo_agent/
│   ├── __init__.py         # Package init
│   ├── data_models.py      # Pydantic data models
│   ├── data_loader.py      # Dataset loading & parsing
│   ├── metrics.py          # Evaluation metrics (EM, F1, BLEU, ROUGE)
│   ├── retriever.py        # Simple RAG retriever with FAISS
│   ├── agent.py            # Main agent implementation
│   ├── evaluator.py        # Evaluation orchestrator
│   └── cli.py              # Command-line interface
├── scripts/
│   ├── run_evaluation.py   # Main evaluation script
│   └── quick_test.py       # Quick test script
├── data/                   # Cached dataset (auto-downloaded)
├── results/                # Evaluation results
├── pyproject.toml          # Project configuration (uv compatible)
├── uv.lock                 # Lock file for reproducible installs
└── README.md
```

## 🔧 Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | LLM model for generation | `gpt-4o-mini` |
| `--mode` | `rag` or `full_context` | `rag` |
| `--top-k` | Number of chunks to retrieve (RAG) | `5` |
| `--use-observations` | Include observations in RAG index | `False` |
| `--use-summaries` | Include session summaries in RAG index | `False` |
| `--max-conversations` | Limit conversations for testing | `None` |
| `--output` | Path to save results | `./results/evaluation_results.json` |

## 📊 Evaluation Metrics

- **Exact Match (EM)**: Binary score for exact string match
- **F1 Score**: Token-level F1 between prediction and ground truth
- **BLEU-1**: Unigram BLEU score
- **ROUGE-L**: Longest common subsequence F-measure

Results are computed overall and per question category.

## 🏗️ Architecture

### RAG Mode (Recommended)
```
Question → Retriever → Top-K Chunks → LLM → Answer
              ↑
    Conversation indexed with FAISS + sentence-transformers
```

### Full Context Mode
```
Question + Truncated Conversation → LLM → Answer
```

## 📈 Expected Results

Based on the LoCoMo paper, typical results on gpt-3.5-turbo with RAG:
- F1: ~0.30-0.40
- BLEU-1: ~0.25-0.35

Better models (GPT-4, Claude) typically achieve higher scores.

## 🛠️ Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run linting
ruff check src/

# Run type checking
mypy src/

# Run tests
pytest
```

## 🔗 References

- [LoCoMo Paper](https://arxiv.org/abs/2402.17753) - "Evaluating Very Long-Term Conversational Memory of LLM Agents"
- [LoCoMo GitHub](https://github.com/snap-research/locomo)

## 📝 Citation

If you use this code, please cite the original LoCoMo paper:

```bibtex
@article{maharana2024evaluating,
  title={Evaluating very long-term conversational memory of llm agents},
  author={Maharana, Adyasha and Lee, Dong-Ho and Tulyakov, Sergey and Bansal, Mohit and Barbieri, Francesco and Fang, Yuwei},
  journal={arXiv preprint arXiv:2402.17753},
  year={2024}
}
```

## 📄 License

MIT License
