# LoCoMo Agent - Long-Term Conversational Memory Evaluation

A minimal viable product (MVP) agent for running [LoCoMo](https://github.com/snap-research/locomo) evaluation - a benchmark for evaluating very long-term conversational memory of LLM agents.

## 🎯 Overview

This project implements a flexible agent that can:
1. **Load LoCoMo dataset** - Automatically downloads and parses the benchmark data
2. **Answer questions** - Uses RAG or full-context approaches to answer questions about long conversations
3. **Evaluate performance** - Computes metrics (Exact Match, F1, BLEU, ROUGE-L) per question category
4. **Compare models** - Supports multiple LLM providers for easy benchmarking

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

### 2. Set up API Keys

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key  # Optional
```

### 3. Run Quick Test

```bash
source .venv/bin/activate
python scripts/quick_test.py
```

### 4. Run Experiments

```bash
# Run with YAML config
python scripts/run_experiment.py --config configs/gpt4o-mini.yaml

# Run with model preset
python scripts/run_experiment.py --model gpt-4o-mini

# Run with local Ollama model
python scripts/run_experiment.py --model ollama:llama3.2

# Compare multiple models
python scripts/run_experiment.py --compare gpt-4o-mini claude-3-5-sonnet --max-conversations 2
```

## 🔧 Multi-Model Support

### Supported Providers

| Provider | Models | Setup |
|----------|--------|-------|
| **OpenAI** | gpt-4o, gpt-4o-mini, gpt-3.5-turbo | Set `OPENAI_API_KEY` |
| **Anthropic** | claude-3-5-sonnet, claude-3-opus | Set `ANTHROPIC_API_KEY` |
| **Ollama** | llama3.2, mistral, qwen2.5, phi3 | `ollama pull <model>` |
| **vLLM** | Any HuggingFace model | Start vLLM server |
| **OpenAI-compatible** | Together, Anyscale, etc. | Set base_url |

### Using Local Models (Ollama)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3.2

# Run experiment
python scripts/run_experiment.py --model ollama:llama3.2
```

### Using Finetuned Models

See `configs/finetuned-model.yaml` for examples of:
- OpenAI finetuned models (`ft:gpt-xxx`)
- Ollama custom models
- vLLM local inference
- Any OpenAI-compatible API

## 📁 Project Structure

```
long-mem-agent/
├── src/locomo_agent/
│   ├── config.py           # Unified configuration system
│   ├── llm_client.py       # Multi-provider LLM client
│   ├── agent_v2.py         # Enhanced agent with multi-provider support
│   ├── agent.py            # Original agent (OpenAI only)
│   ├── data_models.py      # Pydantic data models
│   ├── data_loader.py      # Dataset loading & parsing
│   ├── metrics.py          # Evaluation metrics
│   ├── retriever.py        # RAG retriever with FAISS
│   ├── evaluator.py        # Evaluation orchestrator
│   └── cli.py              # Command-line interface
├── configs/                # YAML experiment configs
│   ├── gpt4o-mini.yaml
│   ├── claude-sonnet.yaml
│   ├── ollama-llama.yaml
│   └── finetuned-model.yaml
├── scripts/
│   ├── run_experiment.py   # Multi-model experiment runner
│   ├── run_evaluation.py   # Simple evaluation script
│   └── quick_test.py       # Quick test script
├── data/                   # Cached dataset
├── results/                # Evaluation results
└── pyproject.toml          # Project configuration
```

## 📊 Configuration Options

### YAML Config Example

```yaml
model:
  name: gpt-4o-mini
  provider: openai
  temperature: 0.0
  max_tokens: 256

use_rag: true

retriever:
  model_name: all-MiniLM-L6-v2
  top_k: 5
  use_observations: false

evaluation:
  experiment_name: my-experiment
  tags: [baseline, test]
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config` | Path to YAML config file | - |
| `--model` | Model name (preset or provider:name) | `gpt-4o-mini` |
| `--compare` | Compare multiple models | - |
| `--max-conversations` | Limit conversations for testing | `None` |

## 📈 Evaluation Metrics

- **Exact Match (EM)**: Binary score for exact string match
- **F1 Score**: Token-level F1 between prediction and ground truth
- **BLEU-1**: Unigram BLEU score
- **ROUGE-L**: Longest common subsequence F-measure

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

## 🛠️ Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run linting
ruff check src/

# Run type checking
mypy src/
```

## 🔗 References

- [LoCoMo Paper](https://arxiv.org/abs/2402.17753) - "Evaluating Very Long-Term Conversational Memory of LLM Agents"
- [LoCoMo GitHub](https://github.com/snap-research/locomo)

## 📝 Citation

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
