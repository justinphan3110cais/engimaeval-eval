# EnigmaEval

Evaluation harness for EnigmaEval - a benchmark for testing AI systems on puzzle-solving tasks.

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/justinphan3110cais/engimaeval-eval.git
cd engimaeval-eval
pip install -r requirements.txt
```

### 2. Setup Environment Variables

```bash
# Copy the environment template
cp env.example .env

# Edit .env and add your API keys
```

### 3. Setup Dataset

**Important:** The EnigmaEval dataset is private and not included in this repository. 

```bash
# Copy the dataset pickle file to the data/ directory
cp /path/to/enigmaeval.pkl data/enigmaeval.pkl
```

The pickle file should contain the EnigmaEval test set exported from the `cais/enigmaeval` HuggingFace dataset.

### 4. Run Evaluation

```bash
python enigmaeval_eval.py \
  --model gpt-4o \
  --split all \
  --output_file results/enigmaeval_gpt4o.json \
  --models_config configs/models.yaml \
  --max_concurrent 4
```

## 📊 Usage

### Command Line Arguments

```bash
python enigmaeval_eval.py \
  --model MODEL_NAME \              # Model alias from configs/models.yaml
  --split SPLIT_NAME \               # Dataset split (see below)
  --output_file OUTPUT_PATH \        # Path for results JSON
  --models_config CONFIG_PATH \      # Path to models config (default: configs/models.yaml)
  --max_concurrent N \               # Max concurrent requests (default: 4)
  --text_only \                      # Skip puzzles with images (optional)
  --exclude_meta                     # Exclude meta puzzles (optional)
```

### Available Splits

- `all`: All puzzle sources
- `normal`: PuzzledPint, Cryptic Crossword, Puzzle Potluck, CRUMS, CS50x
- `hard`: MIT Mystery Hunt, Labor Day Extravaganza, GMPuzzles
- `pdf`: CRUMS, Cryptic Crossword, PuzzledPint, Labor Day Extravaganza, CS50x, GMPuzzles
- `html`: MIT Mystery Hunt, Puzzle Potluck
- Individual sources: `pp`, `lde`, `mit`, `crums`, `cc`, `potluck`, `cs50`, `gm`

### Model Configuration

Models are configured in `configs/models.yaml`. Example:

```yaml
gpt-4o:
  model: openai/gpt-4o
  generation_config:
    max_tokens: 16000

claude-sonnet-4:
  model: anthropic/claude-sonnet-4
  generation_config:
    max_tokens: 8000
    
deepseek-v3.2:
  model: openai/deepseek-reasoner
  generation_config:
    api_key_env: DEEPSEEK_API_KEY
    api_base_url: https://api.deepseek.com
```

## 📁 Repository Structure

```
engimaeval-eval/
├── enigmaeval_eval.py           # Main evaluation script
├── enigmaeval_utils.py          # Utility functions and data loading
├── configs/
│   └── models.yaml              # Model configurations
├── shared/
│   ├── __init__.py
│   └── llm_agents.py            # LLM agent interface
├── prompt_templates/            # Prompt templates for different puzzle types
│   ├── standard_tips.txt
│   ├── imageless_tips.txt
│   ├── mit_tips.txt
│   └── plagiarism.txt
├── data/                        # Dataset storage (gitignored)
│   └── enigmaeval.pkl           # Dataset file (not in repo - must be copied)
├── results/                     # Evaluation results (gitignored)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🔒 Privacy Note

The EnigmaEval dataset (`cais/enigmaeval`) is **private** and requires special access. The dataset is NOT included in this repository and must be obtained separately and copied to `data/enigmaeval.pkl`.


## 🙏 Citation

If you use EnigmaEval in your research, please cite:

```bibtex
@article{enigmaeval2024,
  title={EnigmaEval: A Benchmark for Puzzle-Solving Intelligence},
  author={...},
  journal={...},
  year={2024}
}
```

## 🔗 Related

- Main CAIS simple-evals repository: [simple-evals](https://github.com/centerforaisafety/simple-evals)
