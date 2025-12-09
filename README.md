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

The EnigmaEval dataset is private and not included in this repository. 

```bash
# Copy the dataset pickle file to the data/ directory
cp /path/to/enigmaeval.pkl data/enigmaeval.pkl
```

### 4. Run Evaluation

```bash
python enigmaeval_eval.py \
  --model gpt-5-mini \
  --split all \
  --output_dir results/enigmaeval/ \
  --models_config configs/models.yaml \
  --max_concurrent 128
```

## 📊 Usage

### Command Line Arguments

```bash
python enigmaeval_eval.py \
  --model MODEL_NAME \               # Model alias from configs/models.yaml
  --split SPLIT_NAME \               # Dataset split (see below)
  --output_file OUTPUT_PATH \        # Path for results JSON
  --models_config CONFIG_PATH \      # Path to models config (default: configs/models.yaml)
  --max_concurrent N \               # Max concurrent requests (default: 4)
```

### Model Configuration

Models are configured in `configs/models.yaml`.

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
## 🙏 Citation

If you use EnigmaEval in your research, please cite:

```bibtex
@misc{wang2025enigmaevalbenchmarklongmultimodal,
      title={EnigmaEval: A Benchmark of Long Multimodal Reasoning Challenges}, 
      author={Clinton J. Wang and Dean Lee and Cristina Menghini and Johannes Mols and Jack Doughty and Adam Khoja and Jayson Lynch and Sean Hendryx and Summer Yue and Dan Hendrycks},
      year={2025},
      eprint={2502.08859},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2502.08859}, 
}
```

## 🔗 Related

- Main CAIS simple-evals repository: [simple-evals](https://github.com/centerforaisafety/simple-evals)
