# CLAUDE.md - CogBench Repository Guide

## Project Overview

CogBench is a cognitive psychology benchmark for evaluating Large Language Models (LLMs). It runs LLMs through 7 cognitive psychology experiments and computes both behavioral and performance metrics, enabling comparison against human baselines and random agents. Published at ICML 2024.

## Repository Structure

```
CogBench/
├── base_classes.py          # Core base classes: Experiment, LLM, StoringScores
├── utils.py                 # Text processing utilities
├── full_run.py              # Main orchestration script for full benchmark
├── requirements.txt         # Python dependencies (pip)
├── Experiments/             # 7 cognitive psychology experiments
│   ├── BART/                # Balloon Analogue Risk Task
│   ├── HorizonTask/         # Horizon Task
│   ├── InstrumentalLearning/# Instrumental Learning (slow fitting)
│   ├── ProbabilisticReasoning/
│   ├── RestlessBandit/      # Has V1 and V2 versions
│   ├── TemporalDiscounting/
│   └── TwoStepTask/         # Has V1 and V2 versions
├── llm_utils/               # LLM provider implementations
│   ├── llms.py              # Main dispatcher (get_llm function)
│   ├── gpt.py               # OpenAI GPT-3/GPT-4
│   ├── anthropic.py         # Anthropic Claude models
│   ├── google.py            # Google Vertex AI (text-bison)
│   └── hf.py                # Hugging Face transformers (local models)
└── Analysis/                # Analysis and visualization scripts
    ├── utils.py             # Core analysis utilities (merging, stats, plotting)
    ├── phenotype_comp.py    # Bar plots comparing LLM phenotypes
    ├── umap_plot.py         # UMAP dimensionality reduction
    ├── regression.py        # Behavioral score regression
    ├── perf_regression.py   # Performance score regression
    ├── cot_sb.py            # Chain of Thought vs Step Back comparison
    ├── radar_plot.py        # Radar plot analysis
    └── data/                # LLM feature data and aggregated scores
```

## Architecture

### Class Hierarchy

- **`Experiment`** (`base_classes.py`): Base class for all experiments. Handles CLI argument parsing, LLM initialization, experiment loop, and CSV result storage. Each experiment subclass implements `run_single_experiment()`.
- **`LLM`** (`base_classes.py`): Abstract base for LLM providers. Implements prompt engineering (Chain of Thought via `_cot` suffix, Step Back via `_sb` suffix) and robust postprocessing with multi-attempt answer extraction. Subclasses implement `_generate()`.
- **`StoringScores`** (`base_classes.py`): Base class for computing and storing behavioral/performance metrics from raw experiment CSV data. Each experiment subclass implements `get_scores()`.

### Data Flow

```
query.py (per experiment) → LLM interaction → data/{engine}.csv (raw responses)
store.py (per experiment) → Parse results  → scores_data.csv (behavioral + performance metrics)
Analysis scripts          → Aggregate data → plots and statistical summaries
```

### Each Experiment Directory Contains

- `query.py` - Runs the experiment against one or more LLMs
- `store.py` - Computes behavioral and performance scores from raw data
- `data/` - Raw CSV results per engine (`{engine}.csv`)
- `envs/` - Task environment implementations (where applicable)

## How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Full Benchmark for an LLM

```bash
python3 full_run.py --engine <LLM_NAME> [--compare_with gpt-4 claude-2] [--only_analysis]
```

### Run a Single Experiment

```bash
cd Experiments/<ExperimentName>
python3 query.py --engines <ENGINE_NAME> [--num_runs N] [--debug]
python3 store.py --engines <ENGINE_NAME>
```

### Key CLI Arguments

- `--engines`: LLM engine name(s) (e.g., `gpt-4`, `claude-2`, `random`, `interactive`)
- `--num_runs`: Number of experimental runs per engine
- `--version_number`: Experiment version (default `1`; some experiments have V2)
- `--debug`: Step-through debugging mode (prints LLM I/O and drops into ipdb)
- `--max-tokens`: Maximum tokens for LLM response (default 2)
- `--temp`: Temperature for LLM generation (default 0)

### Prompt Engineering Suffixes

Append to engine name to activate prompt engineering techniques:
- `_cot` - Chain of Thought (e.g., `gpt-4_cot`)
- `_sb` - Step Back prompting (e.g., `claude-2_sb`)

## Environment Variables

API keys are loaded from a `.env` file in the project root via `python-dotenv`:

```
OPENAI_API_KEY=<your-key>          # Required for GPT models
ANTHROPIC_API_KEY=<your-key>       # Required for Claude models
GOOGLE_CREDENTIALS_FILENAME2=<path> # Required for Google Vertex AI (JSON credentials file)
```

The `.env` file is gitignored. Never commit API keys or credentials.

## Supported LLM Engines

| Provider | Engine Names |
|----------|-------------|
| OpenAI | `gpt-4`, `text-davinci-003`, `text-davinci-002`, `text-curie-001`, `text-babbage-001`, `text-ada-001` |
| Anthropic | `claude-1`, `claude-2`, `claude-3-opus-20240229` |
| Google | `text-bison@002` |
| Hugging Face | `llama-2-7b`, `llama-2-13b`, `llama-2-70b` (+ chat variants), `hf-falcon`, `hf-mistral`, `hf-mixtral`, `hf-yi`, etc. |
| Baselines | `random` (random agent), `interactive` (human input) |

## Testing

There is no automated test suite. Validation is done through:
- `--debug` flag for step-by-step inspection of LLM I/O
- `--engines interactive` for manual human-in-the-loop testing
- `--engines random` for random baseline verification

## Code Conventions

- **Python 3** throughout; no type hints are used
- **CSV-based data storage** - all results written to CSV files, no database
- **Inheritance pattern** - experiments and LLM providers extend base classes
- **No build system** - scripts run directly with `python3`
- **No linter/formatter configuration** - no enforced style rules
- **No CI/CD** - manual execution via command line or HPC job scripts
- **Logging** - uses `print()` statements (no logging framework)
- **Error handling** - LLM API calls use retry with exponential backoff

## Important Notes for AI Assistants

1. **API rate limits**: Google Vertex AI is hard-limited to 5 queries/minute (12-second sleep between calls in `google.py`). GPT-4 has a 1-second delay between calls.
2. **InstrumentalLearning fitting is slow**: The score computation for this experiment takes significantly longer than others.
3. **Experiment versions**: RestlessBandit and some other experiments have V2 variants. Version is controlled via `--version_number` and affects data folder paths (`data/` vs `dataV2/`).
4. **Debug breakpoints**: `base_classes.py:112` has an `ipdb.set_trace()` that activates only when `--debug` is passed. Some files in `llm_utils/` may contain stray `ipdb` imports.
5. **Working directory matters**: `full_run.py` uses `os.chdir()` to navigate between experiment directories. Individual experiment scripts (`query.py`, `store.py`) expect to be run from within their experiment directory.
6. **No package installation**: The project is not an installable package. It uses relative imports (`from ..base_classes import ...`) which means scripts must be run from the repo root or the correct working directory.
7. **Data files are large**: The `data/` directories contain many CSV files (~520 total). Be mindful of this when searching or reading.
