# CRSLab: Complete Project Guide
> Cross-Platform Setup (macOS M2 · Linux · Windows) + Full Architecture Deep-Dive

---

## Table of Contents

1. [What is CRSLab?](#1-what-is-crslab)
2. [Environment Setup](#2-environment-setup)
   - [macOS (Apple Silicon M2)](#21-macos-apple-silicon-m2)
   - [Linux (Ubuntu / Debian)](#22-linux-ubuntu--debian)
   - [Windows](#23-windows)
3. [Installing Dependencies](#3-installing-dependencies)
4. [Project Structure](#4-project-structure)
5. [Component Deep-Dive](#5-component-deep-dive)
   - [Config Module](#51-config-module)
   - [Data Module](#52-data-module)
   - [Model Module](#53-model-module)
   - [System Module](#54-system-module)
   - [Evaluator Module](#55-evaluator-module)
   - [Quick Start Module](#56-quick-start-module)
6. [Data Flow: End-to-End Walkthrough](#6-data-flow-end-to-end-walkthrough)
7. [Supported Datasets](#7-supported-datasets)
8. [Supported Models](#8-supported-models)
9. [Configuration Files Explained](#9-configuration-files-explained)
10. [Running the Project](#10-running-the-project)
11. [Evaluation Metrics](#11-evaluation-metrics)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. What is CRSLab?

**CRSLab** (Conversational Recommender System Lab) is an open-source research toolkit from the **AI Box group at Renmin University of China (RUC)**. It is a unified framework for building, training, and evaluating **Conversational Recommender Systems (CRS)**.

### The Core Problem

Traditional recommenders (Netflix, Spotify) are static — they push items based on past behavior. A Conversational Recommender System engages the user in a **natural language dialogue** to elicit preferences and refine recommendations in real time. Think of it as a shopping assistant you can chat with.

### The Three Sub-Tasks CRSLab Unifies

```
User: "I want something like Inception but lighter"
        │
        ▼
┌──────────────────────────────────────────────┐
│  POLICY (what to do next?)                   │
│  → Decide: ask a clarifying question or      │
│            recommend directly?               │
└────────────────┬─────────────────────────────┘
                 │
    ┌────────────┴────────────┐
    ▼                         ▼
┌──────────────┐    ┌──────────────────────────┐
│ RECOMMENDATION│    │    CONVERSATION           │
│ (what items?) │    │ (what words to say?)      │
│ → The Dark    │    │ → "You might enjoy        │
│   Knight,     │    │    The Dark Knight —      │
│   Interstellar│    │    it's intense but       │
└──────────────┘    │    accessible!"            │
                    └──────────────────────────┘
```

| Sub-Task | Question Answered | Output |
|----------|------------------|--------|
| **Recommendation** | What items to suggest? | Ranked list of items (movies, books, etc.) |
| **Conversation** | What natural language response to generate? | A sentence or paragraph |
| **Policy** | What action/topic to pursue next? | An action label or topic keyword |

---

## 2. Environment Setup

The project targets Python 3.7.17 (see `.python-version`), but several pinned packages lack native binary wheels for that version on all platforms. **The recommended approach is Python 3.9 with Miniforge/Miniconda** — it avoids binary compatibility issues on all operating systems.

---

### 2.1 macOS (Apple Silicon M2)

#### Step 1 — Install Miniforge (arm64 Conda)

Miniforge is the recommended Python distribution for Apple Silicon — it uses conda-forge packages compiled natively for arm64.

```bash
# Download the arm64 installer
curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh -o miniforge.sh

# Run installer (follow prompts, say yes to conda init)
bash miniforge.sh

# Reload your shell
source ~/.zshrc
```

#### Step 2 — Create a Dedicated Conda Environment

```bash
conda create -n crslab python=3.9 -y
conda activate crslab
```

> **Why 3.9 and not 3.7?**
> Python 3.7 does not have native arm64 binary wheels for packages like numpy, fasttext, and sentencepiece on PyPI. Python 3.9 has full arm64 support and is backward-compatible with this codebase (which only requires Python >= 3.6).

#### Step 3 — Install Xcode Command Line Tools (required for C extensions)

```bash
xcode-select --install
```

---

### 2.2 Linux (Ubuntu / Debian)

Tested on Ubuntu 20.04 and 22.04. Other Debian-based distributions should work the same way.

#### Step 1 — Install System Build Tools

```bash
sudo apt update
sudo apt install -y build-essential git cmake curl wget pkg-config \
    libsentencepiece-dev
```

#### Step 2 — Install Miniforge (x86_64 or aarch64)

```bash
# Detect architecture automatically and download the right installer
ARCH=$(uname -m)
curl -L "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${ARCH}.sh" -o miniforge.sh

# Run the installer
bash miniforge.sh -b -p "$HOME/miniforge3"

# Initialise conda in your shell
"$HOME/miniforge3/bin/conda" init bash
source ~/.bashrc
```

> If you use zsh, replace `bash` with `zsh` in the `conda init` call and source `~/.zshrc`.

#### Step 3 — Create and Activate the Environment

```bash
conda create -n crslab python=3.9 -y
conda activate crslab
```

> **NVIDIA GPU users:** if you have a CUDA-capable GPU, install the matching CUDA toolkit via `sudo apt install nvidia-cuda-toolkit` (or use the official NVIDIA runfile installer) before installing PyTorch in section 3. Section 3 covers how to select a CUDA-enabled PyTorch wheel.

---

### 2.3 Windows

Tested on Windows 10 / 11 (64-bit). Use **PowerShell** or **Git Bash** for all commands below.

#### Step 1 — Install Miniforge

1. Download the Windows installer from the [Miniforge releases page](https://github.com/conda-forge/miniforge/releases/latest) — pick `Miniforge3-Windows-x86_64.exe`.
2. Run the installer. When prompted:
   - Choose **"Just Me"** (avoids admin requirements).
   - Check **"Add Miniforge3 to my PATH environment variable"** (or use the Miniforge Prompt shortcut).

Open a new **Miniforge Prompt** (Start menu) after installation.

#### Step 2 — Install Microsoft C++ Build Tools (required for C extensions)

Several packages (fasttext, sentencepiece, pkuseg) compile C/C++ code during installation.

1. Download [Visual Studio Build Tools 2022](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
2. In the installer, select the **"Desktop development with C++"** workload.
3. Restart your terminal after installation.

#### Step 3 — Create and Activate the Environment

```powershell
conda create -n crslab python=3.9 -y
conda activate crslab
```

> If `conda activate` is not recognised in PowerShell, run:
> ```powershell
> conda init powershell
> ```
> Then restart PowerShell and try again.

---

## 3. Installing Dependencies

### Step 1 — Install PyTorch

The correct install command depends on your platform and whether you have a GPU.

**macOS (Apple Silicon M2 — MPS backend)**

```bash
pip install torch torchvision torchaudio
```

Verify MPS is available:

```python
import torch
print(torch.backends.mps.is_available())  # Should print: True
```

> **Note on GPU in CRSLab:** CRSLab uses the `--gpu` flag to select a CUDA device. On M2, pass `--gpu -1` (CPU mode) — MPS is not directly mapped by the framework. CPU mode is sufficient for experimentation.

**Linux — CPU only**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Linux — NVIDIA GPU (CUDA 12.1 example)**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Replace `cu121` with `cu118` for CUDA 11.8, etc. Check [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the exact command matching your CUDA version.

Verify CUDA is available:

```python
import torch
print(torch.cuda.is_available())   # Should print: True
print(torch.cuda.get_device_name(0))
```

**Windows — CPU only**

```powershell
pip install torch torchvision torchaudio
```

**Windows — NVIDIA GPU (CUDA 12.1 example)**

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 2 — Install PyTorch Geometric (for Graph Neural Network models)

PyTorch Geometric is used by KGSF and KBRD models for relational graph convolution. It requires its own set of extension packages.

```bash
# First get your torch version
python -c "import torch; print(torch.__version__)"
# Example output: 2.1.0

# Install PyG dependencies (replace torch-2.1.0 with your version)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)")+cpu.html

# Install torch-geometric
pip install torch-geometric
```

> If the wheel URL above fails, visit https://pytorch-geometric.readthedocs.io for the exact command for your PyTorch version.

### Step 3 — Install the Remaining Dependencies

The original `requirements.txt` uses pinned old versions that conflict with M2. Install a relaxed version:

```bash
# Install numpy first (arm64 native)
pip install "numpy>=1.21"

# Install transformers (relaxed from 4.1.1 — newer versions are compatible)
pip install "transformers>=4.1.1,<5.0"

# Install sentencepiece (relaxed constraint — new versions work fine)
pip install sentencepiece

# Install fasttext (may need compilation — ensure cmake is available)
pip install fasttext

# If fasttext fails on M2, try the community fork:
# pip install fasttext-wheel

# Install remaining packages
pip install \
    pkuseg \
    "pyyaml>=5.4" \
    "tqdm>=4.55" \
    "loguru>=0.5.3" \
    "nltk>=3.4.4" \
    "requests>=2.25" \
    "scikit-learn>=0.24" \
    fuzzywuzzy \
    python-Levenshtein \
    "tensorboard>=2.4"
```

### Step 4 — Install NLTK Data (needed for tokenization)

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Step 5 — Install CRSLab itself

From the project root directory:

```bash
# Navigate to wherever you cloned the repo, e.g.:
#   macOS/Linux: cd ~/Desktop/CRSLab
#   Windows:     cd C:\Users\<you>\Desktop\CRSLab

# Install in editable/development mode (so you can modify source)
pip install -e . --no-deps --no-build-isolation
```

> `--no-deps` skips re-installing the pinned dependencies from setup.py since you already installed compatible versions above.
>
> `--no-build-isolation` is required because `setup.py` imports `torch` and `torch_geometric` at the top level to validate they are present. Newer pip (25+) runs the build in an isolated subprocess where conda-env packages are not visible, causing a `ModuleNotFoundError: No module named 'torch'` even when torch is correctly installed. This flag disables that isolation so the build uses your active environment.

### Verify Installation

```bash
python -c "
import torch
import torch_geometric
import transformers
import nltk
import crslab
print('All imports successful!')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
"
```

---

## 4. Project Structure

```
CRSLab/
│
├── run_crslab.py              ← Entry point (CLI script)
├── requirements.txt           ← Pinned dependencies
├── setup.py                   ← Package install config
├── .python-version            ← Specifies Python 3.7.17
│
├── config/                    ← YAML config files (one per model+dataset combo)
│   ├── crs/                   ← Full CRS model configs
│   │   ├── kgsf/redial.yaml
│   │   ├── tgredial/tgredial.yaml
│   │   └── ...
│   ├── conversation/          ← Standalone conversation model configs
│   ├── policy/                ← Standalone policy model configs
│   └── recommendation/        ← Standalone recommendation model configs
│
└── crslab/                    ← Main Python package
    ├── config/                ← Config loader
    ├── data/                  ← Dataset + DataLoader implementations
    │   ├── dataset/           ← One folder per dataset (redial, tgredial, etc.)
    │   └── dataloader/        ← One file per model's data format
    ├── model/                 ← Neural network models
    │   ├── crs/               ← Full CRS models (rec + conv + policy)
    │   ├── recommendation/    ← Standalone rec models
    │   ├── conversation/      ← Standalone conv models
    │   ├── policy/            ← Standalone policy models
    │   └── utils/             ← Shared layers (attention, transformer blocks)
    ├── system/                ← Training orchestrators (one per model family)
    ├── evaluator/             ← Metrics computation
    │   └── metrics/           ← Hit@K, NDCG, BLEU, Distinct, etc.
    ├── quick_start/           ← run_crslab() — the main pipeline function
    └── download.py            ← Auto-download datasets and pretrained models
```

---

## 5. Component Deep-Dive

### 5.1 Config Module

**Location:** [crslab/config/](crslab/config/)

This module loads the YAML configuration file and makes all parameters accessible throughout the codebase.

**Key files:**
- [crslab/config/config.py](crslab/config/config.py) — `Config` class
- [crslab/config/__init__.py](crslab/config/__init__.py) — defines global path constants

**What it does:**

```python
# From run_crslab.py
config = Config('config/crs/kgsf/redial.yaml', gpu='-1', debug=False)

# Now config works like a dict:
config['model']           # → 'KGSF'
config['dataset']         # → 'ReDial'
config['token_emb_dim']   # → 300
config['rec']['epoch']    # → 9
```

**Global Path Constants** (defined in `__init__.py`):

| Constant | Default Path | Purpose |
|----------|-------------|---------|
| `SAVE_PATH` | `./saved` | Where trained systems (checkpoints) are stored |
| `DATA_PATH` | `./data` | Root for all downloaded data |
| `DATASET_PATH` | `./data/dataset` | Raw + processed dataset files |
| `MODEL_PATH` | `./data/model` | Pretrained model weights |
| `PRETRAIN_PATH` | `./data/pretrain` | BERT/GPT-2 weights |
| `EMBEDDING_PATH` | `./data/embedding` | Word2Vec/FastText embeddings |

**How Config loads YAML:**
1. Reads the YAML file with PyYAML
2. Sets up a Loguru logger (writes to `./log/` directory)
3. Parses GPU ids to set up `device` 
4. Makes all YAML keys accessible via `config['key']`

---

### 5.2 Data Module

**Location:** [crslab/data/](crslab/data/)

This is one of the most important modules. It handles everything from raw data download to creating batches for training.

**Registry System:**

```python
# crslab/data/__init__.py
dataset_register_table = {
    'ReDial': ReDialDataset,
    'TGReDial': TGReDialDataset,
    'GoRecDial': GoRecDialDataset,
    'DuRecDial': DuRecDialDataset,
    'INSPIRED': InspiredDataset,
    'OpenDialKG': OpenDialKGDataset,
}

dataloader_register_table = {
    'KGSF': KGSFDataLoader,
    'KBRD': KBRDDataLoader,
    'TGReDial': TGReDialDataLoader,
    ...
}
```

#### 5.2.1 BaseDataset

**Location:** [crslab/data/dataset/base.py](crslab/data/dataset/base.py)

This abstract class defines the contract all datasets must follow.

**Lifecycle:**

```
__init__() called
     │
     ├─ build() → downloads dataset files if not present (uses MD5 checksum)
     │
     ├─ [if not restore]
     │     ├─ _load_data() → reads raw JSON/text files, tokenizes, builds vocab
     │     └─ _data_preprocess() → converts to model-ready format
     │
     └─ [if restore]
           └─ _load_from_restore() → loads saved all_data.pkl
```

**What each data sample looks like** (the dict each training sample returns):

```python
{
    'role': 'Recommender',          # Who is speaking
    'context_tokens': [[4,12,99], [5,88,12,3]],  # Token IDs per dialogue turn
    'response': [5, 12, 99, 104],   # Token IDs of the target response
    'context_items': [1042, 88],     # Item IDs mentioned in conversation so far
    'items': [1042],                 # Items mentioned in THIS turn (ground truth rec)
    'context_entities': [22, 45, 9], # Entity IDs from knowledge graph in context
    'context_words': [300, 12, 44],  # Word IDs from word knowledge graph in context
    'context_policy': [[[0,1],[2,3]], [[1,4]]],  # Policy for each turn
    'target': [0, 5],               # Target policy for current turn
    'interaction_history': [88, 201, 44],  # Items user has interacted with before
    'user_profile': [[4,12],[88,99,3]]    # Token IDs of user profile sentences
}
```

**Side Data** (auxiliary data not per-sample, e.g. knowledge graphs):

```python
side_data = {
    'entity_kg': {
        'edge': [(head_id, tail_id, relation_id), ...],  # KG triples
        'n_relation': 64,                                 # Number of relation types
        'entity': ['Inception', 'Christopher Nolan', ...]  # Entity string names
    },
    'word_kg': {
        'edge': [(word_id1, word_id2), ...],  # Undirected word co-occurrence
        'entity': ['film', 'movie', 'actor', ...]
    },
    'item_entity_ids': [22, 45, 9, ...]  # Maps item index → entity KG index
}
```

#### 5.2.2 BaseDataLoader

**Location:** [crslab/data/dataloader/base.py](crslab/data/dataloader/base.py)

Takes the processed dataset samples and creates batches suitable for each model.

**Key responsibilities:**
- Padding sequences to uniform length
- Sorting by length (for RNN efficiency)
- Converting lists to PyTorch tensors
- Yielding batches in `(batch_data, interaction_data)` format

---

### 5.3 Model Module

**Location:** [crslab/model/](crslab/model/)

All models inherit from `BaseModel` which itself extends PyTorch's `nn.Module`.

#### 5.3.1 BaseModel

**Location:** [crslab/model/base.py](crslab/model/base.py)

```python
class BaseModel(ABC, nn.Module):
    def build_model(self):  # abstract — subclasses define architecture
        pass
    
    def recommend(self, batch, mode):   # optional — for recommendation task
        pass
    
    def converse(self, batch, mode):    # optional — for conversation task
        pass
    
    def guide(self, batch, mode):       # optional — for policy task
        pass
```

Each method receives a `batch` (dict of tensors) and `mode` ('train'/'valid'/'test') and returns a `(loss, predictions)` tuple.

#### 5.3.2 CRS Models (Full Systems)

These combine recommendation + conversation (+ optionally policy) into one unified model.

**KGSF (Knowledge Graph-based Semantic Fusion)**
- File: [crslab/model/crs/kgsf/](crslab/model/crs/kgsf/)
- Key idea: Uses two knowledge graphs (entity KG from DBpedia, word KG from ConceptNet) to enrich the representation of dialogue context
- Architecture: R-GCN (Relational Graph Convolutional Network) over entity KG + GCN over word KG → fused context → transformer-based conversation generator
- Training: 3 stages — pretrain (KG encoding), rec (item scoring), conv (response generation)

**KBRD (Knowledge-Based Reasoning and Dialogue)**
- File: [crslab/model/crs/kbrd/](crslab/model/crs/kbrd/)
- Key idea: Uses entity knowledge graph to propagate item-entity relationships for recommendation, then uses a transformer decoder with DB-grounded attention for generation
- Training: 2 stages — rec, then conv

**TG-ReDial (Topic-Guided)**
- File: [crslab/model/crs/tgredial/](crslab/model/crs/tgredial/)
- Key idea: Decomposes CRS into 3 explicit tasks with separate sub-models: a topic predictor (policy), a topic-guided response generator (conversation), and a collaborative filtering recommender
- Uses Chinese BERT and GPT-2

**ReDial**
- File: [crslab/model/crs/redial/](crslab/model/crs/redial/)
- The original CRS model from the ReDial dataset paper
- Uses a sentiment-aware autoencoder for recommendation + hierarchical RNN for generation

**INSPIRED**
- File: [crslab/model/crs/inspired/](crslab/model/crs/inspired/)
- Focuses on social strategies (appeals to quality, genre, popularity) in recommendation conversations

**NTRD (New Training Recipe for Dialogue)**
- File: [crslab/model/crs/ntrd/](crslab/model/crs/ntrd/)
- Decouples item selection from natural language generation

#### 5.3.3 Standalone Models

**Recommendation models** (used when `config` type is `recommendation`):

| Model | Key Idea |
|-------|---------|
| BERT | Fine-tuned BERT for item scoring from dialogue context |
| SASRec | Self-attention sequential recommendation (from interaction history) |
| GRU4Rec | GRU-based sequential recommendation |
| TextCNN | CNN on item content/descriptions |
| R-GCN | Relational GCN directly on entity knowledge graph |
| Popularity | Simple baseline: recommend most popular items |

**Conversation models** (used when `config` type is `conversation`):

| Model | Key Idea |
|-------|---------|
| GPT-2 | Fine-tuned autoregressive language model |
| Transformer | Standard seq2seq encoder-decoder |
| HERD | Hierarchical encoder-recurrent decoder |

**Policy models** (used when `config` type is `policy`):

| Model | Key Idea |
|-------|---------|
| Topic-BERT | BERT classifier for next topic prediction |
| Profile-BERT | BERT using user profile for action selection |
| Conv-BERT | BERT using full conversation context |
| MGCG | Multi-goal conversation graph |
| PMI | Pointwise Mutual Information baseline |

#### 5.3.4 Shared Utility Layers

**Location:** [crslab/model/utils/modules/](crslab/model/utils/modules/)

- `attention.py` — Multi-head attention, scaled dot-product attention
- `transformer.py` — Full transformer encoder/decoder blocks
- `utils.py` — Layer normalization, positional encoding, etc.

---

### 5.4 System Module

**Location:** [crslab/system/](crslab/system/)

Systems are the **training orchestrators**. They tie together the model, dataloaders, optimizer, and evaluator. Think of them as the "trainer" in other frameworks (like PyTorch Lightning's `Trainer`).

#### BaseSystem

**Location:** [crslab/system/base.py](crslab/system/base.py)

**Core responsibilities:**
- Moves model to the right device (CPU/GPU)
- Initializes optimizer(s) and LR schedulers from config
- Runs the train/valid/test epoch loops
- Calls `model.recommend()`, `model.converse()`, `model.guide()` per batch
- Passes predictions to evaluator
- Saves and loads model checkpoints
- Provides `interact()` mode for live chatting

**Training loop (simplified):**

```python
def fit(self):
    # Each model may have multiple training stages
    for stage in ['pretrain', 'rec', 'conv', 'policy']:
        if stage in config:
            for epoch in range(config[stage]['epoch']):
                # Training pass
                for batch in train_dataloader:
                    loss, preds = model.recommend(batch, 'train')  # or converse/guide
                    loss.backward()
                    optimizer.step()
                    evaluator.evaluate(preds, labels)
                
                # Validation pass
                for batch in valid_dataloader:
                    with torch.no_grad():
                        loss, preds = model.recommend(batch, 'valid')
                        evaluator.evaluate(preds, labels)
                
                evaluator.report()  # print metrics
```

#### System Implementations

| System | File | Special Behavior |
|--------|------|-----------------|
| KGSFSystem | [kgsf.py](crslab/system/kgsf.py) | 3-stage training: pretrain → rec → conv |
| KBRDSystem | [kbrd.py](crslab/system/kbrd.py) | 2-stage: rec → conv |
| ReDialSystem | [redial.py](crslab/system/redial.py) | Separate rec and conv training |
| TGReDialSystem | [tgredial.py](crslab/system/tgredial.py) | 3 tasks with separate sub-models and dataloaders |
| InspiredSystem | [inspired.py](crslab/system/inspired.py) | Social strategy-aware training |
| NTRDSystem | [ntrd.py](crslab/system/ntrd.py) | Decoupled item/text training |

---

### 5.5 Evaluator Module

**Location:** [crslab/evaluator/](crslab/evaluator/)

Computes and reports metrics after each epoch.

#### Evaluator Types

| Class | File | Used For |
|-------|------|---------|
| `RecEvaluator` | [rec.py](crslab/evaluator/rec.py) | Recommendation metrics only |
| `ConvEvaluator` | [conv.py](crslab/evaluator/conv.py) | Conversation metrics only |
| `StandardEvaluator` | [standard.py](crslab/evaluator/standard.py) | Both rec + conv |
| `EndToEndEvaluator` | [end2end.py](crslab/evaluator/end2end.py) | Full dialogue evaluation |

#### Metric Classes

**Recommendation Metrics** ([crslab/evaluator/metrics/rec.py](crslab/evaluator/metrics/rec.py)):

| Metric | Meaning | Formula |
|--------|---------|---------|
| Hit@K | Was the true item in the top-K predictions? | 1 if item in top-K else 0 |
| MRR@K | Mean Reciprocal Rank — how early in top-K is the true item? | 1/rank if rank ≤ K |
| NDCG@K | Normalized Discounted Cumulative Gain — position-weighted hit | log2(2) / log2(rank+1) |

**Conversation Metrics** ([crslab/evaluator/metrics/gen.py](crslab/evaluator/metrics/gen.py)):

| Metric | Meaning |
|--------|---------|
| BLEU-{1,2,3,4} | N-gram precision overlap between generated and reference response |
| Distinct-{1,2,3,4} | Ratio of unique n-grams — measures response diversity |
| PPL (Perplexity) | How surprised the model is by the ground truth — lower is better |
| Embedding Average | Cosine similarity between averaged word vectors of generated vs reference |
| Embedding Greedy | Greedy matching of word vectors between generated and reference |
| Embedding Extrema | Extrema pooling of word vectors |

---

### 5.6 Quick Start Module

**Location:** [crslab/quick_start/quick_start.py](crslab/quick_start/quick_start.py)

This is the **main pipeline orchestrator**. It wires together all the components in the right order.

```python
def run_crslab(config, save_data, restore_data, save_system, 
               restore_system, interact, debug, tensorboard):
    
    # 1. Build dataset (downloads data automatically)
    dataset = get_dataset(config, tokenize, restore_data, save_data)
    
    # 2. Create dataloaders
    train_dl = get_dataloader(config, dataset.train_data, vocab)
    valid_dl = get_dataloader(config, dataset.valid_data, vocab)
    test_dl  = get_dataloader(config, dataset.test_data,  vocab)
    
    # 3. Build system (model + trainer + evaluator bundled)
    CRS = get_system(config, train_dl, valid_dl, test_dl,
                     vocab, side_data, restore_system, interact, debug)
    
    # 4. Run
    if interact:
        CRS.interact()   # Chat with the trained model
    else:
        CRS.fit()        # Train the model
        if save_system:
            CRS.save_model()
```

**Multi-tokenizer case (TG-ReDial):** TG-ReDial uses two tokenizers — BERT-based for policy/conversation and a custom tokenizer for recommendation. The quick_start handles this by building separate datasets per tokenizer and passing dict-keyed dataloaders to the system.

---

## 6. Data Flow: End-to-End Walkthrough

Here's exactly what happens when you run `python run_crslab.py --config config/crs/kgsf/redial.yaml`:

```
run_crslab.py
    │
    ├─ 1. Config('config/crs/kgsf/redial.yaml')
    │      └─ Reads YAML → Config object with all hyperparameters
    │
    ├─ 2. get_dataset(config, tokenize='nltk')
    │      └─ ReDialDataset.__init__()
    │           ├─ build() → checks if data/dataset/redial/ exists
    │           │            downloads + verifies MD5 if not
    │           ├─ _load_data()
    │           │   ├─ reads raw JSON dialogue files
    │           │   ├─ NLTK tokenizes all text
    │           │   ├─ builds token vocab (word→id, id→word)
    │           │   └─ returns train/valid/test splits + vocab
    │           └─ _data_preprocess()
    │               ├─ converts each dialogue turn to standardized dict
    │               ├─ maps entity mentions to KG entity IDs
    │               ├─ maps movie mentions to item IDs
    │               └─ builds side_data (entity_kg, word_kg, item_entity_ids)
    │
    ├─ 3. get_dataloader(config, train_data, vocab)  [×3 for train/valid/test]
    │      └─ KGSFDataLoader(train_data, vocab, batch_size=128)
    │           └─ Wraps data, yields padded tensor batches on iteration
    │
    ├─ 4. get_system(config, dataloaders, vocab, side_data)
    │      └─ KGSFSystem.__init__()
    │           ├─ get_model() → KGSFModel(opt, device, vocab, side_data)
    │           │                 └─ build_model():
    │           │                     ├─ R-GCN layers over entity_kg
    │           │                     ├─ GCN layers over word_kg
    │           │                     ├─ Transformer decoder for generation
    │           │                     └─ item scoring MLP
    │           ├─ get_evaluator() → StandardEvaluator
    │           └─ sets up 3 Adam optimizers (one per training stage)
    │
    └─ 5. CRS.fit()
           ├─ Stage 1: Pretrain (3 epochs)
           │   └─ model.pretrain_infomax(batch) → mutual info loss on KG embeddings
           │
           ├─ Stage 2: Rec (9 epochs)
           │   ├─ model.recommend(batch, 'train') → cross-entropy loss on item scores
           │   ├─ evaluator.rec_evaluate(preds, items) → Hit/MRR/NDCG
           │   └─ evaluator.report()
           │
           └─ Stage 3: Conv (90 epochs)
               ├─ model.converse(batch, 'train') → token-level cross-entropy loss
               ├─ evaluator.conv_evaluate(preds, responses) → BLEU/Distinct/PPL
               └─ evaluator.report()
```

---

## 7. Supported Datasets

| Dataset | Lang | Domain | Dialogues | Utterances | Entity KG | Word KG |
|---------|------|--------|-----------|-----------|-----------|---------|
| ReDial | EN | Movie | 10,006 | 182,150 | DBpedia | ConceptNet |
| TG-ReDial | ZH | Movie | 10,000 | 129,392 | CN-DBpedia | HowNet |
| GoRecDial | EN | Movie | 9,125 | 170,904 | DBpedia | ConceptNet |
| DuRecDial | ZH | Movie+Music | 10,200 | 156,000 | CN-DBpedia | HowNet |
| INSPIRED | EN | Movie | 1,001 | 35,811 | DBpedia | ConceptNet |
| OpenDialKG | EN | Movie+Book | 13,802 | 91,209 | DBpedia | ConceptNet |

**All datasets are downloaded automatically** on first run. They are stored in `data/dataset/<DatasetName>/`.

---

## 8. Supported Models

### Full CRS Models

| Model | Dataset | Rec | Conv | Policy | Paper |
|-------|---------|-----|------|--------|-------|
| KGSF | ReDial, GoRecDial, INSPIRED | R-GCN+KG | Transformer | - | SIGIR 2020 |
| KBRD | ReDial, GoRecDial | R-GCN+KG | Transformer+DB | - | EMNLP 2019 |
| ReDial | ReDial | Autoencoder | HRED | - | NeurIPS 2018 |
| TG-ReDial | TG-ReDial | CF | GPT-2 | BERT | ACL 2021 |
| INSPIRED | INSPIRED | Autoencoder | Transformer | - | EMNLP 2020 |
| NTRD | TG-ReDial | BERT | GPT-2 | - | ACL 2021 |

### Standalone Models

**Recommendation:** BERT, SASRec, GRU4Rec, TextCNN, R-GCN, Popularity  
**Conversation:** GPT-2, Transformer, HERD  
**Policy:** Topic-BERT, Profile-BERT, Conv-BERT, MGCG, PMI  

---

## 9. Configuration Files Explained

**Location:** `config/crs/kgsf/redial.yaml` (example)

```yaml
# ── Dataset & Preprocessing ──────────────────────────────────────────
dataset: ReDial           # Which dataset class to load (from registry)
tokenize: nltk            # Tokenizer to use (nltk | bert | jieba | etc.)
embedding: word2vec.npy   # Optional: pre-trained word embedding file

# ── Dataloader ────────────────────────────────────────────────────────
context_truncate: 256     # Max tokens in dialogue context (truncated from right)
response_truncate: 30     # Max tokens in response
scale: 1                  # Data scaling factor (1.0 = use full dataset)

# ── Model Architecture ────────────────────────────────────────────────
model: KGSF               # Which model class to use (from registry)
token_emb_dim: 300        # Word embedding dimension
kg_emb_dim: 128           # Knowledge graph embedding dimension
num_bases: 8              # Number of basis decompositions in R-GCN
n_heads: 2                # Transformer attention heads
n_layers: 2               # Transformer encoder/decoder layers
ffn_size: 300             # Feed-forward network hidden size
dropout: 0.1              # Dropout probability
n_positions: 1024         # Maximum sequence positions (for positional encoding)

# ── Training Stage: Pretrain ──────────────────────────────────────────
pretrain:
  epoch: 3                # Number of pretrain epochs
  batch_size: 128
  optimizer:
    name: Adam
    lr: 1e-3              # Learning rate

# ── Training Stage: Recommendation ───────────────────────────────────
rec:
  epoch: 9
  batch_size: 128
  optimizer:
    name: Adam
    lr: 1e-3

# ── Training Stage: Conversation ─────────────────────────────────────
conv:
  epoch: 90
  batch_size: 128
  optimizer:
    name: Adam
    lr: 1e-3
  lr_scheduler:
    name: ReduceLROnPlateau   # Reduce LR when validation loss plateaus
    patience: 3               # Wait 3 epochs before reducing
    factor: 0.5               # Multiply LR by 0.5 on plateau
  gradient_clip: 0.1          # Gradient norm clipping threshold
```

**How config keys are resolved:**
- `dataset` → `dataset_register_table[dataset]` → dataset class
- `model` → `model_register_table[model]` → model class
- `model` → `system_register_table[model]` → system class
- `model` → `dataloader_register_table[model]` → dataloader class

---

## 10. Running the Project

### Basic Training

```bash
cd /Users/tilakgupta/Desktop/CRSLab
conda activate crslab

# Train KGSF on ReDial dataset (CPU mode — good for M2)
python run_crslab.py --config config/crs/kgsf/redial.yaml --gpu -1
```

### Save/Restore Processed Data (speeds up repeated runs)

```bash
# First run: process data and save
python run_crslab.py --config config/crs/kgsf/redial.yaml --save_data

# Subsequent runs: skip preprocessing
python run_crslab.py --config config/crs/kgsf/redial.yaml --restore_data
```

### Save/Restore Trained Model

```bash
# Train and save model checkpoint
python run_crslab.py --config config/crs/kgsf/redial.yaml --save_system

# Load trained model and test
python run_crslab.py --config config/crs/kgsf/redial.yaml --restore_system
```

### Debug Mode (fast iteration — uses validation set as training set)

```bash
python run_crslab.py --config config/crs/kgsf/redial.yaml --debug
```

### Interactive Mode (chat with the trained system)

```bash
python run_crslab.py --config config/crs/kgsf/redial.yaml \
    --restore_system --interact
```

### TensorBoard Monitoring

```bash
# Terminal 1: Train with TensorBoard logging
python run_crslab.py --config config/crs/kgsf/redial.yaml --tensorboard

# Terminal 2: Launch TensorBoard
tensorboard --logdir ./log
# Open http://localhost:6006 in browser
```

### All CLI Flags Reference

| Flag | Short | Description |
|------|-------|-------------|
| `--config PATH` | `-c` | Path to YAML config (default: tgredial/tgredial.yaml) |
| `--gpu IDs` | `-g` | GPU ids, e.g. `0`, `0,1`, or `-1` for CPU |
| `--save_data` | `-sd` | Save preprocessed dataset to disk |
| `--restore_data` | `-rd` | Load preprocessed dataset from disk |
| `--save_system` | `-ss` | Save trained model checkpoint |
| `--restore_system` | `-rs` | Load model from checkpoint |
| `--debug` | `-d` | Quick debug run using validation set |
| `--interact` | `-i` | Interactive chat mode (needs restored system) |
| `--tensorboard` | `-tb` | Enable TensorBoard metric logging |

---

## 11. Evaluation Metrics

### Recommendation Metrics

Given a list of recommended items ranked by score, and the ground truth item(s):

- **Hit@K** — Binary: did any ground truth item appear in top-K?
  ```
  Hit@1=0.02 means: 2% of turns had the correct item as the #1 prediction
  Hit@50=0.15 means: 15% of turns had the correct item somewhere in top-50
  ```

- **MRR@K** — Mean Reciprocal Rank: `1/rank` averaged over turns
  ```
  MRR@10=0.011 means: on average, the correct item appears at position 1/0.011 ≈ 91
  ```

- **NDCG@K** — Normalized Discounted Cumulative Gain: rewards earlier correct predictions more
  ```
  NDCG@10=0.015 (higher is better)
  ```

### Conversation Metrics

- **BLEU-N** — Precision of N-grams in the generated response vs reference
  ```
  BLEU@1=0.38 means: 38% of unigrams in the generated response appear in the reference
  ```

- **Distinct-N** — Vocabulary diversity: `(unique N-grams) / (total N-grams)` in generation
  ```
  Dist@1=0.03 is low (repetitive) | Dist@2=0.91 is high (diverse)
  ```

- **PPL (Perplexity)** — Model's surprise at the reference text — lower is better
  ```
  PPL=7.4 means: model assigns average probability of 1/7.4 per token to reference
  ```

---

## 12. Troubleshooting

### macOS (M2) — `fasttext` fails to install

```bash
# Try the prebuilt wheel fork
pip install fasttext-wheel

# Or install via conda
conda install -c conda-forge fasttext
```

### macOS — `sentencepiece` build errors

```bash
# Install system-level sentencepiece library first
brew install sentencepiece
pip install sentencepiece
```

### All platforms — `numpy` version conflict

The project pins `numpy~=1.19.4` but M2 native wheels start from ~1.21. Use a newer compatible version:

```bash
pip install "numpy>=1.21,<2.0"
```

Then if you see a `numpy` version warning when running the project, it's safe to ignore — the API used here is compatible.

### All platforms — `torch_geometric` import errors

```bash
# Reinstall with explicit version matching
pip uninstall torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -y

TORCH=$(python -c "import torch; print(torch.__version__)")
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-${TORCH}+cpu.html
pip install torch-geometric
```

### All platforms — `pkuseg` or Chinese tokenizer errors

`pkuseg` is only needed for Chinese datasets (TG-ReDial, DuRecDial). If you're only using English datasets, you can skip it.

```bash
# If pkuseg has C++ compilation issues:
pip install pkuseg --no-build-isolation
```

### macOS — GPU not recognized (MPS)

CRSLab uses CUDA-style GPU indexing. On M2, pass `--gpu -1` to use CPU. If you want to experiment with MPS, you would need to modify [crslab/config/config.py](crslab/config/config.py) to map device appropriately — this is a code change, not a config change.

### Linux — CUDA device not found

If `torch.cuda.is_available()` returns `False`:

```bash
# Check that your driver and CUDA toolkit versions match
nvidia-smi           # shows driver version
nvcc --version       # shows toolkit version

# Reinstall PyTorch with the correct CUDA index URL
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Pass the GPU index to CRSLab with `--gpu 0` (or `0,1` for multi-GPU).

### Linux — `libstdc++` / `GLIBCXX` version errors

Occurs when conda's bundled libstdc++ is older than the one used to compile a wheel:

```bash
conda install -c conda-forge libstdcxx-ng -y
```

### Windows — C++ compiler not found during `pip install`

If you see `error: Microsoft Visual C++ 14.0 or greater is required`:

1. Make sure **Visual Studio Build Tools** are installed (see section 2.3 Step 2).
2. Open a **Developer Command Prompt for VS** or the **Miniforge Prompt** which inherits the VS environment, then retry.
3. Alternatively, install the package via conda-forge which ships pre-built binaries:
   ```powershell
   conda install -c conda-forge sentencepiece fasttext -y
   ```

### Windows — `fasttext` build failure

```powershell
# Use the pre-built wheel fork
pip install fasttext-wheel

# Or install via conda-forge (no compilation needed)
conda install -c conda-forge fasttext -y
```

### Windows — long path errors (`FileNotFoundError`)

Windows limits paths to 260 characters by default. Enable long path support:

1. Open **Group Policy Editor** (`gpedit.msc`) → Computer Configuration → Administrative Templates → System → Filesystem → Enable **Win32 long paths**.
2. Or run this in an **elevated** PowerShell:
   ```powershell
   Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
       -Name LongPathsEnabled -Value 1
   ```

### Windows — `pkuseg` segmentation fault

```powershell
# Install the pure-Python fallback
pip install pkuseg --no-build-isolation
```

If it still crashes, switch to using English-only datasets (`ReDial`, `OpenDialKG`, `INSPIRED`) which do not depend on `pkuseg`.

### All platforms — Data download failures

If auto-download fails (network issues or changed URLs), you can manually download datasets. Check [crslab/data/dataset/redial/](crslab/data/dataset/redial/) for the URL and expected MD5 in each dataset's `__init__.py`.

### All platforms — Memory issues during training

On memory-constrained machines (e.g. 8 GB RAM), reduce batch size in the YAML config:

```yaml
rec:
  batch_size: 32    # reduced from 128
conv:
  batch_size: 32    # reduced from 128
```

---

## Quick Reference Card

```
Project root: wherever you cloned the repo (e.g. ~/Desktop/CRSLab/ on macOS/Linux)

Entry point:       run_crslab.py
Config files:      config/crs/<model>/<dataset>.yaml
Package source:    crslab/

Key registry files (maps name strings → classes):
  Data:      crslab/data/__init__.py
  Models:    crslab/model/__init__.py
  Systems:   crslab/system/__init__.py
  Evaluators: crslab/evaluator/__init__.py

Adding a new model:
  1. Create crslab/model/crs/<yourmodel>/<yourmodel>.py  (extend BaseModel)
  2. Create crslab/system/<yourmodel>.py               (extend BaseSystem)
  3. Create crslab/data/dataloader/<yourmodel>.py       (extend BaseDataLoader)
  4. Register in the __init__.py registry tables
  5. Write a YAML config under config/crs/<yourmodel>/
```
