# 🧠 LitGHRAG

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![GPU](https://img.shields.io/badge/CUDA-Supported-orange.svg)](https://developer.nvidia.com/cuda-zone)
[![LLM](https://img.shields.io/badge/LLM-Compatible%20API-lightgrey.svg)](https://platform.openai.com/docs/api-reference)

**A Lightweight Heterogeneous Graph RAG Framework for Precise Retrieval and Adaptive Generation**

LitGHRAG builds a heterogeneous knowledge graph from QA datasets, retrieves both passages and structured graph facts, and queries an LLM to answer **multi-hop questions** efficiently and interpretably.

---

## 🚀 Highlights

- 🕸 **Heterogeneous KG** integrating entities, events, concept, and passage nodes for structured yet flexible knowledge representation
- 🔍 **Hybrid retrieval** that jointly leverages subgraphs with personalized PageRank (PPR)-based propagation to capture global relevance
- 🧩 **Symbolic path reasoning** over induced subgraphs to improve multi-hop explainability and factual grounding
- ⚖️ **CrossEncoder reranker** for fine-grained passage ordering
- 🪶 **Adaptive generation** module that dynamically adjusts reasoning depth and response style based on query complexity
- ⚙️ **Configurable components**:
  - Encoder: Sentence-Transformers / NV-Embed
  - Reader: local (Ollama) or cloud (OpenAI-compatible) LLM
- 📊 **Built-in benchmarking** for:
  - HotpotQA
  - 2WikiMultihopQA
  - MuSiQue
  - PopQA
  - NQ-ReaR

---

## 📁 Project Structure

```
litgh_rag/
 ├── kg_construction/     # Triple extraction, concept generation, GraphML export
 ├── retrieval/           # Embedding models, indexers, retrievers
 ├── evaluation/          # Benchmarking and metrics
 ├── reader/              # LLM generation utilities
 ├── utils/               # Data conversion, helpers
 ├── 1_kg_construction.py # Build KG from dataset
 ├── 3_litgh_rag.py       # End-to-end RAG pipeline
 ├── benchmark_data/      # Raw datasets (JSON)
 ├── kg_dataset/          # Preprocessed corpus for KG building
 ├── graph_data/          # Final .graphml files for retrieval
 └── result/              # Evaluation outputs
```

---

## 🧩 Requirements

- Python ≥ 3.9
- Linux (CUDA GPU recommended; CPU supported but slower)

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> 💡 _For GPU FAISS, install a version matching your CUDA.  
> If using Ollama or a local LLM server, ensure it supports the OpenAI-compatible API (base_url + api_key)._

---

## 📚 Supported Datasets

Available keywords (`--keyword` argument):

- `hotpotqa`
- `2wikimultihopqa`
- `musique`
- `popqa`
- `nq_rear`

---

## ⚙️ Step 1: Build the Knowledge Graph

**Script:** `1_kg_construction.py`  
Generates triples, concepts, and the final GraphML file using an LLM-based extractor.

```bash
python 1_kg_construction.py --keyword 2wikimultihopqa
```

**Outputs:**

```
./dataset/{keyword}/{model_name}/
 ├── triples.json / triples.csv
 ├── concepts.csv
 └── {keyword}.graphml
```

---

## 🧠 Step 2: Question Classification

Classify questions (type/routing) before retrieval and reasoning.

```bash
python 2_qa_classifier.py
```

---

## 🔎 Step 3: Retrieval + Adaptive Generation

**Script:** `3_litgh_rag.py`

Runs retrieval over the constructed KG, queries the LLM, and evaluates results.

```bash
python 3_litgh_rag.py --keyword 2wikimultihopqa
```

**Results:**

Metrics computed in `atlas_rag/evaluation/benchmark.py`:

- **Exact Match (EM)**
- **F1 Score**
- **Recall@k**

Output files:

```
result/{dataset}/
 ├── summary_*.json
 └── result_*.json
```

---

## ⚙️ Model Configuration

### 🔤 Encoder (Sentence Embeddings)

Default:

```python
from sentence_transformers import SentenceTransformer
encoder_model_name = "sentence-transformers/all-MiniLM-L6-v2"
sentence_model = SentenceTransformer(encoder_model_name, trust_remote_code=True, model_kwargs={'device_map': 'auto'})
```

Switch to **NV-Embed**:

```python
from transformers import AutoModel
from atlas_rag.retrieval.embedding_model import NvEmbed
model = AutoModel.from_pretrained("nvidia/NV-Embed-v2")
sentence_encoder = NvEmbed(model)
```

---

### ⚖️ CrossEncoder Reranker

```python
from sentence_transformers import CrossEncoder
rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
```

---

### 🤖 Reader LLM (OpenAI-compatible)

Both scripts support local or remote endpoints:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
```

Example model names:

- Local: `qwen3:4b`, `llama3.1:8b`
- Remote: `gpt-4o-mini`, `claude-3-sonnet`, etc.

---

## 🧰 Command-Line Usage

```bash
# Build KG
python 1_kg_construction.py --keyword hotpotqa

# Run retrieval + generation
python 3_litgh_rag.py --keyword hotpotqa
```

---

## 🧭 Tips & Troubleshooting

- 💾 **Out-of-memory (OOM):** Reduce batch size in `create_embeddings_and_index` or KG construction.
- 🧩 **FAISS errors:** Ensure embedding dimension matches the index; rebuild if encoder changes.
- 📂 **Missing files:** Check paths — both raw data and final `.graphml` must exist.
- 🌐 **Hugging Face caching:** Set `HF_HOME` or `TRANSFORMERS_CACHE` for a custom path.
