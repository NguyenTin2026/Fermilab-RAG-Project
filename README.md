# Fermilab Project: Data Crawling & Domain-Specific Language Model Fine-Tuning (LLM & Embeddings)

This project is a comprehensive end-to-end repository consisting of two main tracks to harvest and process scientific/technical documents, construct synthetic training pairs, and fine-tune specialized models for particle physics and tech-scouting applications:

1. **Fermilab Q&A Assistant Track**: An automated pipeline that crawls Fermilab websites, cleans and chunks technical web content, utilizes the Gemini API to synthesize high-quality Question-Answering pairs, fine-tunes a `Qwen/Qwen3-4B-Instruct-2507` model via QLoRA, and serves it through a Streamlit dashboard powered by Retrieval-Augmented Generation (RAG).
2. **Discovery Hub Track (Embedding Fine-Tuning)**: A pipeline that ingests, cleans, and standardizes diverse technical/legal documents (USPTO patents, AUTM invention listings) and maps them against clinical/business research queries (ClinicalTrials.gov), generates matching query-document pairs, mines hard negatives, and fine-tunes an embedding model (`Qwen/Qwen3-Embedding-0.6B` or 4B LoRA) via Multiple Negatives Ranking Loss (MNRL).

---

## 📂 File and Directory Map

Below is the core structure of the repository:

* **Root Directory (`/home/x-ndo5/Fermilab_Project/`)**:
  * [fermilab.py](file:///home/x-ndo5/Fermilab_Project/fermilab.py): BFS web crawler script targeting public Fermilab domains.
  * [wait_and_report.py](file:///home/x-ndo5/Fermilab_Project/wait_and_report.py): Script to automatically wait for the crawler milestone output and construct the submission report.
  * **[Fermilab-process_data_and_finetune_model](file:///home/x-ndo5/Fermilab_Project/Fermilab-process_data_and_finetune_model)**: Directory for corpus processing and Q&A model fine-tuning.
    * [01b_process_text_corpus.py](file:///home/x-ndo5/Fermilab_Project/Fermilab-process_data_and_finetune_model/01b_process_text_corpus.py): Cleaning, boilerplate stripping, greedy-packing text into chunks, and deterministic train/test splitting.
    * [02_generate_qa.py](file:///home/x-ndo5/Fermilab_Project/Fermilab-process_data_and_finetune_model/02_generate_qa.py): Generates grounded Q&A pairs from training chunks using the Gemini API.
    * [03_finetune_qlora.py](file:///home/x-ndo5/Fermilab_Project/Fermilab-process_data_and_finetune_model/03_finetune_qlora.py): 4-bit QLoRA training script. Includes `--chat` and `--merge` modes.
    * [app.py](file:///home/x-ndo5/Fermilab_Project/Fermilab-process_data_and_finetune_model/app.py): Streamlit web application serving the fine-tuned model with a hybrid RAG search engine, RAG vs. No-RAG comparison, metrics dashboard, and corpus topology map.
    * [retriever.py](file:///home/x-ndo5/Fermilab_Project/Fermilab-process_data_and_finetune_model/retriever.py): Hybrid RAG retriever combining BM25 lexical search and dense embeddings via Reciprocal Rank Fusion (RRF).
  * **[Discovery code and instructions](file:///home/x-ndo5/Fermilab_Project/Discovery%20code%20and%20instructions)**: Directory for embedding model fine-tuning.
    * [parse.py](file:///home/x-ndo5/Fermilab_Project/Discovery%20code%20and%20instructions/parse.py): Standardizes USPTO, AUTM, and ClinicalTrials documents into a unified schema.
    * [generate_queries.py](file:///home/x-ndo5/Fermilab_Project/Discovery%20code%20and%20instructions/generate_queries.py): Generates clinical/business research queries from technical documents using Gemini API.
    * [build_dataset.py](file:///home/x-ndo5/Fermilab_Project/Discovery%20code%20and%20instructions/build_dataset.py): Mines hard negative documents using the base model and creates train/eval splits.
    * [train.py](file:///home/x-ndo5/Fermilab_Project/Discovery%20code%20and%20instructions/train.py): Fine-tunes the embedding model via Multiple Negatives Ranking Loss (MNRL).
    * [evaluate.py](file:///home/x-ndo5/Fermilab_Project/Discovery%20code%20and%20instructions/evaluate.py): Computes Recall@k and MRR@k metrics for evaluation.
    * [Makefile](file:///home/x-ndo5/Fermilab_Project/Discovery%20code%20and%20instructions/Makefile): Declares commands for automating the embedding fine-tuning steps.
    * [requirements.txt](file:///home/x-ndo5/Fermilab_Project/Discovery%20code%20and%20instructions/requirements.txt): Dependency package list for the Discovery Hub track.

---

## ⚡ Track 1: Fermilab Q&A Assistant Pipeline

This pipeline builds a domain-specific Q&A model trained on crawled Fermilab documents. Data flows as follows:
```
crawled web pages (.txt) ──► [Stage 1] Process ──► corpus_st7.jsonl ──► [Stage 2] Generate QA ──► train_qa.jsonl ──► [Stage 3] Fine-tune QLoRA ──► ft-qwen3-fermilab/
```

### 0. Environment Setup

Activate the pre-existing virtual environment (e.g., `ndo1`):
```bash
cd /home/x-ndo5/Fermilab_Project/Fermilab-process_data_and_finetune_model
source ndo1/bin/activate
```
Install dependencies if required (append `--break-system-packages` if system package controls prevent basic pip installs):
```bash
pip install --break-system-packages google-genai torch transformers trl peft bitsandbytes accelerate datasets PyMuPDF
```
Acquire a Gemini API key from [Google AI Studio](https://aistudio.google.com) and export it to your environment:
```bash
export GEMINI_API_KEY=AIzaSy...your_key...
```

### Step 1: Data Collection (Web Crawler)
Execute the web crawler to download up to 1,000 public pages from Fermilab subdomains (`news.fnal.gov`, `education.fnal.gov`, etc.):
```bash
cd /home/x-ndo5/Fermilab_Project
python3 fermilab.py
```
* **How it works**: Starts from 4 seed URLs, runs BFS link discovery restricted to approved domains, and observes a polite 1.5-second delay between requests.
* **Outputs**: 
  * Raw HTML pages at `crawled_data/html_pages/`
  * Clean text pages at `crawled_data/text_pages/`
  * Metadata spreadsheet tracking success/fail status at [download_report_1000.csv](file:///home/x-ndo5/Fermilab_Project/crawled_data/download_report_1000.csv)

To monitor checkpoints and output the final assignment report automatically, run:
```bash
python3 wait_and_report.py
```

### Step 2: Corpus Ingestion & Processing
To clean text documents, strip recurring website headers/footers (site chrome/boilerplate), partition text into 350-450 word chunks, and enforce a leakage-safe Train/Test split:
```bash
cd /home/x-ndo5/Fermilab_Project/Fermilab-process_data_and_finetune_model
python3 01b_process_text_corpus.py \
    --text_dir ../crawled_data/text_pages \
    --report ../crawled_data/download_report_1000.csv \
    --out corpus_st7.jsonl
```
* **Reference Metrics**: Successfully processed 2,233 text documents into 8,847 chunks, yielding 7,100 train chunks (80.25%) and 1,747 test chunks (19.75%). Strip-filtered 68 unique boilerplate lines that occurred on 30% or more pages.

### Step 3: Synthetic Q&A Generation
Run the offline validation self-test to verify formatting and deduplication logic without calling the API:
```bash
python3 02_generate_qa.py --selftest
```
Run Q&A generation using Gemini (fully resume-safe; if interrupted with Ctrl+C, it will resume from the last unprocessed chunk):
```bash
python3 02_generate_qa.py --corpus corpus_st7.jsonl --out train_qa.jsonl
```
* **Anti-Leakage Design**: The script only reads chunks marked as `train` in `corpus_st7.jsonl` and drops generated questions sharing an 8-gram sequence with the evaluation gold set.

### Step 4: QLoRA Fine-Tuning
This is a computationally heavy step requiring a GPU with at least 12GB VRAM. It trains the default `Qwen/Qwen3-4B-Instruct-2507` model.
Run inside a `tmux` session to ensure training is unaffected by SSH disconnections:
```bash
tmux new -s finetune
python3 03_finetune_qlora.py --data train_qa.jsonl --out ./ft-qwen3-fermilab
```
* **Hyperparameters**: Effective batch size of 16 (per-device batch 2 * gradient accumulation 8). Cosine learning rate schedule starting at $2 \times 10^{-4}$ with 3% warmup. Max sequence length of 2,048 tokens. 4-bit NF4 quantization.
* **Reference Results**:
  * Training completed in 232 seconds for 54 steps (2 epochs) on a single GPU.
  * Training loss dropped ~58% (from **2.921** down to **1.101**).
  * Mean token accuracy rose from **56.3%** to **73.88%** (training) and **72.6%** (validation).
  * Output adapters are saved in `./ft-qwen3-fermilab`.

### Step 5: Validation & Model Merging
To inspect base vs. fine-tuned responses side-by-side on verification questions:
```bash
python3 03_finetune_qlora.py --chat ./ft-qwen3-fermilab
```
*(If you encounter a `transformers` version-specific generation error, please refer to the Troubleshooting section).*

Once satisfied with performance, bake the adapters back into the base model weights for simple, fast inference:
```bash
python3 03_finetune_qlora.py --merge ./ft-qwen3-fermilab
```
The full weights will be written to `./ft-qwen3-fermilab/merged`.

### Step 6: Deploy Streamlit Dashboard
Launch the web interface for interactive QA and verification:
```bash
streamlit run app.py --server.fileWatcherType none
```
Open the provided URL in your web browser. The app features:
* **Standard Chat Engine**: A chat window comparing RAG grounding vs. open-ended generation.
* **Hallucination reduction tests**: Runs comparative analysis side-by-side with RAG active/inactive.
* **Corpus Topology Map**: 2D scatter visualization of topic density in the crawled documents.
* **Metrics Dashboard**: Visualizes crawler progress, HTTP status distribution, and model training metrics.

---

## ⚡ Track 2: Discovery Hub (Embedding Fine-Tuning)

This track focuses on training a retrieval/embedding model to bridge the vocabulary and register gap between technical/legal descriptions and clinical/business inquiries:
* **Document side**: Sáng chế/Patent (USPTO - st5) and technology transfer listings (AUTM - st1) in technical language.
* **Query side**: Clinical trials (st2) and scout research interests in clinical/business registers.

### ⚠️ Package Path Notice
The scripts in this track are packaged as a module named `discovery_ft`. To run commands via `make` or python CLI without hitting a `ModuleNotFoundError`, you must do one of the following:
1. **Set `PYTHONPATH` (Recommended)**: Set the environment variable to point to the source folder:
   ```bash
   export PYTHONPATH="/home/x-ndo5/Fermilab_Project/Discovery code and instructions"
   ```
2. **Rename/Symlink Directory**: Rename or create a symlink named `discovery_ft` mapping to the folder.

### 0. Environment Setup
Install the specific dependencies for sentence embeddings:
```bash
cd "/home/x-ndo5/Fermilab_Project/Discovery code and instructions"
pip install -r requirements.txt --break-system-packages
```
Ensure your `GEMINI_API_KEY` is exported in the shell.

### Command Execution Workflow (via Makefile)
We provide a structured [Makefile](file:///home/x-ndo5/Fermilab_Project/Discovery%20code%20and%20instructions/Makefile) to automate all steps of the embedding pipeline. Run `make help` to inspect available steps.

#### 1. Ingest and Inspect Dictionaries
```bash
make inspect
```
Runs [parse.py](file:///home/x-ndo5/Fermilab_Project/Discovery%20code%20and%20instructions/parse.py). It reads raw JSON files, parses USPTO patents, ClinicalTrials, and AUTM pages, strips legal boilerplate, maps CPC codes, and aggregates them into a standardized `embedding_text` schema.

#### 2. Generate Synthetic Training Queries
```bash
make pairs
```
Runs [generate_queries.py](file:///home/x-ndo5/Fermilab_Project/Discovery%20code%20and%20instructions/generate_queries.py). It reads the documents, prompts Gemini API with few-shot business registers, and creates matching query-document pairs saved to `pairs.jsonl`.
* *Tip*: Dry-run prompt verification can be run for free using:
  ```bash
  python -m discovery_ft.generate_queries --in data --dry-run | head -40
  ```

#### 3. Mine Hard Negatives & Build Splits
```bash
make dataset
```
Runs [build_dataset.py](file:///home/x-ndo5/Fermilab_Project/Discovery%20code%20and%20instructions/build_dataset.py). The base model encodes all documents and finds the most similar documents (excluding the true target) to act as hard contrastive negative samples. Writes the outputs to `dataset/train.jsonl` and `eval.jsonl`.

#### 4. Benchmark Baseline Model (Before Fine-Tuning)
```bash
make baseline
```
Runs [evaluate.py](file:///home/x-ndo5/Fermilab_Project/Discovery%20code%20and%20instructions/evaluate.py) to assess the untrained model's Recall@k and MRR@10 performance on the held-out evaluation triplets.

#### 5. Fine-Tune Embedding Model
```bash
make train
```
Runs [train.py](file:///home/x-ndo5/Fermilab_Project/Discovery%20code%20and%20instructions/train.py) to train `Qwen/Qwen3-Embedding-0.6B` using Multiple Negatives Ranking Loss (MNRL). Features like mixed-precision (`fp16`/`bf16`) and gradient checkpointing are pre-configured to fit onto a single 12GB GPU.
* *Note on larger architectures*: To train a larger 4B model using parameter-efficient adapters (LoRA), add the `--lora` flag:
  ```bash
  python -m discovery_ft.train --model Qwen/Qwen3-Embedding-4B --lora --batch 4 --grad-accum 4 --out runs/qwen4b-lora
  ```

#### 6. Evaluate Fine-Tuned Model
```bash
make eval
```
Runs [evaluate.py](file:///home/x-ndo5/Fermilab_Project/Discovery%20code%20and%20instructions/evaluate.py) using the fine-tuned checkpoint at `runs/qwen06b-ft/final` to measure the metrics improvement against the baseline recorded in Step 4.

To run the entire pipeline in a single command sequence:
```bash
make all
```

---

## 🛠️ Troubleshooting & Frequently Asked Questions (FAQs)

| Error Message / Issue | Root Cause | Resolution |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'fitz'` | Trying to run PDF parser (`01_process_corpus.py`) on crawled raw text data. | Run `python3 -m pip install PyMuPDF` (do not install the package named `fitz`) or use the text-ingest variant [01b_process_text_corpus.py](file:///home/x-ndo5/Fermilab_Project/Fermilab-process_data_and_finetune_model/01b_process_text_corpus.py) instead. |
| `error: externally-managed-environment` | System Python environment blocks external packages to maintain OS integrity. | Append `--break-system-packages` to the `pip install` command string. |
| `KeyError: 'GEMINI_API_KEY'` | Gemini API authorization key not found in shell environment variables. | Set the key: `export GEMINI_API_KEY=your_actual_key_from_google_ai_studio`. |
| `AttributeError` / `KeyError` in `--chat` interface comparison | In newer `transformers` library versions (5.x), the dictionary must be unpacked explicitly when calling `model.generate()`. | Patch the `ask()` function in the script to pass tokenizer dict inputs as unpacked kwargs: `model.generate(**ids, ...)` and access prompt length via `ids["input_ids"].shape[-1]`. |
| Out-Of-Memory (OOM) on GPU | Batch size is too high for VRAM capacity. | Decrease `--batch` (e.g. to 8 or 4) and increase `--grad-accum` to match the target effective batch size. |

---

## 🔮 Limitations & Future Work

1. **Synthesized Training Size**: 438 training pairs is small for a 4B parameter language model. Scaling the dataset using varied generation temperatures and more diverse seed prompts will improve out-of-domain robustness.
2. **Retrieval Reranking**: Integrating a cross-encoder model as a secondary stage inside `retriever.py` to rerank the top retrieved documents will yield a significant boost in precision.
3. **Retrieval-Augmented Fine-Tuning (RAFT)**: Tinh chỉnh (fine-tune) models on prompts that deliberately mix relevant context alongside unrelated distractors. This forces the model to ignore noisy information during RAG execution.
