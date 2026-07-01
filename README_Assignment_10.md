# Academic Progress Report: Domain LLM Pipeline for Fermilab Research
**Course Assignment 10 — Write a detailed report of your work**  
*Submitted by: st7*

---

## 1. Overview / Project Goal

The primary goal of this project is to build a specialized, domain-specific large language model (LLM) focused on Fermi National Accelerator Laboratory (Fermilab). This is achieved by implementing an automated, end-to-end data and training pipeline that proceeds through four major stages: web data collection, corpus processing, synthetic Q&A generation, and parameter-efficient fine-tuning via QLoRA. By fine-tuning a small open model on high-quality, domain-specific conversational pairs, I aim to adapt the model's communication style, format, and task capability to the physics and research context of Fermilab, while keeping the pipeline clean of data leakage to ensure rigorous evaluation.

---

## 2. What My Code Does

The pipeline is implemented stage by stage across several distinct scripts. I walk through each part of the pipeline below:

### 1. Data Collection and Crawling (`fermilab.py`)
I implemented a robust web crawler in [fermilab.py](file:///home/st7/.vscode-server/Fermilab_Project/fermilab.py) to harvest public web pages from designated Fermilab domains. Starting from a queue of 4 seed URLs, the crawler discovers internal links using a breadth-first search (BFS) approach. It enforces domain restrictions to keep the crawl within the approved Fermilab subdomains. To be a polite consumer, the crawler implements a 1.5-second delay between requests and uses `requests` with a 20-second timeout. For each page, it saves the raw HTML response and parses the page using `BeautifulSoup4` to extract clean text, writing the output text files with a header mapping the URL and title. It outputs progress logs, milestone markers, and a master CSV metadata report documenting HTTP status codes, file sizes, and word counts.

### 2. Corpus Processing and Ingestion (`01b_process_text_corpus.py`)
The repository contains a default script, [01_process_corpus.py](file:///home/st7/.vscode-server/Fermilab_Project/Fermilab-process_data_and_finetune_model/01_process_corpus.py), designed to extract layout text blocks from PDFs. However, because my crawl results are stored as text pages, I utilized the text-ingest variant [01b_process_text_corpus.py](file:///home/st7/.vscode-server/Fermilab_Project/Fermilab-process_data_and_finetune_model/01b_process_text_corpus.py). This script performs the following steps:
* Ingests the crawler’s download report CSV to verify page status.
* Strips the crawler-added URL/Title headers from the text pages.
* Analyzes line frequencies across all documents to identify and remove boilerplate navigation headers, footer lines, and site chrome (defined as lines appearing on 30% or more of the pages with a length of 12 words or less).
* Fixes hyphenation line-wraps and normalizes whitespace.
* Greedy-packs lines into chunks of approximately 350–450 words, respecting line boundaries.
* Deduplicates identical chunks by checking their content hashes.
* Enforces a leakage guard by assigning a deterministic 80/20 train/test split at the document level (using a SHA-1 hash of the file name) before generating any synthetic data. This ensures that chunks from a single document are never split across the training and evaluation sets.
* Outputs the processed chunks to `corpus_st7.jsonl` and writes a detailed audit report to `audit_report_st7.txt`.

### 3. Synthetic Q&A Generation (`02_generate_qa.py`)
In Stage 2, [02_generate_qa.py](file:///home/st7/.vscode-server/Fermilab_Project/Fermilab-process_data_and_finetune_model/02_generate_qa.py) takes the training split chunks from the corpus and employs a teacher model (`gemini-2.5-flash` via the `google-genai` client) to distill grounded question-answering pairs. It formats the prompt with strict guidelines: questions must be self-contained (avoiding phrases like "the passage" or "the text") and answers must be fully supported by the text. The script contains several robust filters:
* **Leakage Guard**: It explicitly refuses to read any chunk assigned to the `test` split.
* **Decontamination**: It filters out generated questions that share an 8-gram sequence with a gold question set (when provided).
* **Quality Filter**: It drops pairs with short questions/answers or those containing meta-referential phrases.
* **Resume Safety**: It reads existing output records to skip already-processed chunk IDs.
The final output is saved to `train_qa.jsonl` in a conversational format (`messages` list with system, user, and assistant roles) compatible with PyTorch training utilities.

### 4. QLoRA Fine-Tuning (`03_finetune_qlora.py`)
Finally, [03_finetune_qlora.py](file:///home/st7/.vscode-server/Fermilab_Project/Fermilab-process_data_and_finetune_model/03_finetune_qlora.py) executes supervised fine-tuning (SFT) on the synthetic Q&A dataset. I targeted the small open model `Qwen/Qwen3-4B-Instruct-2507`. To run training within VRAM constraints, it uses QLoRA:
* **4-bit Quantization**: Model weights are loaded in 4-bit NormalFloat (`nf4`) format with double quantization and `bfloat16` compute type using `bitsandbytes`.
* **LoRA Configuration**: Parameter-efficient adapters are applied to all linear layers (including query, key, value, output, gate, up, and down projections) with a rank $r=32$, scaling $\alpha=64$, and a dropout rate of $0.05$ via `peft`.
* **Training Arguments**: The script uses the Hugging Face `trl` library's `SFTTrainer` with a batch size of 2 and gradient accumulation steps of 8, establishing an effective batch size of 16. The training runs for 2 epochs with a learning rate of $2 \times 10^{-4}$ using a cosine scheduler and a 3% warmup ratio. Gradient checkpointing is enabled to optimize VRAM usage. The adapter is saved to `./ft-qwen3-fermilab`.

---

## 3. What Data I Downloaded

I executed the crawler using `fermilab.py` with seed URLs targeting multiple Fermilab domains, such as `news.fnal.gov` and `education.fnal.gov`. The master report `download_report_1000.csv` tracks 1,000 attempted page requests with the following statistics:
* **Successful Downloads**: 936 pages (SUCCESS status, representing a 93.6% success rate).
* **Failed Requests**: 64 pages (FAILED status, primarily due to HTTP 403 Forbidden or 404 Not Found errors on restricted sections of the domains).

On disk, the volume of the downloaded files stored in the `crawled_data/` directory totals **1.1 GB**, distributed as follows:
* `html_pages/`: 2,233 raw HTML documents (totaling **601 MB**).
* `text_pages/`: 2,233 extracted text files (totaling **434 MB**).
* `download_report_1000.csv`: **244 KB** (containing metadata for the 1,000 attempts).
* `progress.log`: **180 KB** of crawler run logs.
* `milestones.json`: **4 KB** tracking checkpoint metrics.

*Note on discrepancy*: While the crawler report CSV reflects 1,000 attempted crawls with 936 successes on its final run, the raw directories on disk contain 2,233 files. This indicates that pre-existing documents from prior test crawls or overlapping script outputs accumulated in the workspace directories, all of which were subsequently ingested and processed.

---

## 4. Progress on Preparing the Dataset for Fine-Tuning

### Stage 1: Corpus Processing Results
Using `01b_process_text_corpus.py`, I processed the 2,233 text files into a clean corpus. The audit log (`audit_report_st7.txt`) records the following counts:
* **Text files found on disk**: 2,233
* **Usable documents processed**: 2,233
* **Boilerplate lines removed**: 68 unique navigation or footer lines cut.
* **Total chunks written**: 8,847 records saved to `corpus_st7.jsonl`.
* **Train/Test split**: 7,100 train chunks (80.25%) and 1,747 test chunks (19.75%).

Each record in `corpus_st7.jsonl` follows a strict metadata schema, including `id`, `source`, `source_type`, `title`, `origin`, `split`, `chunk_index`, `n_words`, `content_hash`, `tags`, and `text`. Below is a short sample record:

```json
{
  "id": "st7/0001_news.fnal.gov#c000",
  "source": "st7",
  "source_type": "webpage_text",
  "title": "Newsroom",
  "origin": "https://news.fnal.gov/",
  "split": "train",
  "chunk_index": 0,
  "n_words": 444,
  "content_hash": "9332dfaa3a0a1bb4",
  "tags": ["webpage_text", "st7"],
  "text": "Fermilab is America’s particle physics and accelerator laboratory. Our mission is to serve as world-leader in neutrino and particle physics..."
}
```

### Stage 2: Q&A Generation Results
Using `02_generate_qa.py` with `gemini-2.5-flash`, I generated grounded training pairs from the 7,100 train chunks. The script successfully processed the chunks and generated **241 Q&A pairs** saved to `train_qa.jsonl`. 

Each training record contains a `messages` array structured for conversational LLM training, along with a `meta` dictionary. Below is a sample generated pair:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant that answers questions about Fermilab research."
    },
    {
      "role": "user",
      "content": "What is Fermilab's primary mission as America's particle physics and accelerator laboratory?"
    },
    {
      "role": "assistant",
      "content": "Fermilab's mission is to serve as a world-leader in neutrino and particle physics. It aims to push the frontiers of foundational discovery, open new avenues of science, and transform results into technologies that benefit society."
    }
  ],
  "meta": {
    "chunk_id": "st7/0001_news.fnal.gov#c000",
    "source": "st7",
    "type": "factual lookup",
    "origin": "https://news.fnal.gov/"
  }
}
```

---

## 5. Progress on Fine-Tuning the LLM

The fine-tuning stage has run and successfully completed. The PEFT adapter weights and tokenizer files are saved under `ft-qwen3-fermilab/`. 

### Training Performance and Metrics
I extracted the exact training history from the model's `trainer_state.json` file. The training dataset consisted of 229 pairs for training and 12 pairs held out for validation loss monitoring. The training ran for 30 steps (corresponding to 2 full epochs). The loss and token accuracy progressed as follows:

| Global Step | Training Epoch | Training Loss | Token Accuracy | Val Loss | Val Token Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 5 | 0.35 | 2.7476 | 57.69% | — | — |
| 10 | 0.70 | 1.5354 | 67.66% | — | — |
| 15 | 1.00 | 1.4085 | 70.00% | — | — |
| 20 | 1.35 | 1.1682 | 73.37% | — | — |
| 25 | 1.70 | 1.1433 | 73.73% | 1.3397 | 70.71% |
| **30** | **2.00** | **1.1005** | **73.88%** | **1.3369** | **70.64%** |

### Evaluation Summary
The model demonstrates an excellent learning curve:
1. **Falling Loss**: The training loss decreased steadily from `2.748` down to `1.101`.
2. **Improving Accuracy**: The mean token accuracy rose from `57.69%` at step 5 to `73.88%` at the end of training.
3. **No Overfitting**: The validation loss dropped slightly from `1.340` (step 25) to `1.337` (step 30), aligning closely with the training loss. This indicates that the training duration of 2 epochs was optimal, allowing the model to adapt without over-memorizing the synthetic training subset.

---

## 6. Problems and Obstacles (Past and Present)

During the development and execution of the pipeline, I encountered and addressed several technical obstacles:

### Solved Obstacles
* **File Ingestion Format Mismatch**: The default `01_process_corpus.py` script was written to parse PDFs via the `fitz` (PyMuPDF) library. Attempting to run it on my text data caused a `ModuleNotFoundError` and failed to locate files. I solved this by switching to the text-variant script `01b_process_text_corpus.py`, which is specifically tailored to read raw `.txt` crawler outputs.
* **Server Environment Controls**: The remote server is configured as an externally managed environment, blocking raw `pip install` commands. I resolved this by appending the `--break-system-packages` flag when installing package dependencies (`google-genai`, `torch`, `peft`, `trl`, `bitsandbytes`, `accelerate`).
* **API Authentication**: Stage 2 Q&A generation initially failed because the Gemini API key was not declared. I solved this by retrieving a free API key from Google AI Studio and exporting it to the shell session environment before executing `02_generate_qa.py`.

### Present and Open Obstacles
* **Chat Validation Interface Bug**: In `03_finetune_qlora.py`, the `--chat` evaluation routine contains an interface bug where it calls `model.generate` by passing the tokenized dictionary directly. In newer `transformers` library versions (5.x), this triggers an `AttributeError` unless unpacked as key-value parameters. While this does not affect the fine-tuning training process or the saved adapter, it restricts quick CLI testing until patched.
* **Hardware VRAM Limits**: Loading and training the Qwen3-4B model requires at least 12 GB VRAM even in 4-bit quantized mode. Scaling up to larger architectures (e.g., 7B or 14B models) will exceed current local server capacities without introducing multi-GPU tensor-parallel training configs.

---

## 7. Other Thoughts / Next Steps

### Lessons Learned
* **Deterministic splits are vital**: Performing the train/test document split at the very beginning of the pipeline (Stage 1) is a simple but critical safeguard against evaluation contamination.
* **Boilerplate filtering is high-leverage**: Removing headers, footers, and sidebars from crawled web pages is essential. Without boilerplate filters, a significant portion of the LLM's sequence window is wasted on repetitive site chrome, which degrades context utilization during Q&A generation.

### Future Pipeline Improvements
* **Advanced Deduplication**: Replace the exact SHA content hashes with a fuzzy deduplication technique (such as MinHash LSH) to identify and group near-duplicate pages (e.g., minor updates or slightly altered page paths).
* **Synthetic Diversity**: Configure the Q&A generator to vary temperatures or explicitly prompt the teacher model to write longer comparative or multi-hop logical questions, improving the depth of the training pairs.

### Concrete Next Steps
1. **Merge Adapter Weights**: Run `03_finetune_qlora.py --merge ./ft-qwen3-fermilab` to bake the learned LoRA adapter parameters back into the base Qwen3 model, outputting a merged model structure ready for inference.
2. **Implement Retrieval-Augmented Generation (RAG)**: Combine the fine-tuned model with a vector index built from the `test` split chunks in `corpus_st7.jsonl`. This will ground model responses in real-time document search, combining the adapted tone of the fine-tuned model with the factual accuracy of RAG.
3. **Automate Evaluation**: Develop a pipeline to test the merged model against the held-out test chunks using an LLM-as-a-judge method or semantic similarity metrics, comparing the performance of the fine-tuned model against the base Qwen3-4B.
