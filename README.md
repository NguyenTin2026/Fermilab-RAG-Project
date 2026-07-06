# Fermilab RAG Project

This repository contains a full end-to-end workflow for building a Fermilab-focused question-answering assistant.

In short, you can:
1. crawl public Fermilab web pages,
2. clean and chunk the text,
3. generate synthetic Q&A pairs,
4. fine-tune a language model with QLoRA,
5. compare before/after responses,
6. launch a Streamlit app for interactive use.

The project is organized into two main parts:
- Fermilab Q&A assistant pipeline: crawl data, process corpus, generate training pairs, fine-tune the model, and run the app.
- Discovery track: optional embedding model experiments for retrieval tasks.

---

## 1. Before you start

Make sure you have:
- Python 3.10+ installed
- a GPU with CUDA support if you want to run fine-tuning
- a Gemini API key from Google AI Studio
- internet access for downloading the model and crawling pages

### Recommended folder layout
After cloning, the project root should look like this:
```bash
Fermilab-Project/
├── fermilab.py
├── wait_and_report.py
├── Fermilab-process_data_and_finetune_model/
├── Discovery code and instructions/
└── README.md
```

---

## 2. Clone the repository

```bash
git clone https://github.com/NguyenTin2026/Fermilab-RAG-Project.git
cd Fermilab-RAG-Project
```

---

## 3. Create and activate a Python environment

If you already have a virtual environment, you can reuse it. Otherwise create one:

```bash
python3 -m venv ndo1
source ndo1/bin/activate
```

If you are on a system where pip is blocked by OS packaging rules, use:

```bash
pip install --break-system-packages -U pip
```

---

## 4. Install dependencies

From the repository root:

```bash
pip install --break-system-packages torch transformers trl peft bitsandbytes accelerate datasets google-genai PyMuPDF streamlit pandas
```

If you also want to run the optional discovery/embedding track, install the extra requirements from the subfolder:

```bash
cd "Discovery code and instructions"
pip install --break-system-packages -r requirements.txt
cd ..
```

---

## 5. Set your Gemini API key

Export your API key in the terminal before running any generation step:

```bash
export GEMINI_API_KEY="your_actual_key_here"
```

You can verify it with:

```bash
echo "$GEMINI_API_KEY"
```

> Keep your key private. Do not commit it into the repository.

---

## 6. Run the Fermilab Q&A workflow step by step

### Step 6.1: Crawl the Fermilab pages

Run the crawler from the repository root:

```bash
python3 fermilab.py
```

This will create:
- crawled HTML files in `crawled_data/html_pages/`
- cleaned text files in `crawled_data/text_pages/`
- a crawling report in `crawled_data/download_report_1000.csv`

If you want a summary report after the crawl finishes, run:

```bash
python3 wait_and_report.py
```

### Step 6.2: Process the corpus

Go to the training folder and build the processed corpus:

```bash
cd Fermilab-process_data_and_finetune_model
python3 01b_process_text_corpus.py \
  --text_dir ../crawled_data/text_pages \
  --report ../crawled_data/download_report_1000.csv \
  --out corpus_st7.jsonl
```

This produces a cleaned and chunked corpus file for later training.

### Step 6.3: Generate synthetic Q&A pairs

First, run a self-test to verify the generation script works:

```bash
python3 02_generate_qa.py --selftest
```

Then generate the training data:

```bash
python3 02_generate_qa.py --corpus corpus_st7.jsonl --out train_qa.jsonl
```

This creates the training file used for fine-tuning.

### Step 6.4: Fine-tune the model

This step is the most expensive one and requires GPU memory. It uses QLoRA and trains the base model `Qwen/Qwen3-4B-Instruct-2507`.

Run it inside a terminal session that will stay alive:

```bash
tmux new -s finetune
python3 03_finetune_qlora.py --data train_qa.jsonl --out ./ft-qwen3-fermilab
```

If you prefer to run it directly without tmux, that also works, but tmux helps if your SSH connection drops.

### Step 6.5: Compare base vs fine-tuned answers

After training, compare the base model and the fine-tuned adapter:

```bash
python3 03_finetune_qlora.py --chat ./ft-qwen3-fermilab
```

This prints example answers for a few Fermilab-related questions.

### Step 6.6: Merge the adapter into full weights

If you want a full merged model for simpler inference:

```bash
python3 03_finetune_qlora.py --merge ./ft-qwen3-fermilab
```

The merged model will be saved under:

```bash
Fermilab-process_data_and_finetune_model/ft-qwen3-fermilab/merged
```

---

## 7. Launch the Streamlit app

From the training folder:

```bash
streamlit run app.py --server.fileWatcherType none
```

Open the local URL shown in the terminal.

The app supports:
- chat with the model,
- RAG-based retrieval,
- side-by-side comparison,
- simple metrics and feedback views.

---

## 8. Optional: Discovery/embedding track

If you also want to try the embedding-based retrieval workflow, go to:

```bash
cd "Discovery code and instructions"
```

Useful commands:

```bash
make help
make inspect
make pairs
make dataset
make baseline
make train
make eval
make all
```

This track is optional and is separate from the main Q&A fine-tuning pipeline.

---

## 9. Common issues

### Problem: `ModuleNotFoundError`
Install the missing package with pip.

### Problem: `KeyError: 'GEMINI_API_KEY'`
Make sure you exported the key before running the generation script.

### Problem: out-of-memory during training
Reduce the batch size or use a smaller model configuration.

### Problem: disk full during merge
The merged model can be large. Free some space or remove older checkpoints before running the merge step.

---

## 10. Security note

This repository may contain local data, model outputs, and private environment information. Keep your API keys and any sensitive files out of Git.

The repository already uses a Git ignore file to avoid pushing large artifacts such as:
- virtual environment folders,
- model weights,
- crawled data,
- embeddings and logs.

---

## 11. Expected output files

After a full run, you will usually have:
- `crawled_data/html_pages/`
- `crawled_data/text_pages/`
- `Fermilab-process_data_and_finetune_model/corpus_st7.jsonl`
- `Fermilab-process_data_and_finetune_model/train_qa.jsonl`
- `Fermilab-process_data_and_finetune_model/ft-qwen3-fermilab/`
- `Fermilab-process_data_and_finetune_model/ft-qwen3-fermilab/merged/`
